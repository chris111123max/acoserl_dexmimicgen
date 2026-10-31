"""OSC action screens: full control-interval trajectory, not endpoint only.

This is an explicit nominal model, not a safety certificate for MuJoCo contact
physics. ActionOnly's live/preview substep checks remain authoritative.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize


def _controller(env, ai):
    ctl = env.robots[ai].part_controllers['right']
    if ctl.input_type != 'delta' or ctl.input_ref_frame != 'base':
        raise ValueError('workspace guard requires delta/base OSC')
    kp = np.asarray(ctl.kp, dtype=float)
    if kp.shape != (6,) or np.any(kp <= 0) or not np.allclose(kp, kp[0]):
        raise ValueError('workspace guard requires positive isotropic OSC gains')
    if getattr(ctl,'impedance_mode','fixed')!='fixed' or not np.allclose(np.asarray(ctl.kd),2*np.sqrt(kp)):
        raise ValueError('workspace guard requires fixed critically damped OSC')
    if not np.allclose(ctl.output_max, -np.asarray(ctl.output_min)):
        raise ValueError('workspace guard requires symmetric OSC output scaling')
    return ctl


def _motion(u, v, omega, times):
    t = np.asarray(times, dtype=float).reshape(-1, 1)
    decay = np.exp(-t * omega)
    response = 1.0 - (1.0 + t * omega) * decay
    return response * u + t * decay * v


def _extrema(u, v, omega, horizon):
    """Exact coordinate extrema of the critically damped nominal trajectory."""
    end = _motion(u, v, omega, [horizon])[0]
    low, high = np.minimum(0.0, end), np.maximum(0.0, end)
    denominator = omega * omega * u - omega * v
    t = np.divide(-v, denominator, out=np.full(3, -1.0),
                  where=np.abs(denominator) > 1e-14)
    valid = (t > 0) & (t < horizon)
    for axis in np.flatnonzero(valid):
        value = _motion(u, v, omega, [t[axis]])[0, axis]
        low[axis], high[axis] = min(low[axis], value), max(high[axis], value)
    return low, high


def predict(env, ai, action):
    ctl = _controller(env, ai)
    action = np.asarray(action, dtype=float)
    if action.shape != (14,) or not np.isfinite(action).all():
        raise ValueError('expected finite 14D controller action')
    site = env.sim.model.site_id2name(env.robots[ai].eef_site_id['right'])
    dt = 1.0 / float(env.control_freq)
    omega = np.sqrt(np.asarray(ctl.kp, dtype=float))
    response = 1.0 - (1.0 + omega * dt) * np.exp(-omega * dt)
    v = np.asarray(env.sim.data.get_site_xvelp(site), dtype=float).copy()
    w = np.asarray(env.sim.data.get_site_xvelr(site), dtype=float).copy()
    origin = np.asarray(ctl.origin_ori, dtype=float)
    scaled = action[7*ai:7*ai+6] * np.asarray(ctl.output_max)
    command, angular = origin @ scaled[:3], origin @ scaled[3:]
    inertia = v * dt * np.exp(-omega[:3] * dt)
    dp = response[:3] * command + inertia
    dw = response[3:] * angular + w * dt * np.exp(-omega[3:] * dt)
    lo, hi = _extrema(command, v, omega[:3], dt)
    return dict(displacement=dp, command_displacement=command,
                angular_displacement=dw, response=response,
                linear_inertia=inertia, linear_velocity=v, angular_velocity=w,
                trajectory_min=lo, trajectory_max=hi,
                control_dt_s=dt, physics_dt_s=float(env.sim.model._model.opt.timestep),
                estimate='critical_OSC_full_interval_nominal_not_certificate')


def command_for_displacement(env, ai, action, desired):
    """Legacy inverse; final-action guards must still validate its result."""
    prediction = predict(env, ai, action)
    ctl = _controller(env, ai)
    result = np.asarray(action, dtype=float).copy()
    world = (np.asarray(desired) - prediction['linear_inertia']) / prediction['response'][:3]
    result[7*ai:7*ai+3] = np.asarray(ctl.origin_ori).T @ world / np.asarray(ctl.output_max)[:3]
    return result


def project_independent_workspace(env, ai, action, position, lower, upper,
                                  risk_margin, action_bounds=None,
                                  contact_normal=None, reserve=.003):
    """Nearest feasible 3D command; preserve all other action dimensions.

    Includes raw setpoint, inertia, full-period motion, controller input limits
    and optional approach half-spaces. Iteratively adds continuous extrema to
    the sampled constraints. A missing feasible command is a refusal, never a
    zero-action fallback. No grasp/contact recovery is authorized here.
    """
    action = np.asarray(action, dtype=float)
    p = np.asarray(position, dtype=float)
    lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    pred = predict(env, ai, action)
    ctl = _controller(env, ai)
    if not all(np.isfinite(x).all() for x in (p, lower, upper)):
        return None, 'nonfinite_workspace'
    if np.any(p < lower) or np.any(p > upper):
        return None, 'already_outside_workspace'
    # Do not demand instantaneous teleportation to the reserve boundary.
    # Existing physical bounds are unchanged; reserve applies to future motion.
    lo = np.minimum(lower + reserve - p, 0.0)
    hi = np.maximum(upper - reserve - p, 0.0)
    B = np.asarray(ctl.origin_ori) @ np.diag(np.asarray(ctl.output_max)[:3])
    omega = np.sqrt(np.asarray(ctl.kp)[:3])
    v, T = pred['linear_velocity'], pred['control_dt_s']
    physics_dt = pred['physics_dt_s']
    times = list(np.linspace(0.0, T, max(2, int(np.ceil(T / physics_dt))) + 1)[1:])
    start = action[7*ai:7*ai+3].copy()
    bounds = (-np.ones(3), np.ones(3)) if action_bounds is None else (
        np.asarray(action_bounds[0])[7*ai:7*ai+3],
        np.asarray(action_bounds[1])[7*ai:7*ai+3])
    normals = []
    if contact_normal is not None:
        for n in np.asarray(contact_normal, dtype=float).reshape(-1, 3):
            norm = float(np.linalg.norm(n))
            if not np.isfinite(norm) or norm < 1e-8:
                return None, 'invalid_contact_normal'
            normals.append(n / norm)
    for _ in range(4):
        mats, lows, highs = [B], [lo.copy()], [hi.copy()]
        # A near-boundary raw setpoint may not direct farther outward, but
        # finite stopping motion is allowed INSIDE the unchanged workspace.
        raw_lo, raw_hi = lo.copy(), hi.copy()
        raw_lo[p-lower < risk_margin] = np.maximum(raw_lo[p-lower < risk_margin], 0.0)
        raw_hi[upper-p < risk_margin] = np.minimum(raw_hi[upper-p < risk_margin], 0.0)
        lows[0], highs[0] = raw_lo, raw_hi
        for t in times:
            decay = np.exp(-omega*t)
            r = 1-(1+omega*t)*decay
            inertia = v*t*decay
            mats.append(r[:, None]*B); lows.append(lo-inertia); highs.append(hi-inertia)
        for n in normals:
            mats.extend([(n@B)[None, :], (n@(pred['response'][:3, None]*B))[None, :]])
            lows.extend([np.array([-np.inf]), np.array([-np.inf])])
            highs.extend([np.array([0.0]), np.array([-float(n@pred['linear_inertia'])])])
        A, lb, ub = np.vstack(mats), np.concatenate(lows), np.concatenate(highs)
        if np.any(lb > ub):
            return None, 'infeasible_workspace_constraints'
        candidate = np.clip(start, *bounds)
        if not (np.all(A@candidate >= lb-1e-10) and np.all(A@candidate <= ub+1e-10)):
            result = minimize(lambda x: .5*float((x-start)@(x-start)), candidate,
                              jac=lambda x: x-start, method='SLSQP',
                              bounds=Bounds(*bounds), constraints=LinearConstraint(A, lb, ub),
                              options={'maxiter': 60, 'ftol': 1e-12})
            if not result.success:
                return None, 'no_feasible_workspace_braking_command'
            candidate = result.x
        final = action.copy(); final[7*ai:7*ai+3] = candidate
        checked = predict(env, ai, final)
        if (np.all(checked['trajectory_min'] >= lo-1e-8) and
                np.all(checked['trajectory_max'] <= hi+1e-8) and
                np.all(A@candidate >= lb-1e-8) and np.all(A@candidate <= ub+1e-8)):
            return final, 'workspace_full_interval_projected'
        u = checked['command_displacement']
        denominator = omega*omega*u-omega*v
        roots = np.divide(-v, denominator, out=np.full(3, -1.0),
                          where=np.abs(denominator)>1e-14)
        times.extend(float(t) for t in roots if 0 < t < T)
    return None, 'continuous_workspace_extremum_unresolved'
