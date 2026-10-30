"""One-control-period OSC nominal response, not a contact dynamics certificate."""
import numpy as np

def predict(env,ai,action):
 ctl=env.robots[ai].part_controllers['right']
 assert ctl.input_type=='delta' and ctl.input_ref_frame=='base'
 site=env.sim.model.site_id2name(env.robots[ai].eef_site_id['right'])
 dt=1./float(env.control_freq);assert np.allclose(ctl.kp,ctl.kp[0]),'nominal screen requires isotropic Panda OSC gains';omega=np.sqrt(np.asarray(ctl.kp));response=1-(1+omega*dt)*np.exp(-omega*dt)
 v=np.asarray(env.sim.data.get_site_xvelp(site));w=np.asarray(env.sim.data.get_site_xvelr(site))
 scaled=np.asarray(action[7*ai:7*ai+6])*np.asarray(ctl.output_max);origin=np.asarray(ctl.origin_ori)
 inertia=v*dt*np.exp(-omega[:3]*dt)
 dp=response[:3]*(origin@scaled[:3])+inertia;dw=response[3:]*(origin@scaled[3:])+w*dt*np.exp(-omega[3:]*dt)
 return dict(displacement=dp,command_displacement=origin@scaled[:3],angular_displacement=dw,response=response,linear_inertia=inertia,control_dt_s=dt,physics_dt_s=float(env.sim.model._model.opt.timestep),estimate='fixed_OSC_one_period_nominal_not_contact_certificate')

def command_for_displacement(env,ai,action,desired):
 prediction=predict(env,ai,action);ctl=env.robots[ai].part_controllers['right'];result=np.asarray(action).copy()
 result[7*ai:7*ai+3]=np.asarray(ctl.origin_ori).T@((desired-prediction['linear_inertia'])/prediction['response'][:3])/np.asarray(ctl.output_max)[:3]
 return result

def project_independent_workspace(env, ai, action, position, lower, upper,
                                  risk_margin, action_bounds=None,
                                  contact_normal=None, reserve=.003):
    """Project independent free-space motion under raw and nominal OSC bounds.

    The raw OSC setpoint and predicted one-control-period displacement MUST
    both fit the workspace. This is not a dynamics/contact certificate.
    """
    original=np.asarray(action,dtype=float)
    pred=predict(env,ai,original)
    response=np.asarray(pred['response'][:3],dtype=float)
    inertia=np.asarray(pred['linear_inertia'],dtype=float)
    current=np.asarray(pred['command_displacement'],dtype=float)
    position=np.asarray(position,dtype=float)
    lo=np.asarray(lower,dtype=float)+reserve-position
    hi=np.asarray(upper,dtype=float)-reserve-position
    if (not all(np.isfinite(x).all() for x in
                (original,response,inertia,current,position,lo,hi))
            or np.any(response<=1e-9)):
        return None,'nonfinite_or_invalid_osc_response'
    # d_nominal = response * u_world + inertia. Enforce BOTH d_nominal
    # and raw u_world, including the existing near-boundary outward screen.
    minimum=np.maximum(lo,(lo-inertia)/response)
    maximum=np.minimum(hi,(hi-inertia)/response)
    low_edge=position-np.asarray(lower)<risk_margin
    high_edge=np.asarray(upper)-position<risk_margin
    minimum[low_edge]=np.maximum(minimum[low_edge],
                                np.maximum(0.,-inertia[low_edge]/response[low_edge]))
    maximum[high_edge]=np.minimum(maximum[high_edge],
                                 np.minimum(0.,-inertia[high_edge]/response[high_edge]))
    if np.any(minimum>maximum+1e-10):
        return None,'no_feasible_workspace_braking_command'
    if np.any(minimum>maximum):
        midpoint=(minimum+maximum)/2.
        minimum=np.minimum(minimum,midpoint)
        maximum=np.maximum(maximum,midpoint)
    projected=np.clip(current,minimum,maximum)
    if contact_normal is not None:
        normal=np.asarray(contact_normal,dtype=float)
        length=float(np.linalg.norm(normal))
        if not np.isfinite(length) or length<1e-8:
            return None,'invalid_gripper_contact_normal'
        normal=normal/length
        coefficient=normal*response
        limit=-float(normal@inertia)-1e-8
        best=float(coefficient@np.where(coefficient>=0,minimum,maximum))
        if best>limit+1e-10:
            return None,'no_feasible_nonapproach_command'
        if float(coefficient@projected)>limit:
            low_lambda=0.;high_lambda=1.
            def clipped(lam):
                return np.clip(current-lam*coefficient,minimum,maximum)
            while float(coefficient@clipped(high_lambda))>limit and high_lambda<1e12:
                high_lambda*=2.
            if float(coefficient@clipped(high_lambda))>limit:
                return None,'nonapproach_projection_unresolved'
            for _ in range(48):
                midpoint=(low_lambda+high_lambda)/2.
                if float(coefficient@clipped(midpoint))>limit:
                    low_lambda=midpoint
                else:
                    high_lambda=midpoint
            projected=clipped(high_lambda)
    ctl=env.robots[ai].part_controllers['right']
    result=original.copy()
    result[7*ai:7*ai+3]=np.asarray(ctl.origin_ori).T@projected/np.asarray(ctl.output_max)[:3]
    if action_bounds is not None:
        result=np.clip(result,*action_bounds)
    checked=predict(env,ai,result)
    raw=np.asarray(checked['command_displacement'])
    nominal=np.asarray(checked['displacement'])
    if (np.any(raw<lo-1e-8) or np.any(raw>hi+1e-8)
        or np.any(nominal<lo-1e-8) or np.any(nominal>hi+1e-8)
        or np.any(low_edge & ((raw< -1e-8)|(nominal< -1e-8)))
        or np.any(high_edge & ((raw>1e-8)|(nominal>1e-8)))):
        return None,'command_bounds_invalidate_safe_projection'
    if contact_normal is not None and float(normal@nominal)>1e-8:
        return None,'contact_approach_risk_after_projection'
    return result,'workspace_raw_and_nominal_projected'
