"""Private-data next-substep workspace check and bounded diagnostic records.

No prediction is written into the live simulator and no predicted transition
enters Replay. A rejection ends collection; it is not a rollback. Force values
remain sampled stopping signals, never a promise that force cannot overshoot.
"""
from __future__ import annotations
import copy
import time
import numpy as np
import mujoco


def refresh_positions(model, data):
    # mj_step2 integrates qpos but does not necessarily refresh site_xpos.
    # These calls update derived kinematics only, never qpos/qvel/act/time.
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    mujoco.mj_comVel(model, data)


class WorkspacePreview:
    """One private MjData, reused when the installed binding supports copying."""
    def __init__(self, env, manager):
        self.env, self.manager = env, manager
        self.model = env.sim.model._model
        self.sites = [r.eef_site_id['right'] for r in env.robots]
        self.lower = manager.positions.min(axis=0) - manager.cfg['free_workspace_margin_m']
        self.upper = manager.positions.max(axis=0) + manager.cfg['free_workspace_margin_m']
        self.shadow = None
        self.stats = dict(calls=0, refusals=0, seconds=0.0,
                          backend='mj_copyData' if hasattr(mujoco, 'mj_copyData') else 'MjData.__copy__')

    def before(self, method):
        start = time.perf_counter()
        live = self.env.sim.data._data
        # Only read live state. An installed global callback would make a
        # shadow invocation side-effectful; reject instead of invoking twice.
        for name in ('get_mjcb_control', 'get_mjcb_passive'):
            fn = getattr(mujoco, name, None)
            if fn is not None and fn() is not None:
                return 'workspace_preview_custom_callback_unsupported', {'method': method}
        copier = getattr(mujoco, 'mj_copyData', None)
        if copier is None:
            self.shadow = copy.copy(live)
        else:
            if self.shadow is None:
                self.shadow = mujoco.MjData(self.model)
            try:
                copier(self.shadow, self.model, live)
            except TypeError:
                # Some Python binding releases infer the model from MjData.
                copier(self.shadow, live)
        if method == 'step2':
            mujoco.mj_step2(self.model, self.shadow)
        elif method == 'step':
            mujoco.mj_step(self.model, self.shadow)
        else:
            raise ValueError('unsupported integration entry: '+str(method))
        predicted_force=0.0
        owners=self.manager.contacts.owners
        for k in range(self.shadow.ncon):
            contact=self.shadow.contact[k]
            a,b=owners[int(contact.geom1)],owners[int(contact.geom2)]
            if a==b or not ({a,b} & {'A','B'}): continue
            force=np.zeros(6)
            mujoco.mj_contactForce(self.model,self.shadow,k,force)
            predicted_force=max(predicted_force,float(max(force[0],0.)))
        refresh_positions(self.model, self.shadow)
        points = self.shadow.site_xpos[self.sites].copy()
        bad = np.any((points < self.lower) | (points > self.upper), axis=1)
        reason = None
        if not np.isfinite(points).all() or not np.isfinite(self.shadow.qvel).all():
            reason = 'workspace_preview_nonfinite'
        elif predicted_force>self.manager.cfg['robot_force_stop_N']:
            reason='force_preview_limit'
        elif np.any(bad):
            reason = 'workspace_preview_exit_'+'AB'[int(np.flatnonzero(bad)[0])]
        self.stats['calls'] += 1
        self.stats['refusals'] += int(reason is not None)
        self.stats['seconds'] += time.perf_counter()-start
        return reason, dict(method=method, live_time=float(live.time),
                            predicted_time=float(self.shadow.time),
                            predicted_positions=points.tolist(),
                            predicted_robot_force_N=predicted_force,
                            lower=self.lower.tolist(), upper=self.upper.tolist(),
                            backend=self.stats['backend'],
                            live_physics_advanced=False)

    def audit(self):
        return dict(self.stats)
