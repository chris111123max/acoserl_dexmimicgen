"""Bounded FORWARD demo grasp guidance; no P0 return or passive-arm Hold.

Only the active arm's demonstrated approach/closure segment is owned. Entry
requires a nearby, reached pregrasp reference, unchanged target object and
independent contact. Actual grasp events (not elapsed time) unlock transport.
This task-specific extension is not a physical qualification certificate.
"""
from collections import deque
import numpy as np
from scipy.spatial.transform import Rotation


def relative_grasp(state, ai, obj):
    pose = state['poses'][ai]
    rotation = np.asarray(pose['rotation'])
    item = state['objects'][obj]
    return (rotation.T @ (np.asarray(item['position']) - pose['position']),
            rotation.T @ Rotation.from_quat(
                np.roll(item['quaternion_wxyz'], -1)).as_matrix())


def grasp_candidate(manager, state, ai):
    """Select only a locally reached frame in the recorded pregrasp interval."""
    points = manager.points['arms']['AB'[ai]]
    if manager.completed[ai] or any(state['actual_grasps']['AB'[ai]].values()):
        return None, 'active_arm_not_empty_or_already_completed'
    if not points.get('validated'):
        return None, 'reference_interval_unvalidated'
    end = min(int(manager.windows[ai].end), int(points['first_close_action']))
    start = int(points['p0'])
    if end < start:
        return None, 'reference_window_has_not_reached_pregrasp'
    pose = state['poses'][ai]
    candidates = []
    for index in range(start, end + 1):
        goal = manager.pose(ai, index)
        distance = float(np.linalg.norm(goal['position'] - pose['position']))
        rotation = float(Rotation.from_matrix(goal['rotation'] @ pose['rotation'].T).magnitude())
        if distance <= .005 and rotation <= .035:
            candidates.append((distance + .01 * rotation, index))
    if not candidates:
        return None, 'outside_local_pregrasp_capture'
    assessment = manager.contact_assessment(state, manager.contacts.read())
    if assessment['category'] != 'Independent' or not assessment['allowed']:
        return None, 'not_independent_pregrasp'
    contact = manager.contacts.read()
    if any('AB'[ai] in (e['a'], e['b']) for e in contact['edges']):
        return None, 'active_arm_already_in_contact'
    from acoserl.object_geometry import object_error
    obj_error = object_error(state['objects'][points['object']],
                             manager.ref_object(points['object'], points['p0']))
    if obj_error['position_m'] > .003 or obj_error['rotation_rad'] > .035:
        return None, 'target_not_at_demonstrated_pick_pose'
    robot = manager.env.robots[ai]
    site = manager.env.sim.model.site_id2name(robot.eef_site_id['right'])
    speed = float(np.linalg.norm(manager.env.sim.data.get_site_xvelp(site)))
    angular = float(np.linalg.norm(manager.env.sim.data.get_site_xvelr(site)))
    if not np.isfinite([speed, angular]).all() or speed > .05 or angular > .25:
        return None, 'pregrasp_capture_motion_too_fast'
    index = min(candidates)[1]
    if not (index < points['first_contact'] <= points['first_grasp'] <= points['p1']):
        return None, 'inconsistent_reference_events'
    return index, 'local_forward_pregrasp_capture'


class ForwardGraspGuide:
    """Event-gated single-arm reference tracking plus bounded physical closure."""
    def __init__(self, manager, active, start):
        self.active = int(active)
        p = manager.points['arms']['AB'[active]]
        self.obj = p['object']
        self.index = int(start)
        self.end = int(p['p1'])
        self.first_grasp = int(p['first_grasp'])
        self.first_close = int(p['first_close_action'])
        self.controls = 0
        self.max_controls = 80
        self.history = deque(maxlen=manager.cfg['stable_grasp_window_steps'])
        self.ready = False
        self.phase = 'approach'
        self.last_target_index = self.index
        self.gripper_owned = False

    def action(self, manager, state, policy):
        """The other arm is copied exactly; the normal final-action shield follows."""
        ai = self.active
        held = bool(state['actual_grasps']['AB'[ai]][self.obj])
        # Never advance into demonstrated transport without a measured grasp.
        waiting = self.index >= self.first_grasp and not held
        endpoint = self.index >= self.end
        reference = manager.pose(ai, self.index)
        targets = [dict(p) for p in state['poses']]
        targets[ai] = reference
        grips = [float(policy[6]), float(policy[13])]
        nominal = (np.zeros(7) if endpoint else
                   np.asarray(manager.trace['actions'][self.index, ai*7:ai*7+7]).copy())
        # This is a forward task demonstration, not a reverse/regrasp planner.
        closing = self.index >= self.first_close
        if closing and not endpoint and nominal[6] <= 0:
            raise ValueError('forward grasp reference contains unexpected opening')
        self.gripper_owned = True
        grips[ai] = 1.0 if closing or endpoint else float(nominal[6])
        correction = manager.command(targets, grips)[ai*7:ai*7+6]
        result = np.asarray(policy, dtype=float).copy()
        # Feed-forward action at s_i plus bounded feedback to s_i. When waiting
        # for actual grasp or stability, only feedback and real closure remain.
        result[ai*7:ai*7+6] = correction + (0.0 if waiting or endpoint else nominal[:6])
        result[ai*7+6] = grips[ai]
        self.last_target_index = self.index if waiting or endpoint else self.index + 1
        manager.targets[ai] = manager.pose(ai, self.last_target_index)
        manager.grips[ai] = grips[ai]
        self.phase = ('stabilize' if endpoint else 'wait_actual_grasp' if waiting
                      else 'close' if closing else 'approach')
        if hasattr(manager, 'action_bounds'):
            # Do not clip or relabel the other arm; its own shield owns that.
            lo, hi = manager.action_bounds
            sl = slice(ai*7, ai*7+7)
            result[sl] = np.clip(result[sl], lo[sl], hi[sl])
        return result

    def observe(self, manager, state):
        """Called exactly once AFTER a complete real physical control."""
        self.controls += 1
        ai = self.active
        held = bool(state['actual_grasps']['AB'[ai]][self.obj])
        target = manager.pose(ai, self.last_target_index)
        position = float(np.linalg.norm(state['poses'][ai]['position'] - target['position']))
        rotation = float(Rotation.from_matrix(
            state['poses'][ai]['rotation'] @ target['rotation'].T).magnitude())
        reached = position <= .005 and rotation <= .035
        if reached and self.index < self.end:
            # Crossing the actual grasp state is event-gated, not clock-gated.
            if self.last_target_index < self.first_grasp or held:
                self.index = self.last_target_index
            else:
                self.index = min(self.last_target_index, self.first_grasp)
        if held:
            self.history.append(relative_grasp(state, ai, self.obj))
        else:
            self.history.clear()
        stable = len(self.history) == self.history.maxlen
        if stable:
            x0, r0 = self.history[0]
            stable = (max(float(np.linalg.norm(x-x0)) for x, _ in self.history)
                      < manager.cfg['relative_grasp_motion_m'] and
                      max(float(Rotation.from_matrix(r @ r0.T).magnitude())
                          for _, r in self.history) <= manager.cfg['passive_object_rotation_drift_rad'])
        command = np.asarray(manager.env.robots[ai].gripper['right'].current_action)
        closed = command.shape == (2,) and np.allclose(command, [-1., 1.])
        self.ready = bool(self.index == self.end and held and stable and closed and reached)
        return dict(phase=self.phase, reference_index=self.index,
                    target_index=self.last_target_index, actual_grasp=held,
                    relative_stable=bool(stable), ready=self.ready,
                    controls=self.controls, position_error_m=position,
                    rotation_error_rad=rotation)
