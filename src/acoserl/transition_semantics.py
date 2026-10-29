"""Collector boundaries independent of sparse success reward."""
from enum import IntEnum

class ActionSource(IntEnum):
    Policy = 0
    Guide_A = 1
    Guide_B = 2
    Recover_A = 3
    Recover_B = 4
    Recover_Joint = 5
    OriginalDemo = 6

class Boundary(IntEnum):
    NONE = 0
    SUCCESS = 1
    TIME_LIMIT = 2
    PHYSICAL_FAILURE = 3
    ACTION_REFUSED = 4
    PARTIAL_PHYSICS = 5
    COLLECTOR_CUT = 6
    DEMO_SEGMENT_END = 7
    SUPERVISOR_STOP = 8

STAGES = ("unknown", "approach_pregrasp", "grasp_stabilization",
          "independent_carry", "first_assembly_contact",
          "second_assembly_contact", "release_completion", "coupled")

def source_code(mode):
    return int(ActionSource[mode if mode != "Normal" else "Policy"])

def boundary(terminated=False, truncated=False, success=False,
             safety_reason=None, executed=True, partial=False):
    if partial:
        return Boundary.PARTIAL_PHYSICS, False
    if not executed:
        return Boundary.ACTION_REFUSED, False
    if success:
        assert terminated and not truncated
        return Boundary.SUCCESS, False
    if safety_reason:
        if safety_reason.startswith(("recovery_timeout", "guide_handback_", "predicted_unsafe_action")):
            return Boundary.SUPERVISOR_STOP, False
        return Boundary.PHYSICAL_FAILURE, False
    if terminated:
        return Boundary.PHYSICAL_FAILURE, False
    if truncated:
        return Boundary.TIME_LIMIT, True
    return Boundary.NONE, True
