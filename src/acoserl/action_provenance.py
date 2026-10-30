"""Per-arm/per-dimension execution provenance; metadata only, never Actor input."""
import numpy as np
# Stable replay ids: Policy0, Guide1, Recover2, SafeHold3, SafetyProjection4,
# OriginalDemo5, Rejected6, Unknown7.
def describe(source,policy,executed,active_arm=None,quality=False):
    policy=np.asarray(policy);executed=np.asarray(executed)
    pose=[0,0];grip=[0,0];expert=np.zeros(14,dtype=bool)
    if source==6:
        pose=grip=[5,5];expert[:]=True
    elif source in (1,2):
        ai=0 if source==1 else 1
        if active_arm in ('A','B'):ai='AB'.index(active_arm)
        # Guide replaces ONLY the deviating arm. The other seven Policy
        # dimensions remain autonomous, never mislabeled as static Hold.
        pose[ai]=1
        grip[ai]=1 if abs(policy[7*ai+6]-executed[7*ai+6])>1e-6 else 0
        other=1-ai
        pose[other]=4 if np.any(abs(policy[7*other:7*other+6]-executed[7*other:7*other+6])>1e-6) else 0
        grip[other]=4 if abs(policy[7*other+6]-executed[7*other+6])>1e-6 else 0
        if quality:
            expert[7*ai:7*ai+6]=True
            expert[7*ai+6]=bool(grip[ai]==1)
    elif source in (3,4,5,7,8):
        ai={3:0,4:1,5:1,7:0,8:1}[source]
        if active_arm in ('A','B'):ai='AB'.index(active_arm)
        pose=[3,3];grip=[3,3];pose[ai]=grip[ai]=2
        if quality:expert[7*ai:7*ai+7]=True
    elif source in (9,10,11):
        for ai in range(2):
            pose[ai]=4 if np.any(abs(policy[7*ai:7*ai+6]-executed[7*ai:7*ai+6])>1e-6) else 0
            grip[ai]=4 if abs(policy[7*ai+6]-executed[7*ai+6])>1e-6 else 0
    elif source==12:pose=grip=[8,8]
    elif source!=0:pose=grip=[7,7]
    return dict(pose_sources=pose,gripper_sources=grip,expert_action_mask=expert.tolist(),
                policy_dimension_mask=(abs(policy-executed)<=1e-6).tolist())
