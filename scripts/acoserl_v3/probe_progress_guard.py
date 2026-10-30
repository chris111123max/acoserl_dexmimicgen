"""Original Contact_B_6 prefix plus V3 recovery and saved Actor proposals.
Not the >=20 fresh Actor acceptance: a targeted software/physics debug probe.
"""
import argparse,copy,json,sys,time,traceback
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/"scripts/assembly_feasibility"))
from common import setup,snapshot,save,SafetyStop
from acoserl.bimanual_autoserl import BimanualAutoSERL
from acoserl.progress_guard import velocities
from resfit.rl_finetuning.utils.normalization import ActionScaler

def main():
    p=argparse.ArgumentParser();p.add_argument("--tag",required=True);a=p.parse_args()
    out=ROOT/"outputs/acoserl_v3"/a.tag;out.mkdir(exist_ok=False)
    old=[json.loads(s) for s in (ROOT/"outputs/acoserl_v2/local_physics_v3/Contact_B_6.jsonl").read_text().splitlines()]
    E,D,R=setup();M=BimanualAutoSERL(E,intervention_enabled=False)
    M.manager_cfg["development_validation"]=True
    norm=json.loads((ROOT/"docs/ascend16/normalization.json").read_text())
    scale=ActionScaler(torch.tensor(norm["effective_action_min"]),torch.tensor(norm["effective_action_max"]),action_scale=0,min_range_per_dim=.1)
    M.action_bounds=(scale.limits.min.numpy(),scale.limits.max.numpy())
    logs=[];substeps=[];i=-1;last=None;prefix_error=0.;start=time.monotonic()
    result=dict(passed=False,prefix_steps=274,actor_proposal_source="five_original_saved_NPU_proposals; repeated for debug only",fresh_actor_inferences=0)
    def measured():
        state=snapshot(E);state["velocities"]=velocities(E)
        state["protection"]=M.progress_guard.check(state);return state
    def safety(c):
        if i>=274:
            state=measured();substeps.append(dict(control=i,state=state,contact=c))
        return M.substep_safety(c)
    R.safety=safety
    def control(policy,prefix=False):
        nonlocal i,last,prefix_error
        i+=1
        if 214<=i<279:
            for robot in E.robots:robot.part_controllers["right"].update()
        before=measured();cmd,decision=M.choose(policy)
        if cmd is None:raise RuntimeError("pre-step refusal:"+str(M.stop_reason))
        action=scale.unscale(scale.scale(torch.tensor(cmd,dtype=torch.float32))).numpy()
        R.step(action);after=M.after_step();state=measured();last=after
        if i%50==0:print("control",i,"phase",M.handoff_phase,flush=True)
        if prefix:
            error=max(np.linalg.norm(state["poses"][ai]["position"]-old[i]["poses"][ai]["position"]) for ai in range(2))
            prefix_error=max(prefix_error,error);assert error<1e-6,(i,error)
        if i>=264:logs.append(dict(control=i,before=before,after=state,policy=policy,controller=cmd,executed=action,
                                  decision=copy.deepcopy(decision),recovery_result=after,contact=M.contacts.read()))
        if M.mode=="Safe_Stop":raise RuntimeError(M.stop_reason)
        return after
    try:
        for row in old[:274]:control(np.asarray(row["policy"]),True)
        M.manager_cfg["v3_enabled"]=True
        okay,reason=M.start_local(1,80,contact_mode=True,validation=True);assert okay,reason
        proposals=[np.asarray(r["policy"]) for r in old[279:284]];count=0
        for _ in range(100):
            actor=M.handoff_phase in ("Handoff_Ready","Policy_Guarded")
            policy=proposals[count%len(proposals)] if actor else np.asarray(old[274]["policy"])
            if actor:count+=1
            control(policy)
            if count>=20:break
        result.update(passed=count>=20,actor_saved_proposals=count,
                      target_reached=M.v3_stats["target_reached"],handbacks=M.v3_stats["handbacks"])
    except BaseException:
        result["error"]=traceback.format_exc()
    finally:
        result.update(complete_controls=i+1,physical_writes=R.physical_writes_attempted,
            prefix_max_error_m=prefix_error,stats=M.v3_stats,stop_reason=M.stop_reason,
            handoff_phase=M.handoff_phase,wall_seconds=time.monotonic()-start,last_result=last,
            max_protected_error_m=max((r["after"]["protection"].get("pose_error",{}).get("position_m",0) for r in logs),default=0))
        save(out/"result.json",result);save(out/"controls.json",logs);save(out/"substeps.json",substeps)
        E.close();print(json.dumps(result,default=lambda x:x.tolist() if hasattr(x,"tolist") else str(x)),flush=True)
if __name__=="__main__":main()
