"""Bounded calibration plus held-out validation of passive captured-pose holds.

25 predeclared physical cases, no Actor or RL, no intermediate state setters.
The certificate is an empirical envelope near tested configurations, not global.
"""
import os,json,copy,time,argparse,traceback,multiprocessing as mp
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]

def specs():
    result=[]
    for domain,prefix,arms in (("free",0,(0,1)),("carry",80,(0,1)),("contact",214,(0,))):
        for passive in arms:
            for split,amps in (("calibration",(0.,.0005,.001)),("validation",(.0003,.0007))):
                for j,amp in enumerate(amps):
                    result.append(dict(domain=domain,prefix=prefix,passive=passive,
                        split=split,amplitude_m=amp,axis=j%3 if split=="calibration" else (j+1)%3,
                        sign=1 if split=="calibration" else -1))
    return result

def run(item):
    tag,item_id,spec=item
    for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):os.environ[key]="1"
    os.environ["ASCEND_RT_VISIBLE_DEVICES"]=""
    import sys,torch
    torch.set_num_threads(1);sys.path.insert(0,str(ROOT/"scripts/assembly_feasibility"))
    from common import setup,snapshot,poses,objects,save,SafetyStop,errors,ARMS
    from acoserl.bimanual_autoserl import LegacyBimanualAutoSERL
    from acoserl.object_geometry import object_error
    E,D,R=setup();M=LegacyBimanualAutoSERL(E,intervention_enabled=False)
    passive=spec["passive"];active=1-passive;logs=[];samples=[];complete=0;entry=None
    result=dict(case=item_id,spec=spec,complete=False,physical_writes=0)
    baseline=None;force_peak=0.;eef_peak=0.;rotation_peak=0.;object_peak=0.;first_peak=0.;object_rotation_peak=0.
    def safety(c):
        nonlocal force_peak,eef_peak,rotation_peak,object_peak,first_peak,object_rotation_peak
        force_peak=max(force_peak,c["max_robot_contact_force_N"])
        if force_peak>80:return "robot_force_limit"
        if baseline is None:return None
        p=poses(E);o=objects(E);error=errors(p[passive],baseline["poses"][passive])
        eef_peak=max(eef_peak,error["position_m"]);rotation_peak=max(rotation_peak,error["rotation_rad"])
        for name,held in baseline["actual_grasps"][ARMS[passive]].items():
            if held:
                oe=object_error(o[name],baseline["objects"][name])
                object_peak=max(object_peak,oe["position_m"])
                object_rotation_peak=max(object_rotation_peak,oe["rotation_rad"])
        if baseline["official_metrics"]["first_piece_assembled"]:
            first_peak=max(first_peak,object_error(o["piece_1"],baseline["objects"]["piece_1"])["position_m"])
        samples.append(dict(time=float(E.sim.data.time),passive_eef_error=error,held_object_peak_m=object_peak,held_object_rotation_peak_rad=object_rotation_peak,first_piece_peak_m=first_peak,max_force_N=c["max_robot_contact_force_N"]))
        if eef_peak>.002:return "passive_pose_changed"
        if object_peak>.003:return "passive_object_changed"
        if rotation_peak>.03:return "passive_rotation_changed"
        if object_rotation_peak>.035:return "passive_object_rotation_changed"
        if first_peak>.002:return "completed_piece1_displaced"
        return None
    R.safety=safety
    def step(action,phase):
        nonlocal complete
        state=R.step(action);complete+=1
        if phase!="prefix":logs.append(dict(phase=phase,action=action,state=snapshot(E),contact=R.contact.read()))
        return state
    try:
        if spec.get("recorded_prefix"):
            old=[json.loads(x) for x in (ROOT/spec["recorded_prefix"]).read_text().splitlines()]
            maximum=0.
            for i,row in enumerate(old[:140]):
                if i>=80:
                    for robot in E.robots:robot.part_controllers["right"].update()
                current=step(np.asarray(row["action"],dtype=np.float32),"prefix")
                maximum=max(maximum,max(np.linalg.norm(p["position"]-ref["position"]) for p,ref in zip(current["poses"],row["poses"])))
            result["prefix_max_error_m"]=maximum;assert maximum<1e-6,("recorded prefix mismatch",maximum)
        else:
            for action in D["actions"][:spec["prefix"]]:step(action,"prefix")
        s=snapshot(E);targets=copy.deepcopy(s["poses"])
        grips=[1. if any(s["actual_grasps"][a].values()) else -1. for a in ARMS]
        settle=0 if spec.get("recorded_prefix") else (60 if spec["domain"]=="contact" else 20)
        for _ in range(settle):step(M.command(targets,grips),"settle")
        targets[passive]["position"][spec["axis"]]+=spec["sign"]*spec["amplitude_m"]
        for _ in range(spec.get("drive_controls",3)):step(M.command(targets,grips),"drive")
        baseline=snapshot(E);entry=M.passive_hold_entry(active,baseline)
        targets=copy.deepcopy(baseline["poses"])
        targets[active]["position"][2 if spec["domain"]=="contact" else 0]+=.003
        for _ in range(spec.get("hold_controls",80)):step(M.command(targets,grips),"hold")
        result["complete"]=True
    except BaseException:result["error"]=traceback.format_exc()
    finally:
        result.update(physical_writes=R.physical_writes_attempted,complete_controls=complete,
            entry=entry,entry_state=baseline,nominal_position_m=None if entry is None else entry["predicted_peak_position_m"],
            actual_peak_position_m=eef_peak,actual_peak_rotation_rad=rotation_peak,
            actual_peak_held_object_m=object_peak,actual_peak_held_object_rotation_rad=object_rotation_peak,first_piece_peak_m=first_peak,max_force_N=force_peak)
        out=ROOT/"outputs/acoserl_v3"/tag/f"case_{item_id:02d}"
        save(out/"result.json",result);save(out/"controls.json",logs);save(out/"substeps.json",samples);E.close()
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--tag",required=True);p.add_argument("--carry-rotation-audit",action="store_true");p.add_argument("--late-carry-b",action="store_true");p.add_argument("--late-in-domain-validation",action="store_true");a=p.parse_args()
    out=ROOT/"outputs/acoserl_v3"/a.tag;out.mkdir(exist_ok=False)
    cases=specs()
    if a.carry_rotation_audit:
        cases=[{**s,"hold_controls":16 if s["passive"]==1 else 80} for s in cases if s["domain"]=="carry"]
    if a.late_carry_b:
        cases=[{**s,"domain":"carry_late","hold_controls":16,"recorded_prefix":"outputs/acoserl_v2/local_physics_v3/Carry_A_0.jsonl"} for s in cases if s["domain"]=="carry" and s["passive"]==1]
    if a.late_in_domain_validation:
        cases=[dict(domain="carry_late",prefix=80,passive=1,split="validation",amplitude_m=amp,
            axis=axis,sign=sign,hold_controls=16,drive_controls=1,
            recorded_prefix="outputs/acoserl_v2/local_physics_v3/Carry_A_0.jsonl")
            for amp,axis,sign in ((.00015,0,1),(.0002,1,-1))]
    (out/"predeclared_cases.json").write_text(json.dumps(cases,indent=2))
    results=[]
    with mp.get_context("spawn").Pool(4) as pool:
        for r in pool.imap_unordered(run,[(a.tag,i,s) for i,s in enumerate(cases)]):
            results.append(r);print(json.dumps({k:r[k] for k in ("case","spec","complete","actual_peak_position_m")}),flush=True)
    (out/"summary.json").write_text(json.dumps(results,indent=2,default=lambda x:x.tolist() if hasattr(x,"tolist") else str(x)))
