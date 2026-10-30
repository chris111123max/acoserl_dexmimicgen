"""V3 exact contact prefix, physical recovery and fresh NPU Actor continuation.

One bounded directed case for debugging. This is not spontaneous online contact
recovery and does not replace the frozen 12+4 acceptance or gradient integration.
"""
import os,json,copy,sys,time,traceback,argparse,multiprocessing as mp,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]

def child(conn,tag,case,kind,short_carry=False):
    os.environ["ASCEND_RT_VISIBLE_DEVICES"]=""
    import torch,robosuite,dexmimicgen
    torch.set_num_threads(1)
    from acoserl.data import load_demo,metadata
    from acoserl.envs import restore_initial
    from acoserl.bimanual_autoserl import BimanualAutoSERL
    from acoserl.progress_guard import velocities,progress_state
    from acoserl.single_demo_rlpd import normalize,IMAGES
    from acoserl.ascend_data import offline,replay,raw_row
    from acoserl.dynamic_demo import DynamicDemo
    from acoserl.isolated_nstep import IsolatedNStep
    from robosuite.utils.mjcf_utils import IMAGE_CONVENTION_MAPPING
    from robosuite import macros
    from resfit.rl_finetuning.utils.normalization import StateStandardizer,ActionScaler
    sys.path.insert(0,str(ROOT/"scripts/assembly_feasibility"))
    from common import ActionOnly,snapshot,save,SafetyStop
    out=ROOT/"outputs/acoserl_v3"/tag;out.mkdir(exist_ok=False)
    source=ROOT/f"outputs/acoserl_v2/local_physics_v3/{kind}_{case}.jsonl"
    old=[json.loads(s) for s in source.read_text().splitlines()]
    prefix_count=next(i for i,r in enumerate(old) if r["mode"].startswith(("Recover_Carry_","Recover_Joint")))
    if short_carry:
        assert kind in ("Carry_A","Carry_B")
        prefix_count=110
    first_actor=next(i+1 for i,r in enumerate(old) if r["after"].get("safe_handback"))
    servo_start=214 if kind=="Contact_B" else 80
    active=0 if kind=="Carry_A" else 1
    data=ROOT/"datasets/generated/two_arm_three_piece_assembly.hdf5"
    demo=load_demo(data,"demo_0");meta=metadata(data);kwargs=copy.deepcopy(meta["env_kwargs"])
    kwargs.pop("env_lang",None)
    kwargs.update(env_name=meta["env_name"],has_renderer=False,has_offscreen_renderer=True,
        use_camera_obs=False,camera_names=["agentview","robot0_eye_in_hand","robot1_eye_in_hand"],camera_heights=84,camera_widths=84)
    E=robosuite.make(**kwargs);restore_initial(E,demo);R=ActionOnly(E)
    M=BimanualAutoSERL(E,intervention_enabled=False);M.manager_cfg["development_validation"]=True
    norm=json.loads((ROOT/"docs/ascend16/normalization.json").read_text())
    scale=ActionScaler(torch.tensor(norm["effective_action_min"]),torch.tensor(norm["effective_action_max"]),action_scale=0,min_range_per_dim=.1)
    std=StateStandardizer(torch.tensor(norm["state_mean"]),torch.tensor(norm["state_std_population"]),min_std=.1)
    M.action_bounds=(scale.limits.min.numpy(),scale.limits.max.numpy())
    bank={mode:DynamicDemo(capacity=128,admission_mode=mode) for mode in ("official","safe_handback")}
    on=replay(128,128);stream=IsolatedNStep(3,.99)
    logs=[];substeps=[];frames=[];i=-1;complete=0;count=0;prefix_error=0.;start=time.monotonic()
    result=dict(passed=False,fresh_actor_inferences=0,optimizer_updates=0,directed_prefix=True,
        controller_fingerprint=M.v3_fingerprint(),source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [ROOT/"src/acoserl"/n for n in ("manager_v3.py","progress_guard.py","local_recovery.py","dynamic_demo.py","ascend_data.py")]})
    def measure():
        s=snapshot(E);s["velocities"]=velocities(E);s["protection"]=M.progress_guard.check(s);return s
    def observe():
        raw=E._get_observations(force_update=True)
        low=np.concatenate([raw[f"robot{ai}_{part}"] for ai in range(2) for part in ("eef_pos","eef_quat","gripper_qpos")]).astype(np.float32)
        obs={"observation.state":low}
        for camera,key in zip(("agentview","robot0_eye_in_hand","robot1_eye_in_hand"),IMAGES):
            obs[key]=np.transpose(E.sim.render(camera_name=camera,width=84,height=84)[::IMAGE_CONVENTION_MAPPING[macros.IMAGE_CONVENTION]],(2,0,1)).copy()
        return normalize(obs,std)
    def safety(contact):
        if i>=prefix_count:
            s=progress_state(E);substeps.append(dict(control=i,sim_time=float(E.sim.data.time),protection=M.progress_guard.check(s),velocities=velocities(E),objects=s["objects"],contact=contact))
        return M.substep_safety(contact)
    R.safety=safety
    def control(policy,prefix=False,obs=None):
        nonlocal i,complete,prefix_error
        i+=1
        if servo_start<=i<first_actor:
            for robot in E.robots:robot.part_controllers["right"].update()
        before=measure();cmd,decision=M.choose(policy)
        if cmd is None:raise RuntimeError("pre-step refusal:"+str(M.stop_reason))
        action_n=scale.scale(torch.tensor(cmd,dtype=torch.float32));action=scale.unscale(action_n).numpy()
        R.step(action);complete+=1;after=M.after_step();state=measure()
        if prefix:
            error=max(np.linalg.norm(state["poses"][ai]["position"]-old[i]["poses"][ai]["position"]) for ai in range(2))
            prefix_error=max(prefix_error,error);assert error<1e-6,(i,error)
        if i>=prefix_count-10:
            logs.append(dict(control=i,before=before,after=state,policy=policy,controller=cmd,executed=action,decision=copy.deepcopy(decision),recovery_result=after,contact=M.contacts.read()))
        if obs is not None:
            nxt=observe();normalized_policy=scale.scale(torch.tensor(policy,dtype=torch.float32)).numpy()
            proposal=np.asarray(decision.get("intervention_action_proposed",cmd),dtype=np.float32)
            meta=dict(auto_intervention=decision,policy_action=normalized_policy,
                intervention_action_normalized=scale.scale(torch.from_numpy(proposal)).numpy(),
                task_quality=after.get("task_quality",{}),takeover_id=decision.get("takeover_id",0),
                stage=decision.get("stage","second_assembly_contact"),supervisor_stop_reason=M.stop_reason,success=False)
            row=raw_row(obs,action_n,nxt,0.,M.mode=="Safe_Stop",False,0,0,i,metadata=meta)
            frames.append(row)
            for r in stream.append(row):on.add(r)
            for d in bank.values():d.observe(row,handback=after.get("safe_handback",False),safe=M.stop_reason is None)
        if i%50==0:conn.send(("progress",dict(control=i,phase=M.handoff_phase)))
        if M.mode=="Safe_Stop":raise RuntimeError(M.stop_reason)
        return after
    try:
        for row in old[:prefix_count]:control(np.asarray(row["policy"]),True)
        M.manager_cfg["v3_enabled"]=True
        okay,reason=M.start_local(active,80,contact_mode=kind=="Contact_B",validation=True);assert okay,reason
        for _ in range(120):
            if time.monotonic()-start>600:raise RuntimeError("bounded_case_deadline")
            obs=observe();actor=M.handoff_phase in ("Handoff_Ready","Policy_Guarded") or M.v3_stats["handbacks"]>0
            if actor:
                conn.send(("act",{k:v.numpy() for k,v in obs.items()}));normalized=conn.recv()
                if normalized is None:raise RuntimeError("parent_actor_failed")
                policy=scale.unscale(torch.tensor(normalized,dtype=torch.float32)).numpy();count+=1
            else:policy=np.asarray(old[prefix_count]["policy"])
            control(policy,obs=obs)
            if count>=20:break
        result.update(passed=count>=20 and M.v3_stats["handbacks"]>0 and M.stop_reason is None)
    except BaseException:result["error"]=traceback.format_exc()
    finally:
        for r in stream.truncate(0):on.add(r)
        for name,d in bank.items():
            d.close_episode(0,True,6);d.save(out/name)
        if len(on):on.dumps(out/"online")
        torch.save(frames,out/"raw_transitions.pt")
        result.update(complete_controls=complete,attempted_controls=i+1,fresh_actor_inferences=count,
            prefix_max_error_m=prefix_error,prefix_controls=prefix_count,shortened_hold_fixture=short_carry,case_kind=kind,case_index=case,stats=M.v3_stats,stop_reason=M.stop_reason,physical_writes=R.physical_writes_attempted,
            wall_seconds=time.monotonic()-start,online_rows=len(on),dynamic={k:dict(rows=len(d.buffer),stats=d.stats) for k,d in bank.items()},
            max_protected_error_m=max((r["after"]["protection"].get("pose_error",{}).get("position_m",0) for r in logs),default=0))
        save(out/"result.json",result);save(out/"controls.json",logs);save(out/"substeps.json",substeps)
        E.close();conn.send(("result",result));conn.send(("done",None))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--tag",required=True);p.add_argument("--case",type=int,default=6);p.add_argument("--kind",choices=["Contact_B","Carry_A","Carry_B"],default="Contact_B");p.add_argument("--short-carry",action="store_true");a=p.parse_args()
    mp.set_start_method("spawn");parent,remote=mp.Pipe();process=mp.Process(target=child,args=(remote,a.tag,a.case,a.kind,a.short_carry));process.start()
    try:
        import torch,torch_npu
        from acoserl.pretrained_visual import make_agent
        from acoserl.single_demo_rlpd import IMAGES
        from resfit.rl_finetuning.config.rlpd import QAgentConfig
        from resfit.rl_finetuning.off_policy.common_utils.utils import eval_mode
        torch.set_num_threads(2);torch.npu.set_device(0)
        torch.npu.set_option({"ACL_PRECISION_MODE":"must_keep_origin_dtype"});torch.npu.config.allow_internal_format=False
        torch.manual_seed(20261029);torch.npu.manual_seed_all(20261029)
        agent=make_agent((3,84,84),(18,),14,IMAGES,QAgentConfig(device="npu:0"),"resnet18_spatial84")
        while process.is_alive():
            if not parent.poll(5):continue
            kind,value=parent.recv()
            if kind=="act":
                data={k:torch.from_numpy(v).unsqueeze(0).to("npu:0") for k,v in value.items()}
                with torch.no_grad(),eval_mode(agent):action=agent.act(data,eval_mode=False,stddev=.1,cpu=True)[0].numpy()
                parent.send(action)
            elif kind in ("result","progress"):print(json.dumps(value),flush=True)
            elif kind=="done":break
        process.join();assert process.exitcode==0
    except BaseException:
        if process.is_alive():parent.send(None)
        raise
