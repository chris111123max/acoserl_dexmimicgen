"""24 carry and 12 insertion-unload trials; real Actor after handback, no RL updates."""
import os,sys,json,copy,time,multiprocessing as mp
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]

def child(conn,tag):
    os.environ["ASCEND_RT_VISIBLE_DEVICES"]=""
    import robosuite,dexmimicgen,torch
    from acoserl.data import load_demo,metadata
    from acoserl.envs import restore_initial
    from acoserl.bimanual_autoserl import BimanualAutoSERL
    from acoserl.single_demo_rlpd import normalize,IMAGES
    from robosuite.utils.mjcf_utils import IMAGE_CONVENTION_MAPPING
    from robosuite import macros
    from resfit.rl_finetuning.utils.normalization import StateStandardizer,ActionScaler
    sys.path.insert(0,str(ROOT/"scripts/assembly_feasibility"))
    from common import ActionOnly,snapshot,servo,errors,ARMS,SafetyStop
    out=ROOT/"outputs/acoserl_v2"/tag
    out.mkdir(exist_ok=False)
    data=ROOT/"datasets/generated/two_arm_three_piece_assembly.hdf5"
    demo=load_demo(data,"demo_0");meta=metadata(data)
    kwargs=copy.deepcopy(meta["env_kwargs"]);kwargs.pop("env_lang",None)
    kwargs.update(env_name=meta["env_name"],has_renderer=False,has_offscreen_renderer=True,
                  use_camera_obs=False,camera_names=["agentview","robot0_eye_in_hand","robot1_eye_in_hand"],
                  camera_heights=84,camera_widths=84)
    env=robosuite.make(**kwargs)
    norm=json.loads((ROOT/"docs/ascend16/normalization.json").read_text())
    std=StateStandardizer(torch.tensor(norm["state_mean"]),torch.tensor(norm["state_std_population"]),min_std=.1)
    scaler=ActionScaler(torch.tensor(norm["effective_action_min"]),torch.tensor(norm["effective_action_max"]),action_scale=0,min_range_per_dim=.1)
    guards=None;records=[]
    try:
      for kind in ("Carry_A","Carry_B","Contact_B"):
       for trial in range(12):
        if guards:
         for obj,key,method in guards:setattr(obj,key,method)
        restore_initial(env,demo)
        guards=[(env.sim,k,getattr(env.sim,k)) for k in ("set_state","set_state_from_flattened","reset")]+[(env,k,getattr(env,k)) for k in ("reset","reset_from_xml_string")]
        runner=ActionOnly(env)
        m=BimanualAutoSERL(env,intervention_enabled=False);m.manager_cfg["autonomy_enabled"]=False;m.manager_cfg["development_validation"]=True
        m.action_bounds=(scaler.limits.min.numpy(),scaler.limits.max.numpy())
        runner.safety=m.substep_safety
        result=dict(kind=kind,trial=trial,passed=False,actor_steps=0,trial_label_seed=20261029+trial,policy_root_seed=20261029,policy_rng='continuous stream; fixed trial order')
        logs=[];total=0
        def control(policy):
         nonlocal total
         final,decision=m.choose(policy)
         if final is None:raise RuntimeError("refused:"+str(m.stop_reason))
         # Same domain and float32 representation as normalized training bridge.
         normalized=scaler.scale(torch.tensor(final,dtype=torch.float32))
         final=scaler.unscale(normalized).numpy()
         state=runner.step(final);total+=1;after=m.after_step()
         logs.append(dict(time=state["sim_time"],action=final.tolist(),policy=np.asarray(policy).tolist(),
             mode=decision["mode"],decision=copy.deepcopy(decision),after=after,poses=[dict(position=p["position"].tolist(),rotation=p["rotation"].tolist()) for p in state["poses"]],
             objects=state["objects"],grasps=state["actual_grasps"],contact=m.contacts.read(),
             metrics=state["official_metrics"],indices=m.indices()))
         if m.mode=="Safe_Stop":raise RuntimeError(m.stop_reason)
         return state,after
        try:
         ai=0 if kind=="Carry_A" else 1
         alone=trial>=6 and kind!="Contact_B"
         prefix=214 if kind=="Contact_B" else 80
         initial=snapshot(env);hold=copy.deepcopy(initial["poses"])
         for t,action in enumerate(demo["actions"][:prefix]):
          if alone:
           action=action.copy();baseline=servo(env,hold,[-1.,-1.],max_position_step=.002)
           action[(1-ai)*7:(1-ai)*7+7]=baseline[(1-ai)*7:(1-ai)*7+7]
          control(action)
         state=snapshot(env);targets=copy.deepcopy(state["poses"])
         grips=[1. if any(state["actual_grasps"][a].values()) else -1. for a in ARMS]
         mm=3+trial%3
         if kind!="Contact_B":
          axis=trial%2
          targets[ai]["position"][axis]+=(1 if trial%2 else -1)*mm*.001
         for t in range(60):control(servo(env,targets,grips,max_position_step=.002,position_gain=2))
         # Physically create differing passive entry velocities; no state writes.
         if trial%3 and kind!="Contact_B":
          targets[1-ai]["position"][2]+=(trial%3)*.0005
          for t in range(3):control(servo(env,targets,grips,max_position_step=.001,position_gain=2))
         state=snapshot(env);entry=m.passive_hold_entry(ai,state)
         result.update(passive_entry=entry,initial_grasps=state["actual_grasps"],offset_mm=mm,
                       passive_empty=not any(state["actual_grasps"][ARMS[1-ai]].values()))
         allowed,reason=m.start_local(ai,80,contact_mode=kind=="Contact_B",validation=True)
         if not allowed:raise RuntimeError("local_eligibility:"+reason)
         baseline=snapshot(env)
         for t in range(85):
          policy=servo(env,targets,grips,max_position_step=.002)
          state,after=control(policy)
          if after.get("safe_handback"):break
         else:raise RuntimeError("no_handback")
         result["supervisor_handback"]=True
         result["recovery_steps"]=m.age
         result["brake_steps"]=after.get("brake_steps")
         result["final_error"]=after.get("error")
         # Actual fresh Actor outputs, rendered real images; no servo substitute.
         for t in range(5):
          raw=env._get_observations(force_update=True)
          low=np.concatenate([raw[f"robot{i}_{part}"] for i in range(2) for part in ("eef_pos","eef_quat","gripper_qpos")]).astype(np.float32)
          obs={"observation.state":low}
          for camera,key in zip(("agentview","robot0_eye_in_hand","robot1_eye_in_hand"),IMAGES):
           obs[key]=np.transpose(env.sim.render(camera_name=camera,width=84,height=84)[::IMAGE_CONVENTION_MAPPING[macros.IMAGE_CONVENTION]],(2,0,1)).copy()
          obs={k:v.numpy() for k,v in normalize(obs,std).items()}
          conn.send(("act",obs));normalized=conn.recv()
          physical=scaler.unscale(torch.tensor(normalized,dtype=torch.float32)).numpy()
          m.cfg["intervention_enabled"]=True
          state,after=control(physical);result["actor_steps"]+=1
          if m.last_decision["mode"].startswith(("Recover_","Guide_")):raise RuntimeError("Actor_retakeover_during_handback_probe")
          result["actor_shield_steps"]=result.get("actor_shield_steps",0)+int(m.last_decision["mode"].startswith("Constrained_"))
         result["passed"]=True
        except BaseException as e:
         import traceback
         result["error"]=traceback.format_exc()
        result.update(actual_steps=total,physical_writes=runner.physical_writes_attempted,
                      max_passive_m=max((x["after"].get("passive_error",{}).get("position_m",0) for x in logs),default=0))
        (out/(kind+"_"+str(trial)+".json")).write_text(json.dumps(result,indent=2))
        with (out/(kind+"_"+str(trial)+".jsonl")).open("w") as f:
         for row in logs:f.write(json.dumps(row,default=lambda x:x.tolist() if hasattr(x,"tolist") else str(x))+"\n")
        records.append(result);conn.send(("result",result))
      (out/"summary.json").write_text(json.dumps(records,indent=2))
    finally:
      if guards:
       for obj,key,method in guards:setattr(obj,key,method)
      env.close();conn.send(("done",None))

if __name__=="__main__":
    import torch
    mp.set_start_method("spawn")
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument("--tag",required=True);args=parser.parse_args()
    parent,remote=mp.Pipe();process=mp.Process(target=child,args=(remote,args.tag));process.start()
    import torch_npu
    from acoserl.pretrained_visual import make_agent
    from acoserl.single_demo_rlpd import IMAGES
    from resfit.rl_finetuning.config.rlpd import QAgentConfig
    from resfit.rl_finetuning.off_policy.common_utils.utils import eval_mode
    torch.npu.set_device(0);torch.npu.set_option({"ACL_PRECISION_MODE":"must_keep_origin_dtype"})
    torch.npu.config.allow_internal_format=False
    torch.manual_seed(20261029);torch.npu.manual_seed_all(20261029)
    agent=make_agent((3,84,84),(18,),14,IMAGES,QAgentConfig(device="npu:0"),"resnet18_spatial84")
    while process.is_alive():
      if not parent.poll(5):continue
      kind,value=parent.recv()
      if kind=="act":
       data={k:torch.from_numpy(v).unsqueeze(0).to("npu:0") for k,v in value.items()}
       with torch.no_grad(),eval_mode(agent):action=agent.act(data,eval_mode=False,stddev=.1,cpu=True)[0].numpy()
       parent.send(action)
      elif kind=="result":print(json.dumps(value),flush=True)
      elif kind=="done":break
    process.join();assert process.exitcode==0
