"""Directed real-physics pregrasp/Guide qualification with full image Replay.

Every preparation, offset and recovery is executed by MuJoCo control actions.
The behavior controller is scripted for qualification, not a learned RL policy.
"""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import torch
from acoserl.parallel_autoserl import WorkerFactory, AutoSERLWorker
from acoserl.bimanual_autoserl import snapshot, servo, ARMS, errors
from acoserl.two_point_supervisor import paper_direction
from acoserl.ascend_data import raw_row, replay
from acoserl.isolated_nstep import IsolatedNStep
from acoserl.dynamic_demo import DynamicDemo

ROOT = Path(__file__).resolve().parents[2]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--case", choices=["Recover_A","Recover_B","Guide_A","Guide_B","free","risk"], required=True)
    p.add_argument("--tag", required=True)
    args = p.parse_args()
    out = ROOT / "outputs/acoserl_foundation" / args.tag
    out.mkdir(exist_ok=False)
    norm = json.loads((ROOT / "docs/ascend16/normalization.json").read_text())
    worker = AutoSERLWorker(WorkerFactory(0, 20261029, norm, str(out / "worker"), True,
                                         manager_config={"autonomy_enabled":False}))
    m = worker.bridge.manager
    result = dict(case=args.case, behavior_controller="scripted_qualification_only",
                  passed=False, formal_training=False)
    assembler = IsolatedNStep()
    online = replay(512, 128)
    dynamic = DynamicDemo(capacity=512)
    records = []
    try:
        worker.reset()
        previous = worker.last
        def step(action):
            nonlocal previous
            normalized = worker.scaler.scale(torch.tensor(action, dtype=torch.float32)).numpy()
            obs, reward, term, trunc, info = worker.step(normalized)
            payload = json.loads(info["payload"])
            payload["behavior_controller"] = "scripted_qualification_only"
            records.append(payload)
            if payload["executed"]:
                physical = worker.scaler.unscale(torch.tensor(payload["executed_action"])).numpy()
                assert np.max(abs(physical-np.array(payload["executed_action_physical"]))) < 2e-6
                assert obs["observation.state"].shape == (18,)
                assert all(v.shape == (3,84,84) and v.dtype == np.uint8
                           for k,v in obs.items() if k != "observation.state")
                row = raw_row({k:torch.from_numpy(v) for k,v in previous.items()},
                              payload["executed_action"],
                              {k:torch.from_numpy(v) for k,v in obs.items()},
                              reward,term,trunc,0,worker.episode,payload["frame"],metadata=payload)
                for item in assembler.append(row):
                    online.add(item)
                dynamic.observe(row,
                    handback=bool((payload.get("recovery_result") or {}).get("safe_handback")),
                    safe=not bool(payload.get("supervisor_stop_reason")))
            else:
                for item in assembler.close(0, bootstrap=payload["bootstrap_allowed"],
                                            boundary_kind=payload["boundary_kind"]):
                    online.add(item)
                dynamic.reject(0, partial=payload.get("partial_physics_step",False))
            previous = obs
            if term or trunc:
                result["stop_reason"] = payload.get("supervisor_stop_reason")
                raise RuntimeError("episode ended: " + str(result["stop_reason"]))
            return payload
        initial = snapshot(worker.w.env)
        target = copy.deepcopy(initial["poses"])
        grips = [-1.,-1.]
        if args.case == "free":
            for t in range(25):
                step(servo(worker.w.env,target,grips))
            result["passed"] = worker.frame == 25 and m.mode != "Safe_Stop"
        else:
            ai = 1 if args.case.endswith("B") else 0
            prefix = m.points["arms"][ARMS[ai]]["p0"]
            if args.case == "Recover_B":
                prefix = 58
            m.cfg["intervention_enabled"] = False
            if args.case != "risk":
                for action in m.trace["actions"][:prefix]:
                    step(action)
                state = snapshot(worker.w.env)
                target = copy.deepcopy(state["poses"])
                grips = [1. if any(state["actual_grasps"][a].values()) else -1. for a in ARMS]
            else:
                prefix = 0
                hi = m.positions[:,0].max(0) + m.cfg["free_workspace_margin_m"]
                target[0]["position"][2] = hi[2] - .006
            if args.case.startswith("Guide"):
                # Choose a 28mm physically reached offset whose measured window
                # distance and paper direction will trigger ordinary guidance.
                chosen = None
                pts = m.positions[:prefix+1,ai]
                for x in (-1,0,1):
                    for y in (-1,0,1):
                        for z in (-1,0,1):
                            vector = np.array([x,y,z],dtype=float)
                            if not np.linalg.norm(vector):
                                continue
                            position = target[ai]["position"] + .028 * vector / np.linalg.norm(vector)
                            dist = np.linalg.norm(pts-position,axis=1)
                            pipoint = int(np.argmin(dist))
                            cpoint = int(np.argmin(np.linalg.norm(m.positions[:,ai]-position,axis=1)))
                            allowed,_,_ = paper_direction(m.positions[:,ai],cpoint,position,pipoint)
                            if dist[pipoint] > .022 and allowed:
                                chosen = position
                                break
                        if chosen is not None:break
                    if chosen is not None:break
                assert chosen is not None
                target[ai]["position"] = chosen
                result["requested_offset_position"] = chosen.tolist()
            for t in range(100):
                step(servo(worker.w.env,target,grips,max_position_step=.006,position_gain=2))
            m.reset((prefix,prefix))
            m.cfg["intervention_enabled"] = True
            baseline = snapshot(worker.w.env)
            result["baseline"] = dict(poses=[dict(position=p["position"].tolist(),rotation=p["rotation"].tolist()) for p in baseline["poses"]],
                                      grasps=baseline["actual_grasps"])
            start = worker.frame
            hold_targets = copy.deepcopy(target)
            handback = False
            for t in range(400):
                proposed = servo(worker.w.env,hold_targets,grips,max_position_step=.006,position_gain=2)
                if args.case == "risk":
                    # Actual OSC positive z in world expressed using existing base.
                    ctl = worker.w.env.robots[0].part_controllers["right"]
                    proposed[:3] = ctl.origin_ori.T @ np.array([0,0,.012]) / ctl.output_max[:3]
                payload = step(proposed)
                if args.case == "risk":
                    decision = payload["auto_intervention"]
                    if decision.get("trigger_reason") == "workspace_risk":
                        result["risk_detected_before_exit"] = True
                        result["passed"] = payload["action_source"] == "Recover_A"
                        break
                if args.case != "risk" and (payload.get("recovery_result") or {}).get("safe_handback"):
                    assert payload["action_source"] == args.case
                    handback = True
                    break
            result["supervised_segment_steps"] = worker.frame - start
            if args.case != "risk":
                assert handback
                held = snapshot(worker.w.env)["actual_grasps"][ARMS[ai]][m.points["arms"][ARMS[ai]]["object"]]
                assert held == args.case.startswith("Recover")
                assert len(dynamic.buffer) > 0
                hold_state = snapshot(worker.w.env)
                hold_targets = copy.deepcopy(hold_state["poses"])
                hold_grips = [1. if any(hold_state["actual_grasps"][a].values()) else -1. for a in ARMS]
                for t in range(5):
                    payload = step(servo(worker.w.env,hold_targets,hold_grips,max_position_step=.006,position_gain=2))
                    assert payload["action_source"] == "Normal"
                result["passed"] = True
                result["handback_policy_steps"] = 5
            result["target_active_grasp"] = snapshot(worker.w.env)["actual_grasps"][ARMS[ai]]
        for item in assembler.truncate(0):
            online.add(item)
        dynamic.save(out / "dynamic_checkpoint")
        result.update(actual_steps=worker.frame, online_rows=len(online),
                      dynamic_rows=len(dynamic.buffer), dynamic_stats=dynamic.stats,
                      modes=sorted(set(r["action_source"] for r in records)),
                      physical_writes=worker.runner.physical_writes_attempted,
                      intervention_events=m.events, decision_stats=m.decision_stats,
                      max_passive_error_m=max(((r.get("recovery_result") or {}).get(
                          "passive_error",{}).get("position_m",0) for r in records),default=0))
        assert result["physical_writes"] == 0
    except BaseException:
        import traceback
        result["error"] = traceback.format_exc()
    finally:
        worker.close()
        result["records_file"] = str(out / "worker/env00.jsonl")
        (out / "result.json").write_text(json.dumps(result,indent=2))
        print(json.dumps(result),flush=True)

if __name__ == "__main__":
    main()
