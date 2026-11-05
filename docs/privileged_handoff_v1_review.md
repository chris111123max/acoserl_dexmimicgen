# Privileged Handoff V1: implementation contract and review

This branch is an **unvalidated implementation candidate**. GitHub source-level review does not establish robot/MuJoCo safety. No training, NPU execution, server changes, or Git merge were performed.

## Scope

- Preserve the existing 3 RGB + 18D proprioception observation contract and complete 14D Actor actions; simulated object state is used only in the supervisor and is never added to the Actor's input.
- Preserve the existing six-dimensional AutoSERL Guide trajectory command, pretrained visual encoder, ten-Q RLPD learner, sparse reward, and ActionOnly physical safety checks.
- Add one 1D per-arm privileged gripper expert that uses the **existing** recorded demo_0 close/release relative pose, actual MuJoCo object transforms and grasp verification. No demo step-number replay is used to trigger a live event.
- Close when a current object-relative grasp pose matches the demonstrated physical pre-close reference; latch Close until a stable physical grasp is confirmed. Hold Close through carry; when the carried piece is at its demonstrated assembly position relative to base (A) or assembled piece_1 (B), latch Open until physical grasp is released.
- A positive Panda gripper command means close, negative means open. Treat a nonzero same-sign Actor command as compatible with the expert's desired effective action; a neutral command only agrees at the proper gripper actuator end stop. The expert overrides **only** the gripper dimension on a disagreement.
- Leave the other arm's motion and gripper untouched by a unilateral expert.
- Guide may independently own 6D on an arm while privileged Handoff owns 1D. Existing pre-grasp P0 recovery retains its own action authority and no privileged Handoff override is applied during Recovery.
- Privileged phase transitions and intervention statistics commit only after a *complete* real control step. A refused or interrupted physics step cannot create successful expert records.
- Online Replay stores the complete 14D executed action. Retain existing TensorDict shape, with 1D expert action mask and gripper ownership code 8. A Handoff segment enters Dynamic Demo only after an official strict successful terminal episode. Guide candidates still require verified handback; safety-projected rows never qualify.
- Successful episode triggers AutoSERL intervention termination when **Guide takeovers (one per episode event) + privileged Handoff disagreement steps (per arm, per completed control) + pre-grasp Recovery starts** is *strictly less than 10*. A safety-projected success cannot trigger the unlock. The global capability disables intervention only for subsequent control cycles/episodes.
- Keep step-based intervention statistics for diagnosis but do not use them as the new termination metric.

## Known limitations / acceptance blocks

1. The minimum physical Handoff reproduction yielded A PASS, B PASS, C FAIL under a three-step pause. This branch does NOT claim to fix C. The initial pick alignment reuses a documented 5 mm / 0.035 rad local grasp capture threshold, not the overly narrow single-point window from the failed run. Physical performance needs separate qualification.
2. The live relative-pose alignment needs direct confirmation against the user's actual demo_0/controller convention and two-arm full physical runs. Tests added here check pure geometric logic, not true MuJoCo grasp/contact safety.
3. Source protocol, collector protocol, and dynamic demo protocol changed. Existing checkpoints must NOT silently resume against this branch; start a new run only after approval or separately validated migration.
4. Pending candidate demo tail increased to 256 to preserve a complete 229-control official trajectory; evaluate CPU memory overhead with 16 collectors before formal training.
5. No automatic post-grasp regrasp is implemented. Only the original initial-grasp P0 recovery is eligible from new stage selection.
6. Current code's BoundaryGuard and physical safety refusal remain in place, but any safety-projected episode is ineligible to trigger the low-intervention termination.
7. The simple baseline's Guide is still pre-assembly scoped; this change does not automatically guarantee assisted complete assembly. Running an episode through Handoff is not evidence of independent 6D success.
8. Handoff intervention can be disabled by the global AutoSERL capability; there is no separate adaptive turn-off probability or new network, reward, or BC loss.

## Codex source-bound review before any production use

- Confirm branch SHA and worktree cleanliness, diff against main; do not overwrite uncommitted server files.
- Python compile check for changed Python files and target tests. Run only focused `tests/test_privileged_handoff.py`; no general smoke suite, no long rollout.
- Review gripper sign using actual loaded PandaGripper and ActionScaler physical values, including zero and actuator endpoint behavior.
- Review exact per-arm 6D ownership, Guide/Handoff concurrency, Recovery skip, Safety projection/no-override on disabled Policy, and no privileged observation leakage.
- Verify actual source of `snapshot()['objects']`, reference rotation ordering, official termination and object/base relative pose frames.
- Review make_row masks, dynamic demo terminal admission, bounded tail and original/online n-step replay schema; explicitly reject terminal after incomplete physics.
- Verify Guide + Handoff + Recovery event counts, strict 9 PASS / 10 FAIL threshold, boundary-projected non-autonomous exclusion, propagation of global capability across all workers, and checkpoint restore incompatibility.
- Report failures and exact source lines; do not edit training state, run long training, push, or merge.
