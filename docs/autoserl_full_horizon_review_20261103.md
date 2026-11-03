# Full-horizon dual-arm AutoSERL: implementation and self-review (2026-11-03)

Status: **work in progress; not ready for formal retraining or merge**. Development branch only. Production `main` and running server training files are unchanged.

## Scope implemented
- Extend per-arm AutoSERL Guide's reference domain through carry, assembly, controlled release, and retreat using the original recorded `demo_0` TCP trajectory, retaining per-arm sliding windows. Do not add object-relative action control or ContactGuide.
- Preserve existing S0/P0/P1 recovery anchors and explicitly forbid unverified recovery replay across assembly contact.
- Permit a Guide only at task-expected object/object contact, with existing robot contact force, workspace, shadow-substep and cross-arm coupling checks still authoritative.
- Gate reference-gripper close on measured pregrasp TCP alignment and no assembly coupling. Gate release on actual object grasp, expected assembly contact, official placement geometry, and A-before-B assembly prerequisite. Keep Guide gripper ownership active throughout release completion.
- Prevent Guide release on TCP pose convergence alone while owned gripper events remain incomplete. Keep per-arm Policy ownership independent; force-preserve the actual grasp when a Guide is moving a held piece.
- Encode Guide gripper ownership in the existing per-dimension action provenance fields in Online Replay. Do not introduce a new TensorDict key (Online/Offline batch schemas must remain identical).
- Extend candidate intervention tail from 32 to 96 to cover the bounded 80-step Guide. Admit safe physically complete terminal successes even if the terminal step preempts a separate Policy handback cycle.
- Increment controller/collector/demo protocol strings to explicitly reject unsafe checkpoint migration.

## Static-review findings and fixes already applied
1. A transition from no-grasp to grasp changed the permitted Guide reference domain and could terminate a real close Guide incorrectly; the active close event now retains the relevant original reference interval.
2. Release removes the grasp before the strict success streak; the active Guide retains the release interval to avoid an accidental reference reset.
3. Successful terminal episodes can end before an explicit handback. Such physically completed successes now qualify the pending intervention tail while failed or incomplete episodes do not.
4. A Guide must not surrender the gripper immediately after the grasp disappears: it could reclose during the release-completion interval. Owned open action now persists until handback or termination.
5. New per-arm gripper attribution MUST reuse the existing TensorDict schema. A proposed extra key was removed during review.
6. The B release Guide persists over the first official task-completion observation to allow strict multi-step success verification.
7. A close Guide cannot remain permanently anchored to the pre-close TCP once the actual grasp is established; it now advances to the original measured first-grasp reference while retaining the takeover ID.

## Blocking items before training
- **No runtime tests conducted here:** server has not been accessed; MuJoCo/robosuite, NPU and TorchRL execution are unverified. Require source-bound targeted tests of both arms' full reference trajectory, gripper ownership, assisted-success terminal Demo admissions, stop/refusal semantics, and disabled-intervention equivalence. These are targeted functional tests, not smoke tests.
- **Checkpoint migration is intentionally disallowed:** original v2 controller and v3 Demo checkpoints require a separately reviewed migration or a fresh run.
- **Buffer capacity has NOT been increased yet.** Existing 3xRGB84 current/next uint8 observations imply ~25.4 GB of raw image payload per 200k-row replay buffer. Two such buffers plus copies/checkpoints can exceed available RAM and disk; a measured memory-efficient storage design is required before enabling 200k/200k.
- **Speed optimizations have NOT been applied yet.** Existing profiling lacks confirmed live server timings; changing shadow simulation, collection scheduling or network update overlap without evidence could corrupt the 80 N/substep safety guarantee or the true UTD counter.
- **Contact and release require physical verification.** A reference index is not evidence of a collision-safe continuous trajectory. Existing limits must never be disabled to make a Guide succeed.
- **Candidate tail is 96, not unbounded.** Long recoveries (up to 350 steps) can still expire earlier candidates; observe expired_candidate_rows and qualify storage before changing.
- **No `qpos`/`qvel` mid-episode teleportation; no new reward or learned controller; keep true sparse strict success.**

## Review checklist for Codex (read-only on this branch)
1. Compare source SHA and `main`; inspect exactly 4 changed source files and this note.
2. Verify Python AST syntax and imports, 14D actions, 7D per-arm gripper sign, no edits to disabled Policy action.
3. Verify no unsafe release or pose-only false handback; A and B closing/opening across real state transitions.
4. Check legitimate object/object contacts versus forbidden contact, actual 80 N and all substep preview constraints, and no unqualified cross-contact rewind.
5. Verify `make_row` ownership mask for 6D Guide, 7D Guide, Recovery, BoundaryGuard, both-arm mixes, original Demo, and successful terminal episode.
6. Check `torch.cat` for Online/Offline TensorDict schema match and 3-step n-step terminal semantics.
7. Verify failure paths preserve logs and checkpoint compatibility rejection.
8. Report exact source line, input state, expected/observed behavior, and evidence for any defect; do not push/merge or start formal retraining.

## Sources
- Official AutoSERL `serl_robot_infra/franka_env/envs/wrappers.py`: sliding intervention, safety recovery, termination.
- Official AutoSERL `examples/train_rlpd.py`: 50/50 online/demo replay ingestion.
- DexMimicGen `two_arm_three_piece_assembly.py`: official placement and release success criteria.
