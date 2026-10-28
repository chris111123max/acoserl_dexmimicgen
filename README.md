# Bimanual AutoSERL V1 on DexMimicGen

Reproducible single-demonstration ResFiT RLPD + conservative bimanual AutoSERL integration for **TwoArmThreePieceAssembly/demo_0**. A = robot0/right; B = robot1/left.

This release targets one Ascend 910B3 learner and exactly 16 CPU MuJoCo workers. Workers initialize in four successive batches of four, then all sixteen sample concurrently. UTD is fixed to 1; n-step is 3 with separate env/episode queues. Networks, losses and optimization are upstream ResFiT implementations.

## Scope and current evidence

See [the formal Chinese report](docs/ascend16/FORMAL_DEPLOYMENT_ZH.md) and [publication notes](docs/PUBLIC_RELEASE_ZH.md). At the reported checkpoint, the live experiment had 1,173 real interactions and 149 finite NPU updates. This is training-loop evidence, not assembly success. Frequent Safe_Stop and the lack of effective recovery-phase coverage remain limitations. Historical server acceptance does not authorize a new machine: the public deployment gate defaults to false.

Old diagnosis drivers, smoke reports, unrelated tasks, data, model weights, environments and run logs are intentionally excluded. Shared OSC/contact helpers remain because the current supervisor uses them. Runtime-specific records in the formal report are sanitized historical evidence, not portable installation paths.

## 1. Exact sources and licenses

```bash
git clone https://github.com/chris111123max/acoserl_dexmimicgen.git
cd acoserl_dexmimicgen
python3 scripts/bootstrap_sources.py
```

The four core repositories are pinned as submodules and in [third_party.lock.json](third_party.lock.json). The bootstrap verifies each exact commit and applies only the included ResFiT environment/entry-point compatibility patch. Two additional NPU build dependencies are pinned to exact commits. Read [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md): DexMimicGen and ResFiT impose non-commercial restrictions. AutoSERL is cited as conceptual prior work; its unlicensed source is not redistributed.

## 2. Independent NPU runtime

Use Linux ARM64, Python 3.10, a working compiler and installed Ascend driver/CANN. The recorded combination is CANN 8.0.RC2, PyTorch 2.3.1, torch_npu 2.3.1, TorchRL/TensorDict 0.4.0. Do not infer CANN readiness from an installed driver alone. MuJoCo uses OSMesa on CPUs, not NVIDIA EGL.

Install your distribution's OSMesa/OpenGL/GLib runtime packages (on Debian: libosmesa6, libgl1-mesa-dri, libglib2.0-0, libegl1 and libopengl0). No system libraries or CANN binaries are bundled.

```bash
export CANN_ROOT=/usr/local/Ascend/ascend-toolkit
bash scripts/ascend16/setup_npu.sh
source scripts/ascend16/activate.sh
```

The setup creates only a new .venv_autoserl_npu and refuses to overwrite an existing one. It does not start training. It requires access to PyPI and the pinned public sources. The portable setup recipe is statically checked in this release; it has not been rerun on an independent clean machine. Existing .venv and .venv_resfit are unnecessary for formal NPU training and are not modified.

## 3. Bring the original dataset and derive the local reference

Obtain the official task HDF5 separately under its license, or place your existing file at datasets/generated/two_arm_three_piece_assembly.hdf5. No dataset is distributed here. Expected SHA256: e12cf336e23b362d8c5317b6e195df472d32bb424a797ab1858e8aa20fabe752.

```bash
python scripts/benchmark.py inspect --dataset datasets/generated/two_arm_three_piece_assembly.hdf5
python scripts/benchmark.py extract --dataset datasets/generated/two_arm_three_piece_assembly.hdf5 --demo demo_0
```

Extraction reads only demo_0 and creates a new single-demo reference file without changing the original HDF5. State reconstruction is an offline preparation operation, never a recovery operation. The learner reads exactly the 224 effective transitions defined in docs/bimanual_autoserl/strict_semantics.json. The grasp-only reference config retains precisely the fields the existing window gate uses; it contains no original physical trajectories.

## 4. Confirm this machine, authorize, then launch

Check npu-smi, versions, CPU/memory quota, OSMesa, dataset hash and available storage. Explicitly set passed=true in docs/ascend16/deployment_gate.json only after your own deployment approval. Historical authorizations are not transferred automatically. Configuration is configs/ascend16/formal.json: 16 environments, UTD=1, max 500000 actual interactions or 10 hours; full checkpoints every 25000 and intervention-free evaluation every 50000 (20 episodes, one extra worker while sampling pauses).

```bash
nohup setsid bash scripts/ascend16/launch_formal.sh my_formal_run > formal_launcher.log 2>&1 < /dev/null &
python scripts/ascend16/status.py
python scripts/ascend16/summarize.py
python scripts/ascend16/plot_curves.py
```

Use a unique run tag. All four groups must initialize; failure never silently falls back to fewer workers. Only the learner imports torch_npu. Single-worker Safe_Stop cuts its episode and resets it without fabricating transitions. Replay stores the executed action. Safe_Stop thresholds and the passive-arm dynamic hold gate are unchanged.

## Persistence and resume

Outputs stay under outputs/ascend16/RUN; logs under logs/ascend16. A complete checkpoint contains model/targets, optimizer/scheduler state, offline/online replay, normalization, pending n-step rows and RNG/worker metadata. To resume, use train.py --mode formal --tag NEW_RUN --utd 1 --resume PATH_TO_COMPLETE_CHECKPOINT after activation. Resume starts new MuJoCo episodes and legally truncates pending real tails; it is not bit-exact mid-episode physics restoration. Integrated sixteen-worker checkpoint restoration is still an explicitly unverified limitation in the report. Run directories are never automatically deleted.
