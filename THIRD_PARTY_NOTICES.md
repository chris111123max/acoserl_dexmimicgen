# Third-party sources, attribution and restrictions

Exact commits and source URLs are in third_party.lock.json and the four submodule gitlinks. Upstream source is fetched separately; its copyright and license notices are retained. This project does not relicense upstream works.

| Component | Upstream | Recorded license | Handling |
|---|---|---|---|
| robosuite | ARISE-Initiative/robosuite | MIT, with an Apache-2.0 notice for included MuJoCo bindings | Exact core submodule; original notices preserved |
| DexMimicGen | NVlabs/dexmimicgen | NVIDIA non-commercial research/evaluation license | Exact core submodule; no data redistributed |
| ResFiT | amazon-far/residual-offpolicy-rl | CC BY-NC 4.0 | Exact core submodule plus explicitly marked environment/entry compatibility patch; no network/loss/optimizer rewrite |
| LeRobot reader | huggingface/lerobot | Apache-2.0 | Exact core submodule |
| TensorDict | pytorch/tensordict | MIT | Exact optional NPU build commit |
| TorchRL | pytorch/rl | MIT | Exact optional NPU build commit |
| AutoSERL ideas | autoserl/AutoSERL | No license file found at the cited commit | Citation only; source not bundled or cloned by setup |

Complete inspected license texts are in licenses/. The ResFiT patch changes resfit/dexmg/environments/dexmg.py and resfit/rl_finetuning/scripts/train_rlpd_dexmg.py for environment/device/entry compatibility; the original files retain their upstream attribution. No claim of new ownership over these changes is made. The non-commercial restrictions of ResFiT and DexMimicGen remain applicable; public availability is not a commercial-use permission. Dataset and asset terms must be reviewed separately when obtaining them.

The supervisor is this project's implementation of published AutoSERL concepts extended to two Panda arms, not a redistribution of the upstream Franka wrapper. Refer to the original repositories and their papers for attribution and methodology. No additional project-wide license is selected in this publication.
