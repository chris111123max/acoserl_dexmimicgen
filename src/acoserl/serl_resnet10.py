"""SERL's official ImageNet-1K ResNet-10 backbone in native PyTorch.

Uses the exact upstream 7x7 stem, 4-group GN, four basic blocks and JAX
SAME padding. All four official pretrained stages and the ImageNet-1K
classification output_head are loaded and checked, with no parameter drops.
For 84px bimanual grasping the frozen spatial feature is extracted after
stage 2 (8x8x256 at the official 128px input), before stage 3 downsamples
it to 4x4. This is a documented RLPD spatial-token ADAPTATION, not an
assertion of end-to-end numerical parity with the SERL JAX actor.
No Hugging Face, JAX or network dependency at training time.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

OFFICIAL_URL = "https://github.com/rail-berkeley/serl/releases/download/resnet10/resnet10_params.pkl"
OFFICIAL_BYTES = 21677915
MODEL_VERSION = "serl_official_resnet10_torch_v2"
WEIGHT_DIR = Path(__file__).resolve().parents[2] / "cache/pretrained_visual/serl_resnet10_official"
SOURCE_FILE = WEIGHT_DIR / "resnet10_params.pkl"
CONVERTED_FILE = WEIGHT_DIR / "serl_resnet10_torch.pt"
FEATURE_STAGE = 2
FEATURE_CHANNELS = 256
FEATURE_SIDE = 8
PRETRAIN_CLASSES = 1000
FULL_BACKBONE_CHANNELS = 512


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def same_pad(x, kernel: int, stride: int, value: float = 0.):
    """Match Flax Conv/Pool SAME (extra pixel on bottom/right for even sizes)."""
    height, width = x.shape[-2:]
    extra_h = max(0, (math.ceil(height / stride)-1)*stride + kernel-height)
    extra_w = max(0, (math.ceil(width / stride)-1)*stride + kernel-width)
    return F.pad(x, (extra_w//2, extra_w-extra_w//2,
                     extra_h//2, extra_h-extra_h//2), value=value)


class SameConv(nn.Conv2d):
    def __init__(self, channels_in, channels_out, kernel, stride=1):
        super().__init__(channels_in, channels_out, kernel, stride=stride,
                         padding=0, bias=False)

    def forward(self, x):
        k = self.kernel_size[0]
        return F.conv2d(same_pad(x, k, self.stride[0]), self.weight,
                        bias=None, stride=self.stride, padding=0)


class SERLBasicBlock(nn.Module):
    def __init__(self, width_in: int, width_out: int, stride: int = 1):
        super().__init__()
        self.conv1 = SameConv(width_in, width_out, 3, stride)
        self.norm1 = nn.GroupNorm(4, width_out, eps=1e-5)
        self.conv2 = SameConv(width_out, width_out, 3, 1)
        self.norm2 = nn.GroupNorm(4, width_out, eps=1e-5)
        self.shortcut = (nn.Sequential(
            SameConv(width_in, width_out, 1, stride),
            nn.GroupNorm(4, width_out, eps=1e-5))
            if width_in != width_out else nn.Identity())

    def forward(self, x):
        residual = self.shortcut(x)
        y = F.relu(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        return F.relu(y + residual)


class SERLResNet10(nn.Module):
    """The complete four-stage backbone and official ImageNet-1K output head.

    The classifier weights remain present, strict-loaded and frozen, but the
    online spatial-token RL forward exits at stage2, saving classifier work.
    """
    def __init__(self):
        super().__init__()
        # Official JAX conv_init has EXPLICIT symmetric 3px padding.
        self.conv_init = nn.Conv2d(3, 64, 7, 2, padding=3, bias=False)
        self.norm_init = nn.GroupNorm(4, 64, eps=1e-5)
        self.blocks = nn.ModuleList((
            SERLBasicBlock(64, 64),
            SERLBasicBlock(64, 128, 2),
            SERLBasicBlock(128, 256, 2),
            SERLBasicBlock(256, 512, 2),
        ))
        # The official ImageNet-1K pickle additionally contains output_head.
        # Preserve and strict-load BOTH kernel and bias rather than silently
        # discarding them. This layer is not executed during spatial RL.
        self.output_head = nn.Linear(FULL_BACKBONE_CHANNELS, PRETRAIN_CLASSES)
        self.register_buffer("mean", torch.tensor((.485, .456, .406)).view(1,3,1,1))
        self.register_buffer("std", torch.tensor((.229, .224, .225)).view(1,3,1,1))

    def forward(self, obs: torch.Tensor, spatial: bool = True):
        # QAgent._encode already scales uint8 to float [0,1] before calling.
        if obs.ndim != 4 or obs.shape[1] != 3:
            raise ValueError("SERL ResNet-10 expects [B,3,H,W]")
        x = F.interpolate(obs, size=(128,128), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        x = F.relu(self.norm_init(self.conv_init(x)))
        # The official JAX max_pool uses SAME and receives nonnegative inputs;
        # zero padding here equals negative infinity for a maximum reduction.
        x = F.max_pool2d(same_pad(x, 3, 2), kernel_size=3, stride=2)
        for i, block in enumerate(self.blocks):
            x = block(x)
            if spatial and i == FEATURE_STAGE:
                if x.shape[1:] != (FEATURE_CHANNELS, FEATURE_SIDE, FEATURE_SIDE):
                    raise AssertionError("unexpected pretrained spatial feature geometry")
                return x
        return x


def official_weights_from_jax(tree):
    """Strictly translate all official Flax parameters; no untrained fallback."""
    import numpy as np
    expected_top = {"conv_init", "norm_init", "output_head",
                    *(f"ResNetBlock_{i}" for i in range(4))}
    observed_top = set(tree)
    if observed_top != expected_top:
        raise ValueError("official SERL checkpoint top-level keys mismatch: "
                         f"missing={sorted(expected_top-observed_top)}, "
                         f"extra={sorted(observed_top-expected_top)}")
    weights = {}
    def conv(key, data):
        if set(data) != {"kernel"}:
            raise ValueError("invalid official JAX convolution "+key)
        value = torch.from_numpy(np.asarray(data["kernel"]).copy()).permute(3,2,0,1).contiguous()
        if not torch.isfinite(value).all():
            raise ValueError("nonfinite official convolution "+key)
        weights[key] = value
    def norm(key, data):
        if set(data) != {"scale", "bias"}:
            raise ValueError("invalid official JAX GroupNorm "+key)
        for suffix, source in (("weight", "scale"), ("bias", "bias")):
            value = torch.from_numpy(np.asarray(data[source]).copy()).contiguous()
            if not torch.isfinite(value).all():
                raise ValueError("nonfinite official GroupNorm "+key)
            weights[f"{key}.{suffix}"] = value

    conv("conv_init.weight", tree["conv_init"])
    norm("norm_init", tree["norm_init"])
    for i in range(4):
        name = f"ResNetBlock_{i}"
        block = tree[name]
        expected = {"Conv_0", "Conv_1", "MyGroupNorm_0", "MyGroupNorm_1"}
        if i > 0:
            expected |= {"conv_proj", "norm_proj"}
        if set(block) != expected:
            raise ValueError(f"unexpected official block keys in {name}")
        base = f"blocks.{i}"
        conv(base+".conv1.weight", block["Conv_0"])
        conv(base+".conv2.weight", block["Conv_1"])
        norm(base+".norm1", block["MyGroupNorm_0"])
        norm(base+".norm2", block["MyGroupNorm_1"])
        if i > 0:
            conv(base+".shortcut.0.weight", block["conv_proj"])
            norm(base+".shortcut.1", block["norm_proj"])

    # Flax Dense(kernel) is [in_features,out_features], whereas PyTorch
    # nn.Linear.weight is [out_features,in_features]. The observed official
    # release also contains the ImageNet classification output head.
    # Never ignore or silently filter the pretraining-only tensors.
    output_head = tree["output_head"]
    if set(output_head) != {"kernel", "bias"}:
        raise ValueError("official SERL output_head keys mismatch: "
                         f"expected=['bias','kernel'], got={sorted(output_head)}")
    head_weight = torch.from_numpy(np.asarray(output_head["kernel"]).copy()).T.contiguous()
    head_bias = torch.from_numpy(np.asarray(output_head["bias"]).copy()).contiguous()
    if (tuple(head_weight.shape) != (PRETRAIN_CLASSES, FULL_BACKBONE_CHANNELS)
            or tuple(head_bias.shape) != (PRETRAIN_CLASSES,)):
        raise ValueError("official SERL output_head shape mismatch: "
                         f"kernel={tuple(head_weight.shape)}, bias={tuple(head_bias.shape)}; "
                         f"expected kernel=({PRETRAIN_CLASSES},{FULL_BACKBONE_CHANNELS}) "
                         f"and bias=({PRETRAIN_CLASSES},)")
    if (head_weight.dtype != torch.float32 or head_bias.dtype != torch.float32
            or not torch.isfinite(head_weight).all()
            or not torch.isfinite(head_bias).all()):
        raise ValueError("official SERL output_head must have finite float32 tensors")
    weights["output_head.weight"] = head_weight
    weights["output_head.bias"] = head_bias

    backbone = SERLResNet10()
    expected_state = backbone.state_dict()
    for name in ("mean", "std"):
        weights[name] = expected_state[name].clone()
    if set(weights) != set(expected_state):
        raise ValueError("incomplete official conversion: "+
                         repr(sorted(set(expected_state)^set(weights))))
    for name, tensor in weights.items():
        if tensor.shape != expected_state[name].shape:
            raise ValueError(f"mismatched official tensor {name}: {tensor.shape} vs {expected_state[name].shape}")
    backbone.load_state_dict(weights, strict=True)
    return weights


def load_verified_backbone() -> tuple[SERLResNet10, dict]:
    if not CONVERTED_FILE.is_file():
        raise FileNotFoundError(
            f"Official SERL converted weight file missing: {CONVERTED_FILE}. "
            "Ask Codex to download the official release and run "
            "scripts/autoserl_baseline/prepare_serl_resnet10.py first.")
    pack = torch.load(CONVERTED_FILE, map_location="cpu", weights_only=True)
    if pack.get("version") != MODEL_VERSION or pack.get("origin") != OFFICIAL_URL:
        raise ValueError("incompatible or non-official SERL visual checkpoint")
    if pack.get("official_source_bytes") != OFFICIAL_BYTES:
        raise ValueError("SERL official checkpoint size does not match release")
    if pack.get("selected_spatial_stage") != FEATURE_STAGE:
        raise ValueError("SERL feature-stage protocol does not match converted checkpoint")
    if (pack.get("official_output_head_preserved") is not True
            or pack.get("official_output_head_classes") != PRETRAIN_CLASSES
            or pack.get("official_output_head_inputs") != FULL_BACKBONE_CHANNELS):
        raise ValueError("SERL converted file must preserve the exact official output_head")
    evidence = CONVERTED_FILE.with_name("conversion_manifest.json")
    if not evidence.is_file():
        raise FileNotFoundError("Missing SERL weight conversion_manifest.json")
    manifest = json.loads(evidence.read_text())
    if (manifest.get("official_source_sha256") != pack.get("official_source_sha256")
            or manifest.get("output_sha256") != file_sha256(CONVERTED_FILE)
            or manifest.get("output_head_preserved") is not True
            or manifest.get("output_head_shapes") != {
                "weight": [PRETRAIN_CLASSES, FULL_BACKBONE_CHANNELS],
                "bias": [PRETRAIN_CLASSES],
            }):
        raise ValueError("SERL conversion provenance, classifier or output SHA256 mismatch")
    if not SOURCE_FILE.is_file() or SOURCE_FILE.stat().st_size != OFFICIAL_BYTES:
        raise FileNotFoundError("Original official SERL checkpoint not found at fixed cache path")
    sha = file_sha256(SOURCE_FILE)
    if sha != pack.get("official_source_sha256"):
        raise ValueError("SERL source checkpoint SHA256 differs from converted checkpoint")
    backbone = SERLResNet10()
    backbone.load_state_dict(pack["weights"], strict=True)
    backbone.requires_grad_(False)
    backbone.eval()
    return backbone, {k:pack[k] for k in (
        "version", "origin", "official_source_sha256", "official_source_bytes",
        "selected_spatial_stage", "conversion_tensors")}
