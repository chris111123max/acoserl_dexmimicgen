#!/usr/bin/env python3
"""Convert the ORIGINAL official SERL ResNet-10 release weights for Ascend.

No network access: Codex downloads the 21.7 MB JAX pickle into the fixed
ignored cache directory. This script imports pickle ONLY when explicitly
confirming a trusted, official release file. It never fetches other models,
and training subsequently loads only a pure weights_only PyTorch artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from acoserl.serl_resnet10 import (
    CONVERTED_FILE, FEATURE_STAGE, FULL_BACKBONE_CHANNELS, MODEL_VERSION,
    OFFICIAL_BYTES, OFFICIAL_URL, PRETRAIN_CLASSES, SOURCE_FILE,
    file_sha256, official_weights_from_jax,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-sha256", required=True,
                        help="64-character digest of official release asset, independently recorded at download")
    parser.add_argument("--trust-official-pickle", action="store_true",
                        help="Required: pickle can execute code. Only use the official URL and fixed destination.")
    opts = parser.parse_args()
    if not opts.trust_official_pickle:
        parser.error("requires --trust-official-pickle after checking the official release URL")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", opts.expected_sha256):
        parser.error("invalid 64-character source SHA256")
    if not SOURCE_FILE.is_file() or SOURCE_FILE.stat().st_size != OFFICIAL_BYTES:
        raise RuntimeError(f"Official file missing or incomplete: {SOURCE_FILE}; "
                           f"expected exactly {OFFICIAL_BYTES} bytes")
    digest = file_sha256(SOURCE_FILE)
    if digest != opts.expected_sha256.lower():
        raise RuntimeError("Official release file SHA256 mismatch")
    if CONVERTED_FILE.exists():
        raise FileExistsError(f"Refusing to overwrite existing converted file: {CONVERTED_FILE}")
    # The executable pickle is never opened before the user acknowledges its
    # trusted release provenance and byte-length/SHA256 checks pass.
    with SOURCE_FILE.open("rb") as stream:
        tree = pickle.load(stream)
    weights = official_weights_from_jax(tree)
    # Classification output_head is present in the real SERL release.
    # It is strict-translated and stored rather than dropped, even though
    # the spatial-token training forward bypasses that classifier.
    assert 'output_head.weight' in weights and 'output_head.bias' in weights
    import torch
    payload = dict(version=MODEL_VERSION, origin=OFFICIAL_URL,
                   official_source_sha256=digest,
                   official_source_bytes=OFFICIAL_BYTES,
                   selected_spatial_stage=FEATURE_STAGE,
                   official_top_level_keys=sorted(tree.keys()),
                   official_output_head_preserved=True,
                   official_output_head_classes=PRETRAIN_CLASSES,
                   official_output_head_inputs=FULL_BACKBONE_CHANNELS,
                   conversion_tensors=len(weights), weights=weights)
    tmp = CONVERTED_FILE.with_name(CONVERTED_FILE.name + ".partial")
    if tmp.exists():
        raise FileExistsError("previous conversion partial exists; inspect before rerunning")
    CONVERTED_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        torch.save(payload, tmp)
        # Re-open in safe weights-only mode and verify exact tensors.
        loaded = torch.load(tmp, map_location="cpu", weights_only=True)
        if loaded["official_source_sha256"] != digest or set(loaded["weights"]) != set(weights):
            raise AssertionError("converted weights metadata/key mismatch")
        for key in weights:
            if not torch.equal(loaded["weights"][key], weights[key]):
                raise AssertionError("converted weights lost tensor parity: "+key)
        with tmp.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(tmp, CONVERTED_FILE)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    meta = dict(status="official_SERL_release_converted_exact_weights",
                source_url=OFFICIAL_URL, source_file=str(SOURCE_FILE),
                official_source_bytes=OFFICIAL_BYTES, official_source_sha256=digest,
                output_file=str(CONVERTED_FILE),
                output_bytes=CONVERTED_FILE.stat().st_size,
                output_sha256=file_sha256(CONVERTED_FILE),
                version=MODEL_VERSION, selected_spatial_stage=FEATURE_STAGE,
                tensors=len(weights), source_trusted_by_operator=True,
                official_top_level_keys=sorted(tree.keys()),
                output_head_preserved=True,
                output_head_shapes=dict(
                    weight=list(weights['output_head.weight'].shape),
                    bias=list(weights['output_head.bias'].shape)),
                output_head_online_executed=False,
                no_network_download_performed=True)
    manifest = CONVERTED_FILE.with_name("conversion_manifest.json")
    manifest.write_text(json.dumps(meta, indent=2, ensure_ascii=False)+"\n")
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
