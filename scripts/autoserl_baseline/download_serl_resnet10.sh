#!/usr/bin/env bash
# Download the OFFICIAL SERL release asset only; invoked by Codex on server.
# No model download runs when this script is added to the repository.
set -euo pipefail
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
DEST_DIR="$ROOT/cache/pretrained_visual/serl_resnet10_official"
DEST="$DEST_DIR/resnet10_params.pkl"
PART="$DEST_DIR/resnet10_params.pkl.part"
URL="https://github.com/rail-berkeley/serl/releases/download/resnet10/resnet10_params.pkl"
EXPECTED_BYTES=21677915
mkdir -p "$DEST_DIR"
size_of() { stat -c '%s' "$1"; }
if [[ -f "$DEST" ]]; then
  [[ $(size_of "$DEST") -eq $EXPECTED_BYTES ]] || {
    echo "Cached file exists but has the wrong size; inspect it before retrying: $DEST" >&2
    exit 2
  }
  echo "Official SERL weight file already cached: $DEST"
else
  echo "Downloading official ResNet-10 checkpoint (~21.7 MB) to $DEST_DIR"
  # aria2 supports HTTP range splitting/resume when the redirect server
  # permits it. Use modest parallelism, never dozens of connections.
  if command -v aria2c >/dev/null 2>&1; then
    aria2c --continue=true --max-connection-per-server=4 --split=4 \
      --min-split-size=4M --max-tries=6 --retry-wait=3 \
      --connect-timeout=15 --timeout=60 \
      --dir="$DEST_DIR" --out="$(basename "$PART")" "$URL"
  else
    curl --fail --location --retry 6 --retry-delay 3 \
      --retry-all-errors --connect-timeout 15 --max-time 1800 \
      --continue-at - --output "$PART" "$URL"
  fi
  [[ -f "$PART" ]] || { echo "No downloaded file" >&2; exit 2; }
  [[ $(size_of "$PART") -eq $EXPECTED_BYTES ]] || {
    echo "Downloaded file size mismatch: $(size_of "$PART") / $EXPECTED_BYTES; do not convert" >&2
    exit 2
  }
  mv -- "$PART" "$DEST"
fi
echo "Official URL: $URL"
sha256sum "$DEST"
echo "Next: source scripts/ascend16/activate.sh"
echo "Then: python scripts/autoserl_baseline/prepare_serl_resnet10.py --expected-sha256 <64-character-sha256-above> --trust-official-pickle"
