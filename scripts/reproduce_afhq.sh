#!/usr/bin/env bash
# Reproduce the OT-MFM_RBF row of Table 2 (unpaired dog -> cat translation in
# SD-VAE latent space on AFHQ): configs/images/mfm.yaml, paper FID 37.87 / LPIPS 0.502.
#
# Steps: 1) download AFHQ + build cached ambient/latent tensors
#        2) train OT-MFM (mfm.train.main, 8000 flow epochs)
#        3) evaluate the final flow checkpoint with FID / LPIPS as in App. D.2
#
# This is a single training run that uses most of the GPU (~26.5 GiB), so unlike
# the single-cell script there is nothing to run in parallel.
#
# Usage:
#   conda activate mfm
#   bash scripts/reproduce_afhq.sh
#   WORK=/path/to/dir bash scripts/reproduce_afhq.sh
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$(realpath -m "${WORK:-$REPO/runs/afhq}")"
PYTHON="${PYTHON:-python}"
CONFIG="configs/images/mfm.yaml"
RUN_DIR="$WORK/mfm"

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

# Paper (App. D.2) uses Adam for both networks; the repo default for the flow net is AdamW.
# This key is not set in the YAML config, so the CLI value takes effect.
EXTRA_ARGS=(--flow_optimizer adam)

cd "$REPO"

# 1) data
if [ ! -f "$WORK/data/afhq/afhq_val_pixels_128.pt" ]; then
  "$PYTHON" scripts/afhq/prepare_afhq.py --working_dir "$WORK" --download
fi
mkdir -p "$RUN_DIR/data"
ln -sfn "$WORK/data/afhq" "$RUN_DIR/data/afhq"

# 2) train. The repo's own test step at the end does not follow the paper's
#    evaluation (see evaluate_afhq.py) and is expected to fail; the flow
#    checkpoint is already saved by then, so a non-zero exit is tolerated here.
if ! ls "$RUN_DIR"/checkpoints/image/*/flow_model/*.ckpt >/dev/null 2>&1; then
  echo "=== training OT-MFM (log: $RUN_DIR/train.log)"
  "$PYTHON" -m mfm.train.main --config_path "$CONFIG" \
    --working_dir "$RUN_DIR" "${EXTRA_ARGS[@]}" 2>&1 | tee "$RUN_DIR/train.log" \
    || echo "=== mfm.train.main exited non-zero (expected at its built-in test step)"
fi

CKPT="$(ls -t "$RUN_DIR"/checkpoints/image/*/flow_model/*.ckpt 2>/dev/null | head -1 || true)"
if [ -z "$CKPT" ]; then
  echo "=== no flow checkpoint found, training failed; see $RUN_DIR/train.log" >&2
  exit 1
fi
if ! grep -q "Epoch 7999" "$RUN_DIR/train.log" 2>/dev/null; then
  echo "=== warning: training may not have reached 8000 epochs; evaluating $CKPT" >&2
fi

# 3) evaluate
echo "=== evaluating $CKPT"
"$PYTHON" scripts/afhq/evaluate_afhq.py --config_path "$CONFIG" \
  --working_dir "$RUN_DIR" --ckpt "$CKPT" --out "$RUN_DIR/eval"

echo
echo "| Method | FID (ours) | LPIPS (ours) | FID (paper) | LPIPS (paper) |"
echo "|---|---|---|---|---|"
"$PYTHON" -c "import json,sys; r=json.load(open(sys.argv[1])); print(f'| OT-MFM_RBF | {r[\"FID\"]:.2f} | {r[\"LPIPS\"]:.3f} | 37.87 | 0.502 |')" \
  "$RUN_DIR/eval/result.json"
