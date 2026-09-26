#!/usr/bin/env bash
# Create the conda env "mfm" for this repo, working on RTX 50-series (Blackwell) GPUs.
#
# Why not just `pip install -r requirements.txt`:
#   - torch==2.3.0 (CUDA 12.1) has no sm_120 kernels, so it cannot use an RTX 5090;
#     this installs torch 2.7.1 + CUDA 12.8 instead
#   - requirements.txt conflicts with itself (lightning-bolts 0.7.0 needs
#     pytorch-lightning<2.0 but 2.2.0 is pinned) and lists CLIP twice in a way pip rejects
#   - the code imports phate and laspy, which requirements.txt does not list
# So the exact versions of a tested working env are pinned in requirements-mfm.lock.txt
# and installed with --no-deps.
#
# The env also gets TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1: torch>=2.6 refuses to load the
# repo's own Lightning checkpoints (GeoPathNetTrain.load_from_checkpoint) otherwise.
#
# Usage:
#   bash scripts/install_mfm_env.sh               # create (or complete) the env "mfm"
#   RECREATE=1 bash scripts/install_mfm_env.sh    # delete an existing "mfm" env first
#   ENV_NAME=other bash scripts/install_mfm_env.sh
set -euo pipefail

ENV_NAME="${ENV_NAME:-mfm}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCK="$SCRIPT_DIR/requirements-mfm.lock.txt"
TORCH_INDEX="https://download.pytorch.org/whl/cu128"

command -v conda >/dev/null || { echo "conda not found in PATH" >&2; exit 1; }
eval "$(conda shell.bash hook)"

env_exists() { conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; }

if env_exists && [ "${RECREATE:-0}" = "1" ]; then
  echo "=== removing existing env '$ENV_NAME'"
  conda env remove -y -n "$ENV_NAME"
fi
if env_exists; then
  echo "=== env '$ENV_NAME' already exists, installing into it (RECREATE=1 to start fresh)"
else
  echo "=== creating env '$ENV_NAME' (python 3.11)"
  conda create -y -n "$ENV_NAME" python=3.11
fi

# conda's activate scripts reference unset variables, which `set -u` would abort on
set +u
conda activate "$ENV_NAME"
set -u
PY="$(command -v python)"
echo "=== using $PY"

# CLIP's setup.py needs pkg_resources, which setuptools>=70 no longer ships;
# CLIP and fld are built with --no-build-isolation so they see this version.
"$PY" -m pip install "setuptools<70" wheel

echo "=== installing torch 2.7.1 + CUDA 12.8"
"$PY" -m pip install --index-url "$TORCH_INDEX" \
  $(grep -E '^(torch|torchvision|torchaudio)==' "$LOCK")

echo "=== installing pinned dependencies"
"$PY" -m pip install --no-deps --no-build-isolation --extra-index-url "$TORCH_INDEX" -r "$LOCK"

conda env config vars set -n "$ENV_NAME" TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 >/dev/null

echo "=== verifying"
"$PY" -m pip check
"$PY" - <<'EOF'
import importlib, pathlib, torch
print("torch", torch.__version__, "| CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0), "| arch list:", torch.cuda.get_arch_list()[-2:])
    x = torch.randn(1024, 1024, device="cuda"); (x @ x).sum().item()
import clip, fld, phate, laspy, torchcfm, pytorch_lightning  # noqa: F401
print("third-party imports ok")
EOF
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
(cd "$REPO" && "$PY" -c "
import importlib, pathlib
mods = ['.'.join(f.with_suffix('').parts) for f in sorted(pathlib.Path('mfm').rglob('*.py'))]
for m in mods: importlib.import_module(m)
print(f'all {len(mods)} mfm modules import ok')
" 2>&1 | grep -v -i warning)

echo
echo "Done. Activate with:  conda activate $ENV_NAME"
