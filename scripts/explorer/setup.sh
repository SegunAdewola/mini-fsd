#!/usr/bin/env bash
#
# One-time setup for training Mini FSD on Northeastern's Explorer cluster.
#
# Run it on a GPU node (the login node is not for work like this):
#   srun -p gpu-interactive --gres=gpu:1 -t 01:00:00 --pty bash
#   bash /scratch/$USER/mini-fsd/repo/scripts/explorer/setup.sh
#
# Under /scratch/$USER/mini-fsd it makes:
#   venv/     training: torch, our package, ultralytics, rfdetr
#   da3env/   Depth Anything 3 only, which needs numpy<2 (E3a)
#   data/     the packs from packs/, unzipped
#   runs/     training output; logs/ Slurm logs
# Each step skips what is already there, so it is safe to run again.

set -euo pipefail

readonly ROOT="/scratch/${USER}/mini-fsd"
readonly REPO="${ROOT}/repo"
# PyTorch's CUDA 12.6 builds still include the V100 (sm_70), and the
# nodes' driver (CUDA 12.3) runs them. Newer CUDA builds drop the V100.
readonly TORCH_INDEX="https://download.pytorch.org/whl/cu126"
readonly TRANSFORMERS="transformers==5.24.0" # as in the local cv env
readonly ULTRALYTICS="ultralytics==8.4.214"  # ADR 0008
readonly RFDETR="rfdetr[train]==1.12.1"      # E1b
readonly DA3="depth-anything-3==0.1.1"       # E3a

uv_bin() {
  if ! command -v uv >/dev/null 2>&1; then
    # Astral's installer puts one binary in ~/.local/bin; no root needed.
    curl -LsSf https://astral.sh/uv/install.sh | sh
  fi
  export PATH="${HOME}/.local/bin:${PATH}"
}

# Makes a Python 3.12 venv with CUDA torch, unless it exists.
make_venv() {
  local venv="$1"
  if [[ -x "${venv}/bin/python" ]]; then
    return
  fi
  uv venv --python 3.12 "${venv}"
  uv pip install --python "${venv}/bin/python" \
    --index-url "${TORCH_INDEX}" torch torchvision
}

setup_training_env() {
  local venv="${ROOT}/venv"
  make_venv "${venv}"
  # torch is already in place, so uv keeps the CUDA 12.6 build.
  uv pip install --python "${venv}/bin/python" -e "${REPO}" \
    "${TRANSFORMERS}" "${ULTRALYTICS}" "${RFDETR}"
  # ultralytics pulls in opencv-python, which needs libGL; swap in the
  # headless build, which doesn't.
  uv pip uninstall --python "${venv}/bin/python" opencv-python
  uv pip install --python "${venv}/bin/python" --reinstall \
    opencv-python-headless
}

setup_da3_env() {
  local venv="${ROOT}/da3env"
  make_venv "${venv}"
  # xformers pins its own torch build; DA3 runs without it, so an
  # override with a marker that never matches leaves it out.
  local overrides="${ROOT}/da3-overrides.txt"
  echo 'xformers; sys_platform == "never"' >"${overrides}"
  uv pip install --python "${venv}/bin/python" --override "${overrides}" \
    "${DA3}"
  uv pip install --python "${venv}/bin/python" --no-deps -e "${REPO}"
}

unpack_data() {
  local zip name
  mkdir -p "${ROOT}/data"
  for zip in "${ROOT}"/packs/*.zip; do
    [[ -e "${zip}" ]] || continue
    name="$(basename "${zip}" .zip)"
    if [[ ! -d "${ROOT}/data/${name}" ]]; then
      echo "unpacking ${name}"
      unzip -q "${zip}" -d "${ROOT}/data"
    fi
  done
}

check_gpu() {
  "${ROOT}/venv/bin/python" - <<'EOF'
import torch

print("torch", torch.__version__, "CUDA", torch.version.cuda)
assert torch.cuda.is_available(), "no GPU seen: run this on a GPU node"
x = torch.randn(256, 256, device="cuda")
# A build without kernels for this GPU fails here, not mid-training.
print("GPU", torch.cuda.get_device_name(), "matmul", float((x @ x).sum()))
EOF
}

main() {
  mkdir -p "${ROOT}/packs" "${ROOT}/runs" "${ROOT}/logs"
  uv_bin
  setup_training_env
  setup_da3_env
  unpack_data
  check_gpu
  echo "setup done in ${ROOT}"
}

main "$@"
