#!/usr/bin/env bash
#
# Runs one minifsd.train module on an Explorer GPU node (ADR 0007's fallback).
#
# Submit from the login node; the first argument is the module, the rest are
# its own flags. D=/scratch/$USER/mini-fsd/data, R=/scratch/$USER/mini-fsd/runs:
#
#   sbatch -J e1a scripts/explorer/train.sbatch yolo --workers 8
#     --train v10=$D/v10/train coco=$D/coco-people/train
#     --val v10=$D/v10/val coco=$D/coco-people/val
#     --run-dir $R/e1a-yolo26n --work-dir /tmp/$USER-yolo
#
# (one command, wrapped here to fit). DA3 runs in its own env:
#   sbatch -J e3a --export=ALL,PYTHON=/scratch/$USER/mini-fsd/da3env/bin/python
#     scripts/explorer/train.sbatch da3_predict --images $D/da3-inputs
#     --out $R/da3-small-predictions.npz
#
# Each module resumes from its checkpoint, so a job that hits the 8 h limit
# is simply submitted again with the same command.
#
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/%u/mini-fsd/logs/%x-%j.out

set -euo pipefail

readonly ROOT="/scratch/${USER}/mini-fsd"
readonly PYTHON="${PYTHON:-${ROOT}/venv/bin/python}"

main() {
  if (($# < 1)); then
    echo "usage: sbatch train.sbatch MODULE [MODULE FLAGS...]" >&2
    exit 2
  fi
  local module="$1"
  shift
  cd "${ROOT}/repo"
  echo "job ${SLURM_JOB_ID:-local} on $(hostname), commit $(git rev-parse --short HEAD)"
  nvidia-smi --query-gpu=name,memory.total --format=csv
  exec "${PYTHON}" -m "minifsd.train.${module}" "$@"
}

main "$@"
