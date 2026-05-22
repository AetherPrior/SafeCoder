#!/usr/bin/env bash
# Multi-GPU training via torchrun (Accelerate auto-detects the distributed env).
#
# Examples:
#   # 2 GPUs
#   NPROC=2 ./scripts/train_torchrun.sh --pretrain_name qwen3-8b --output_name qwen3-8b-lora-safecoder \
#     --datasets evol sec-desc sec-new-desc --lora
#
#   # 4 GPUs, specific devices
#   CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC=4 ./scripts/train_torchrun.sh ...
#
#   # Single GPU (plain python, no torchrun overhead)
#   ./scripts/train_torchrun.sh --pretrain_name qwen3-8b ...

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_train_env.sh"

NPROC="${NPROC:-${NUM_PROCESSES:-1}}"
MASTER_PORT="${MASTER_PORT:-29500}"

echo "[info] Python: ${SAFECODER_PYTHON}"

if [[ "${NPROC}" -le 1 ]]; then
  exec "${SAFECODER_PYTHON}" scripts/train.py "$@"
fi

if [[ ! -x "${TORCHRUN}" ]]; then
  echo "[error] torchrun not found: ${TORCHRUN}" >&2
  exit 1
fi

echo "[info] Launching with torchrun --nproc_per_node=${NPROC} (master_port=${MASTER_PORT})"
exec "${TORCHRUN}" \
  --standalone \
  --nproc_per_node="${NPROC}" \
  --master_port="${MASTER_PORT}" \
  scripts/train.py "$@"
