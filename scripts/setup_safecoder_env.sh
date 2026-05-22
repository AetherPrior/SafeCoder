#!/usr/bin/env bash
# Bootstrap the safecoder conda env for Qwen3 training.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

ENV_NAME="${SAFECODER_CONDA_ENV:-safecoder}"

if ! command -v conda >/dev/null 2>&1; then
  echo "[error] conda not found" >&2
  exit 1
fi

# shellcheck source=/dev/null
source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[info] Creating conda env: ${ENV_NAME}"
  conda create -n "${ENV_NAME}" python=3.11 -y
fi

conda activate "${ENV_NAME}"

echo "[info] Installing Qwen3-compatible dependencies..."
pip install 'torch>=2.6.0,<2.8.0' -r requirements-qwen3.txt
pip install -e .

echo "[info] Verifying imports..."
python - <<'PY'
from safecoder.trainer import Trainer
from safecoder.chat_templates import apply_safecoder_chat_template
from transformers import AutoTokenizer
import transformers
print("safecoder OK")
print("transformers", transformers.__version__)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)
print("Qwen3 tokenizer OK, vocab", tok.vocab_size)
PY

echo "[done] Environment ready. Run:"
echo "  conda activate ${ENV_NAME}"
echo "  cd ${REPO_ROOT}"
echo "  NPROC=2 ./scripts/train_torchrun.sh --pretrain_name qwen3-8b --output_name qwen3-8b-lora-safecoder --datasets evol sec-desc sec-new-desc --lora"
