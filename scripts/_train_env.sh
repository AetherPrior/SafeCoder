# Shared env for SafeCoder training launchers. Source, do not execute.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

if [[ -z "${SAFECODER_PYTHON:-}" ]]; then
  if command -v conda >/dev/null 2>&1; then
    SAFECODER_PYTHON="$(conda run -n safecoder which python 2>/dev/null || true)"
  fi
  SAFECODER_PYTHON="${SAFECODER_PYTHON:-/space1/asura/miniconda3/envs/safecoder/bin/python}"
fi
if [[ ! -x "${SAFECODER_PYTHON}" ]]; then
  echo "[error] Python not found for conda env 'safecoder': ${SAFECODER_PYTHON}" >&2
  exit 1
fi
TORCHRUN="${SAFECODER_PYTHON%/*}/torchrun"
