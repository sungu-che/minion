#!/usr/bin/env bash
# Track 2 L40S (48GB VRAM) All-in-VRAM launcher wrapper for GravityOCR SGLang server.
# Preserves trida-stack's exact self-speculative flags and venv while tuning VRAM/concurrency for L40S.
set -euo pipefail

export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-EAGER}"
export GRAVITY_OCR_RESIDENCY="${GRAVITY_OCR_RESIDENCY:-auto}"
export GRAVITY_OCR_KV="${GRAVITY_OCR_KV:-f32}"
export GRAVITY_OCR_WEIGHTS="${GRAVITY_OCR_WEIGHTS:-half}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export PORT="${PORT:-30000}"
export MODE="${MODE:-self-spec}"
export FAST="${FAST:-1}"
export MAXREQ="${MAXREQ:-96}"
export TOKENIZER_WORKERS="${TOKENIZER_WORKERS:-1}"
export SKIP_PREFLIGHT="${SKIP_PREFLIGHT:-1}"
export MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.88}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export TRACK2_SLICE_DIR="${TRACK2_SLICE_DIR:-${REPO_ROOT}/vocab_slices}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

STOCK_SERVE="${TRIDA_STACK:-/opt/trida-stack}/inference/gravityocr/serve.sh"
if [[ -f "${STOCK_SERVE}" ]]; then
    exec bash "${STOCK_SERVE}" "$@"
fi

# Fallback direct launch if invoked outside track2-base:v1
VBIN="${VBIN:-${SGLANG_SRC:-/opt/sglang}/.venv-sglang/bin}"
CKPT="${CKPT:-${GRAVITYOCR_WEIGHTS:-/opt/gravityocr/weights}}"
exec "${VBIN}/python3" -m sglang.launch_server \
    --model-path "${CKPT}" \
    --served-model-name glm-ocr \
    --host 127.0.0.1 \
    --port "${PORT}" \
    --dtype bfloat16 \
    --mem-fraction-static "${MEM_FRACTION_STATIC}" \
    --max-running-requests "${MAXREQ}" \
    --cuda-graph-max-bs 64 \
    --tokenizer-worker-num "${TOKENIZER_WORKERS}" \
    --trust-remote-code \
    --log-level warning
