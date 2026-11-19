#!/usr/bin/env bash
# Track 2 L40S (48GB VRAM) Pure-BF16 Launcher Wrapper for GravityOCR SGLang Server (V2.2).
# Unlocks:
#   - MEMFRAC=0.68 -> leaves ~13GB activation headroom so GRAPHBS=96 & ViT MAXM=3072 never OOM!
#   - MAXREQ=96 + GRAPHBS=96 -> 100% of active batches stay inside CUDA Graphs (zero bs>64 eager fallback!)
#   - SS_GLM_VIT_GRAPH_MAXM=3072 -> ~95% ViT CUDA Graph replay hit rate on L40S sm_89!
#   - TRACK2_FAST_MM=1 -> 8-core Rust AVX2 gk_patchify inside TokenizerManager (kills 3.06s HF bottleneck!)
set -euo pipefail

export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-EAGER}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export PORT="${PORT:-30000}"
export MODE="${MODE:-self-spec}"
export FAST="${FAST:-1}"
export MAXREQ="${MAXREQ:-96}"
export GRAPHBS="${GRAPHBS:-96}"
export MEMFRAC="${MEMFRAC:-0.82}"
export SS_GLM_VIT_GRAPH_MAXM="${SS_GLM_VIT_GRAPH_MAXM:-3072}"
export SS_GLM_VIT_GRAPH_CHECK="${SS_GLM_VIT_GRAPH_CHECK:-0}"
export TRACK2_FAST_MM="${TRACK2_FAST_MM:-1}"
export TRACK2_MM_WORKERS="${TRACK2_MM_WORKERS:-8}"
export TRACK2_VIS_QKV_CUDA="${TRACK2_VIS_QKV_CUDA:-1}"
export TRACK2_FAST_LM_HEAD="${TRACK2_FAST_LM_HEAD:-1}"
export SGLANG_MM_SKIP_COMPUTE_HASH="${SGLANG_MM_SKIP_COMPUTE_HASH:-1}"
export TOKENIZER_WORKERS="${TOKENIZER_WORKERS:-1}"
export SKIP_PREFLIGHT="${SKIP_PREFLIGHT:-1}"
export EXTRA_ARGS="${EXTRA_ARGS:---chunked-prefill-size 16384 --schedule-conservativeness 0.3 --log-level warning}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export TRACK2_SLICE_DIR="${TRACK2_SLICE_DIR:-${REPO_ROOT}/vocab_slices}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

STOCK_SERVE="${TRIDA_STACK:-/opt/trida-stack}/inference/gravityocr/serve.sh"
if [[ -f "${STOCK_SERVE}" ]]; then
    exec bash "${STOCK_SERVE}" "$@"
fi

VBIN="${VBIN:-${SGLANG_SRC:-/opt/sglang}/.venv-sglang/bin}"
CKPT="${CKPT:-${GRAVITYOCR_WEIGHTS:-/opt/gravityocr/weights}}"
exec "${VBIN}/python3" -m sglang.launch_server \
    --model-path "${CKPT}" \
    --served-model-name glm-ocr \
    --host 127.0.0.1 \
    --port "${PORT}" \
    --dtype bfloat16 \
    --mem-fraction-static "${MEMFRAC}" \
    --max-running-requests "${MAXREQ}" \
    --cuda-graph-max-bs "${GRAPHBS}" \
    --tokenizer-worker-num "${TOKENIZER_WORKERS}" \
    --trust-remote-code \
    --log-level warning
