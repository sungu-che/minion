# track2-base:v1 — Track 2 competition image (Codabench runs every submission and the scorer in it;
# teams use it locally). Toolchain only; see versions.yaml for every pin and where it comes from.
#
# GravityOCR serving comes from trida-stack's recipe (inference/gravityocr/, release v0.2.0), not a local copy:
# upstream SGLang v0.5.10 (1519acf) + gravityocr_selfspec.patch, in a venv built from its requirements-lock.txt.
# Not published to a registry: teams build it from the starter kit's track2-base/ folder (this folder), and every GPU
# worker builds it the same way. Every input is public and pinned (base image by digest, trida-stack by commit, the
# Python lock, the weights by revision). Holds no dev or test data.
#   docker build -t track2-base:v1 .
# Base image pinned by digest (checked 2026-11-17: same layers as the image on the GPU workers).
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04@sha256:ad6d59a3bbf3e82c1c849c9ac09cfc2a3e0bbb8655042fd899be6681b3fe2a85

LABEL org.opencontainers.image.source=https://github.com/trillion-labs/trida-stack \
      org.opencontainers.image.description="Track 2 base image: pinned CUDA, PyTorch, SGLang + GravityOCR patch, GravityOCR weights" \
      org.opencontainers.image.licenses=MIT

# Fetched by commit SHA, not by tag: trida-stack v0.2.0 = 79ed00c (main on 2026-11-16, PR #72 merged).
ARG TRIDA_STACK_COMMIT=79ed00c72c99ad71edd301302a75bc15ada8e927
# GravityOCR-KO on Hugging Face: public, not gated (no token needed). Revision = HF commit of 2026-11-17.
ARG GRAVITYOCR_REPO=trillionlabs/GravityOCR-KO
ARG GRAVITYOCR_REVISION=6c749d417879037da8a1f6af4cf9aaa7632308bc
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    SGLANG_SRC=/opt/sglang \
    TRIDA_STACK=/opt/trida-stack \
    VLLM_USE_FLASHINFER_SAMPLER=0 \
    HF_HUB_OFFLINE=1 \
    GRAVITYOCR_WEIGHTS=/opt/gravityocr/weights \
    UV_PYTHON_INSTALL_DIR=/opt/uv/python
# ^ uv's Python must live outside /root (mode 700): the harness runs team code as an unprivileged user.
ENV PATH=/opt/sglang/.venv-sglang/bin:/usr/local/cuda/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates build-essential ninja-build cmake libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# uv at the version trida-stack's lock was generated and tested with
COPY --from=ghcr.io/astral-sh/uv:0.11.27 /uv /usr/local/bin/uv

# trida-stack at a pinned commit (kept in the image: serve.sh, ocr_request.py, docs for teams)
RUN git init -q $TRIDA_STACK && cd $TRIDA_STACK \
    && git fetch -q --depth 1 https://github.com/trillion-labs/trida-stack.git ${TRIDA_STACK_COMMIT} \
    && git checkout -q ${TRIDA_STACK_COMMIT}

# SGLang v0.5.10 + GravityOCR patch + locked Python 3.12 venv, exactly as trida-stack's setup.sh builds it.
# uv cache clean in the same RUN: otherwise ~11 GB of download cache stays in the layer.
RUN bash $TRIDA_STACK/inference/gravityocr/sglang/setup.sh && uv cache clean

# Weights at a fixed revision (runs have no network). PENDING DECISION: may move to Codabench task data instead,
# so a weights change does not need a new image (see versions.yaml, model:).
RUN HF_HUB_OFFLINE=0 python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('${GRAVITYOCR_REPO}', revision='${GRAVITYOCR_REVISION}', local_dir='${GRAVITYOCR_WEIGHTS}')"

# GravityOCR's evaluation environment, as trida-stack builds it (benchmark/gravityocr/setup.sh at the pinned commit):
# OmniDocBench at its pinned commit, a Python 3.10 scoring venv, and TeX Live with the packages CDM (formula metric)
# needs. The Track 2 scorer runs trida-stack's score.py metrics in it. SKIP_SGLANG=1: the serving env is above.
# magick: trida-stack run.sh's PyMuPDF stand-in for the two ImageMagick calls CDM makes, copied out of run.sh.
ENV GRAVITYOCR_EVAL=/opt/gravityocr-eval
RUN EVAL_HOME=$GRAVITYOCR_EVAL SKIP_SGLANG=1 bash $TRIDA_STACK/benchmark/gravityocr/setup.sh && uv cache clean \
    && mkdir -p $GRAVITYOCR_EVAL/bin \
    && { echo "#!$GRAVITYOCR_EVAL/venv/bin/python"; \
         sed -n '/^import re, sys$/,/^PY$/p' $TRIDA_STACK/benchmark/gravityocr/run.sh | sed '$d'; } > $GRAVITYOCR_EVAL/bin/magick \
    && chmod 755 $GRAVITYOCR_EVAL/bin/magick && grep -q "get_pixmap" $GRAVITYOCR_EVAL/bin/magick

# Runtime libraries the SGLang kernels load (sgl-kernel's common_ops needs libnuma.so.1). Kept as a late layer so
# adding one does not rebuild the SGLang environment; fold into the first apt-get step on the next full rebuild.
RUN apt-get update && apt-get install -y --no-install-recommends libnuma1 && rm -rf /var/lib/apt/lists/*

RUN rm -rf /tmp/* /var/tmp/*

# Nothing in the image may be writable by the unprivileged team user (it could carry data between timed runs):
# uv leaves world-writable .lock files under /opt. The harness wipes /tmp, /var/tmp, /dev/shm, /run/lock, /var/lock.
RUN find / -xdev \( -path /proc -o -path /sys -o -path /tmp -o -path /var/tmp -o -path /dev \
        -o -path /run/lock -o -path /var/lock \) -prune -o -perm -o+w ! -type l -print -exec chmod o-w {} +

# Pinned-version manifest + checker used by the Track 2 harness
COPY versions.yaml /etc/track2/versions.yaml
COPY check_versions.py /opt/track2/check_versions.py
RUN python /opt/track2/check_versions.py --write /etc/track2/pinned.json

WORKDIR /workspace
