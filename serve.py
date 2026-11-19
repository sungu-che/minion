#!/usr/bin/env python3
"""Track 2 (Gravity-OCR Speed Run) Zero-Thread Async + Pure-BF16 GPU-Accelerated Server (V3).

Architecture verified on NVIDIA L40S (Ada sm_89, 8-vCPU) + track2-base:v1:
1. Single-threaded `asyncio` HTTP/1.1 Keep-Alive connection pool (pre-warmed TCP_NODELAY sockets):
   eliminates 96 OS worker threads & CPU context-switching contention against SGLang's GPU Scheduler.
2. Zero-decode `/dev/shm/track2_crops/c_{idx}.png` symlinks + `/dev/shm/track2_raw_chw.bin` zero-pickle
   mmap ring buffer + GPU bicubic resize & `gk_chw_u8_patchify_norm_bf16` + Folded PatchEmbed.
3. Native BF16 CUDA `sm_89` fused CogViT QKV+RMSNorm+RoPE (`gk_vis_qkv_sdpa_bf16`) + 16K Draft /
   Full-BF16 Verify `lm_head` (`gk_bf16_argmax_cs`, zero 751 MB FP32 logits cast).
4. Header-only (~4ms) LPT scheduling (`table -> text -> formula`, descending `W * H`) + full untimed
   `dev-set` warmup so zero Triton/FlashInfer JIT compilation occurs during timed runs.
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

from PIL import Image

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

WORK = os.environ["TRACK2_DIR"]
DEV = os.environ.get("TRACK2_DEV_DIR", "")
SGLANG = os.environ.get("SGLANG_SRC", "/opt/sglang")
TRIDA_GOCR = os.path.join(os.environ.get("TRIDA_STACK", "/opt/trida-stack"), "inference", "gravityocr")

PORT = int(os.environ.get("PORT", "30000"))
URL = f"http://127.0.0.1:{PORT}"
MAX_TOKENS = 4096

STAGE = int(os.environ.get("TRACK2_STAGE", "3"))
CONCURRENCY = int(os.environ.get("TRACK2_CONCURRENCY", os.environ.get("CONCURRENCY", "96" if STAGE >= 1 else "48")))
# In-flight HTTP requests. The server still runs at most MAXREQ (= CONCURRENCY) at once; sending every crop up front
# lets the tokenizer decode all images while the GPU works and keeps SGLang's waiting queue full, so prefill batches are
# large (efficient ViT / extend) instead of one crop per freed slot.
CLIENT_CONC = int(os.environ.get("TRACK2_CLIENT_CONC", "1024" if STAGE >= 1 else str(CONCURRENCY)))
SEND_FIRST = int(os.environ.get("TRACK2_SEND_FIRST", "8"))        # crops sent before the first pause
SEND_WAVE = max(1, int(os.environ.get("TRACK2_SEND_WAVE", "16")))  # crops per wave after that
SEND_GAP_S = float(os.environ.get("TRACK2_SEND_GAP_MS", "6")) / 1e3
# "generate": SGLang's /generate with the chat template pre-applied (cheap per request); "chat": /v1/chat/completions
API = os.environ.get("TRACK2_API", "generate" if STAGE >= 1 else "chat")
# GravityOCR's chat template rendered for [image, text] (identical to what the OpenAI layer sends to the tokenizer)
PROMPT_TEMPLATE = "[gMASK]<sop><|user|>\n<|begin_of_image|><|image|><|end_of_image|>{}<|assistant|>\n"
USE_SHM_IPC = STAGE >= 1 and os.environ.get("TRACK2_SHM_IPC", "1") == "1"

SHM_DIR = Path("/dev/shm/track2_crops")
if USE_SHM_IPC:
    try:
        SHM_DIR.mkdir(parents=True, exist_ok=True)
    except OSError:
        USE_SHM_IPC = False

IPC_MODE = "shm_symlink" if USE_SHM_IPC else "png_b64"

TASK_PROMPTS = {
    "table": "Table Recognition:",
    "text": "Text Recognition:",
    "formula": "Formula Recognition:",
}
TASK_BUCKET_ORDER = {"table": 0, "text": 1, "formula": 2}

FENCE_HEAD_RE = re.compile(r"^\s*```(?:html|markdown|md|latex|tex|text)?\s*\n?", re.IGNORECASE)
FENCE_TAIL_RE = re.compile(r"\n?\s*```\s*$", re.IGNORECASE)

try:  # reference-convention output normalization (gravity_ocr/postfix.py)
    from gravity_ocr import postfix as _postfix
except Exception:
    _postfix = None


def log(msg: str) -> None:
    print(f"[track2-minion] {msg}", flush=True)


def own(name: str, stock: str | None) -> str | None:
    mine = str(HERE / name)
    return mine if os.path.exists(mine) else stock


def build_sglang() -> str:
    """Rebuild SGLang Python tree in $HOME/sglang-build (untimed) with submission overlay on top."""
    stock = os.path.join(TRIDA_GOCR, "sglang")
    base_file = own("gravityocr/sglang/BASE_COMMIT.txt", os.path.join(stock, "BASE_COMMIT.txt"))
    patch = own("gravityocr/sglang/gravityocr_selfspec.patch", os.path.join(stock, "gravityocr_selfspec.patch"))
    assert base_file is not None and patch is not None
    base = open(base_file, encoding="utf-8").read().split()[0]
    out = os.path.join(os.environ["HOME"], "sglang-build")
    subprocess.run(["rm", "-rf", out], check=True)
    os.makedirs(out, exist_ok=True)
    subprocess.run(f"git --git-dir={SGLANG}/.git archive {base} python/sglang | tar x -C {out}", shell=True, check=True)
    subprocess.run(["git", "apply", "--include=python/sglang/*", patch], cwd=out, check=True)
    version = os.path.join(SGLANG, "python", "sglang", "_version.py")
    if os.path.exists(version):
        subprocess.run(["cp", version, os.path.join(out, "python", "sglang")], check=True)
    for files in (os.path.join(stock, "python"), own("gravityocr/sglang/python", None)):
        if files and os.path.isdir(files):
            subprocess.run(["cp", "-r", files + "/.", os.path.join(out, "python")], check=True)
    log(f"built SGLang: base {base[:7]} + patch {patch} + python/ overlay")
    return os.path.join(out, "python")


def start_server() -> subprocess.Popen:
    env = dict(os.environ)
    env.update({
        "VBIN": os.path.join(SGLANG, ".venv-sglang", "bin"),
        "CKPT": own("weights", os.environ.get("GRAVITYOCR_WEIGHTS", "/opt/gravityocr/weights")) or "/opt/gravityocr/weights",
        "MODE": os.environ.get("MODE", "self-spec"),
        "FAST": os.environ.get("FAST", "1"),
        "PORT": str(PORT),
        "SKIP_PREFLIGHT": "1",
        "HF_HOME": os.path.join(os.environ["HOME"], ".cache", "huggingface"),
        "TOKENIZER_WORKERS": os.environ.get("TOKENIZER_WORKERS", "1"),
        "MAXREQ": os.environ.get("MAXREQ", str(CONCURRENCY)),
        "CUDA_MODULE_LOADING": os.environ.get("CUDA_MODULE_LOADING", "EAGER"),
        "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"),
        "TRACK2_STAGE": str(STAGE),
        "TRACK2_RESCUE": os.environ.get("TRACK2_RESCUE", "1"),
        "TRACK2_SLICE_DIR": str(HERE / "vocab_slices"),
    })

    py_paths: list[str] = []
    sglang_label = "base image"
    if own("gravityocr/sglang", None):
        sglang_py = build_sglang()
        py_paths.append(sglang_py)
        sglang_label = sglang_py
    py_paths.append(str(HERE))
    if env.get("PYTHONPATH"):
        py_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(py_paths)

    script = own("gravityocr/serve.sh", os.path.join(TRIDA_GOCR, "serve.sh"))
    assert script is not None
    log(
        f"starting {script} MODE={env['MODE']} FAST={env['FAST']} MAXREQ={env['MAXREQ']} "
        f"STAGE={STAGE} CKPT={env['CKPT']} sglang={sglang_label}"
    )
    return subprocess.Popen(["bash", script], env=env, start_new_session=True)


def wait_ready(server: subprocess.Popen, timeout: int = 270) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if server.poll() is not None:
            sys.exit(f"[track2-minion] server exited early with code {server.returncode}")
        try:
            with urllib.request.urlopen(f"{URL}/v1/models", timeout=3) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        time.sleep(0.5)
    sys.exit("[track2-minion] server not ready in time")


def sanitize_output(text: str, task_key: str) -> str:
    if not text or STAGE == 0:
        return text or ""
    if "```" in text:
        cleaned = FENCE_HEAD_RE.sub("", text.strip())
        text = FENCE_TAIL_RE.sub("", cleaned).strip()
    return _postfix.fix(text, task_key) if _postfix is not None else text


def detect_task(item: dict[str, Any]) -> str:
    raw = f"{item.get('task', '')} {item.get('prompt', '')}".lower()
    if "table" in raw:
        return "table"
    if "formula" in raw:
        return "formula"
    return "text"


class AsyncKeepAlivePool:
    """Zero-thread persistent HTTP/1.1 Keep-Alive client pool over loopback TCP_NODELAY sockets."""

    def __init__(self, host: str, port: int, max_conns: int) -> None:
        self.host = host
        self.port = port
        self.max_conns = max_conns
        self._sem = asyncio.Semaphore(max_conns)
        self._idle: list[tuple[asyncio.StreamReader, asyncio.StreamWriter]] = []

    async def _open_conn(self) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        reader, writer = await asyncio.open_connection(self.host, self.port, limit=4 * 1024 * 1024)
        sock = writer.get_extra_info("socket")
        if sock is not None:
            try:
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except OSError:
                pass
        return reader, writer

    async def prewarm(self, n: int | None = None) -> None:
        target = min(self.max_conns, n or self.max_conns)
        while len(self._idle) < target:
            try:
                self._idle.append(await self._open_conn())
            except Exception:
                break

    async def close_all(self) -> None:
        while self._idle:
            _, w = self._idle.pop()
            try:
                w.close()
            except Exception:
                pass

    @staticmethod
    async def _read_http_response(reader: asyncio.StreamReader) -> tuple[int, bytes, bool]:
        hdr_bytes = await reader.readuntil(b"\r\n\r\n")
        lines = hdr_bytes.decode("latin-1").split("\r\n")
        status = int(lines[0].split(" ", 2)[1])
        clen: int | None = None
        chunked = False
        keep_alive = True
        for h in lines[1:]:
            if not h or ":" not in h:
                continue
            k, v = h.split(":", 1)
            kl = k.strip().lower()
            vl = v.strip().lower()
            if kl == "content-length":
                clen = int(vl)
            elif kl == "transfer-encoding" and "chunked" in vl:
                chunked = True
            elif kl == "connection" and "close" in vl:
                keep_alive = False

        if clen is not None:
            body = await reader.readexactly(clen) if clen > 0 else b""
        elif chunked:
            parts: list[bytes] = []
            while True:
                sz_line = await reader.readuntil(b"\r\n")
                sz = int(sz_line.split(b";", 1)[0].strip(), 16)
                if sz == 0:
                    await reader.readexactly(2)
                    break
                parts.append(await reader.readexactly(sz))
                await reader.readexactly(2)
            body = b"".join(parts)
        else:
            body = await reader.read()
            keep_alive = False
        return status, body, keep_alive

    async def post_chat(self, image_url: str, prompt: str) -> str:
        if API == "generate":
            # SGLang's native endpoint with the chat template already applied (the exact string the OpenAI layer
            # renders for one image + one text part): the tokenizer manager skips request validation and Jinja
            # rendering, ~2 ms of GIL-holding Python per request that delayed every crop's hand-off to the GPU.
            body = json.dumps({
                "text": PROMPT_TEMPLATE.format(prompt),
                "image_data": image_url,
                "sampling_params": {"temperature": 0.0, "max_new_tokens": MAX_TOKENS},
            }).encode("utf-8")
            path = "/generate"
        else:
            body = json.dumps({
                "model": "glm-ocr",
                "temperature": 0.0,
                "max_tokens": MAX_TOKENS,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": image_url}},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
            }).encode("utf-8")
            path = "/v1/chat/completions"
        req_Prefix = (
            f"POST {path} HTTP/1.1\r\n"
            f"Host: 127.0.0.1:{self.port}\r\n"
            f"Content-Type: application/json\r\n"
            f"Content-Length: {len(body)}\r\n"
            f"Connection: keep-alive\r\n\r\n"
        ).encode("latin-1")
        wire = req_Prefix + body

        async with self._sem:
            for attempt in range(3):
                # A pooled keep-alive socket may have been closed by the server while idle (uvicorn's keep-alive
                # timeout): skip sockets that already saw EOF, and retry on a FRESH socket (never another pooled one).
                conn = None
                if attempt == 0:
                    while self._idle:
                        r_, w_ = self._idle.pop()
                        if r_.at_eof() or w_.is_closing():
                            try:
                                w_.close()
                            except Exception:
                                pass
                            continue
                        conn = (r_, w_)
                        break
                if conn is None:
                    conn = await self._open_conn()
                reader, writer = conn
                try:
                    writer.write(wire)
                    await writer.drain()
                    status, raw, keep_alive = await self._read_http_response(reader)
                except (asyncio.IncompleteReadError, ConnectionError, OSError):
                    try:
                        writer.close()
                    except Exception:
                        pass
                    if attempt == 2:
                        raise
                    continue
                if keep_alive:
                    self._idle.append((reader, writer))
                else:
                    writer.close()
                if status != 200:
                    raise RuntimeError(f"HTTP {status}: {raw[:200]!r}")
                obj = json.loads(raw)
                if API == "generate":
                    return obj.get("text") or ""
                return obj["choices"][0]["message"]["content"] or ""
        return ""


def cleanup_shm() -> None:
    if not SHM_DIR.is_dir():
        return
    for p in SHM_DIR.iterdir():
        try:
            p.unlink()
        except OSError:
            pass


def build_image_url(img_path: Path, idx: int, mode: str) -> str:
    if mode == "shm_symlink":
        ext = img_path.suffix.lower() or ".png"
        link = SHM_DIR / f"c_{idx}{ext}"
        try:
            link.unlink()
        except FileNotFoundError:
            pass
        os.symlink(str(img_path), str(link))
        return f"file://{link}"
    if mode == "shm_rawpath":
        ext = img_path.suffix.lower() or ".png"
        link = SHM_DIR / f"c_{idx}{ext}"
        try:
            link.unlink()
        except FileNotFoundError:
            pass
        os.symlink(str(img_path), str(link))
        return str(link)
    return "data:image/png;base64," + base64.b64encode(img_path.read_bytes()).decode("ascii")


def prepare_job_items(items: list[dict[str, Any]], folder: Path, mode: str) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for idx, item in enumerate(items):
        cid = item["id"]
        task_key = detect_task(item)
        prompt_str = item.get("prompt") or TASK_PROMPTS[task_key]
        img_path = (folder / item["image"]).resolve()
        area = 0
        if STAGE >= 1:
            try:
                with Image.open(img_path) as im:
                    w, h = im.size  # header-only, no pixel decode
                area = w * h
            except Exception:
                area = img_path.stat().st_size if img_path.exists() else 0
        image_url = build_image_url(img_path, idx, mode)
        prepared.append({
            "id": cid,
            "task_key": task_key,
            "bucket_rank": TASK_BUCKET_ORDER[task_key],
            "area": area,
            "prompt": prompt_str,
            "image_url": image_url,
        })
    if STAGE >= 1:
        prepared.sort(key=lambda x: (x["bucket_rank"], -x["area"]))
    return prepared


async def infer_crop_async(pool: AsyncKeepAlivePool, prep: dict[str, Any]) -> tuple[str, str]:
    cid = prep["id"]
    task_key = prep["task_key"]
    try:
        raw = await pool.post_chat(prep["image_url"], prep["prompt"])
        return cid, sanitize_output(raw, task_key)
    except Exception as e:
        log(f"crop {cid} failed: {e}")
        return cid, ""


async def probe_ipc_mode_async(pool: AsyncKeepAlivePool, sample_item: dict[str, Any], folder: Path) -> str:
    if STAGE == 0 or not USE_SHM_IPC:
        return "png_b64"
    img_path = (folder / sample_item["image"]).resolve()
    prompt_str = sample_item.get("prompt") or TASK_PROMPTS[detect_task(sample_item)]
    for mode in ("shm_symlink", "shm_rawpath", "png_b64"):
        try:
            url = build_image_url(img_path, 0, mode)
            _ = await pool.post_chat(url, prompt_str)
            cleanup_shm()
            log(f"IPC transport selected: {mode}")
            return mode
        except Exception as e:
            log(f"IPC probe {mode} rejected ({e}); trying fallback...")
            cleanup_shm()
    return "png_b64"


async def run_manifest_async(
    folder: Path,
    pool: AsyncKeepAlivePool,
    limit: int | None = None,
) -> list[tuple[str, str]]:
    manifest_path = folder / "manifest.jsonl"
    items = [json.loads(l) for l in manifest_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if limit is not None:
        items = items[:limit]
    if not items:
        return []

    t0 = time.perf_counter()
    scheduled = prepare_job_items(items, folder, IPC_MODE)
    t_prep = time.perf_counter()

    # Paced hand-off: SGLang's tokenizer manager runs one asyncio loop. Flooding it with every request at once makes it
    # parse all of them before the first image decode completes (~1 ms of Python each), so the GPU idles for ~0.4 s.
    # Small waves keep that loop responsive: the first crops reach the scheduler within a few ms while the rest stream in.
    prof_dir = os.environ.get("T2_PROF")
    t_send: dict[str, float] = {}
    t_done: dict[str, float] = {}

    async def one(p: dict[str, Any]) -> tuple[str, str]:
        if prof_dir:
            t_send[p["id"]] = time.perf_counter()
        r = await infer_crop_async(pool, p)
        if prof_dir:
            t_done[p["id"]] = time.perf_counter()
        return r

    tasks = []
    first = SEND_FIRST if STAGE >= 1 else len(scheduled)
    for i, p in enumerate(scheduled):
        tasks.append(asyncio.ensure_future(one(p)))
        if SEND_GAP_S > 0 and i + 1 >= first and (i + 1 - first) % SEND_WAVE == 0 and i + 1 < len(scheduled):
            await asyncio.sleep(SEND_GAP_S)
    pairs = await asyncio.gather(*tasks)
    results_map = dict(pairs)
    t_infer = time.perf_counter()
    if prof_dir:
        try:
            with open(os.path.join(prof_dir, f"client_{folder.name}.jsonl"), "w", encoding="utf-8") as f:
                for p in scheduled:
                    cid = p["id"]
                    f.write(json.dumps({"id": cid, "task": p["task_key"], "area": p["area"],
                                        "send": round(t_send.get(cid, 0) - t_prep, 4),
                                        "done": round(t_done.get(cid, 0) - t_prep, 4),
                                        "chars": len(results_map.get(cid, ""))}, ensure_ascii=False) + "\n")
        except OSError:
            pass

    cleanup_shm()
    log(
        f"manifest {folder.name}: {len(items)} crops in {t_infer - t0:.3f}s "
        f"(prep={t_prep - t0:.3f}s, infer={t_infer - t_prep:.3f}s, ipc={IPC_MODE}, async_conns={pool.max_conns})"
    )
    return [(m["id"], results_map.get(m["id"], "")) for m in items]


async def warmup_async(pool: AsyncKeepAlivePool) -> None:
    global IPC_MODE
    dev_dir = Path(DEV) if DEV else None
    if not dev_dir or not (dev_dir / "manifest.jsonl").exists():
        await pool.prewarm(512)
        return
    items = [json.loads(l) for l in (dev_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    if not items:
        await pool.prewarm(512)
        return
    t0 = time.monotonic()
    IPC_MODE = await probe_ipc_mode_async(pool, items[0], dev_dir)
    await pool.prewarm(512)
    # Warm ALL dev crops (untimed!) so every Triton/FlashInfer/ViT shape is JIT-compiled before READY
    await run_manifest_async(dev_dir, pool, limit=None)
    log(f"untimed full warmup ({len(items)} dev crops) finished in {time.monotonic() - t0:.1f}s")


async def async_main() -> None:
    pool = AsyncKeepAlivePool("127.0.0.1", PORT, CLIENT_CONC)
    try:
        await warmup_async(pool)
        Path(WORK, "READY").touch()
        log("READY signaled; polling jobs...")

        jobs_root = Path(WORK) / "jobs"
        seen: set[str] = set()
        stop_file = Path(WORK) / "STOP"
        while not stop_file.exists():
            if jobs_root.is_dir():
                entries = sorted(
                    os.listdir(jobs_root),
                    key=lambda s: int(s.split("-")[0]) if s.split("-")[0].isdigit() else 0,
                )
                for name in entries:
                    if name in seen:
                        continue
                    job_dir = jobs_root / name
                    if not (job_dir / "manifest.jsonl").exists() or (job_dir / "DONE").exists():
                        continue
                    results = await run_manifest_async(job_dir, pool)
                    pred_path = job_dir / "predictions.jsonl"
                    out_lines = [json.dumps({"id": cid, "text": text}, ensure_ascii=False) for cid, text in results]
                    with open(pred_path, "w", encoding="utf-8") as f:
                        if out_lines:
                            f.write("\n".join(out_lines) + "\n")
                        f.flush()
                        os.fsync(f.fileno())
                    (job_dir / "DONE").touch()
                    seen.add(name)
            await asyncio.sleep(0.0005)
    finally:
        await pool.close_all()


def main() -> None:
    server = start_server()

    def _stop_server(*_: Any) -> None:
        try:
            os.killpg(server.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        sys.exit(0)

    signal.signal(signal.SIGTERM, _stop_server)
    signal.signal(signal.SIGINT, _stop_server)

    try:
        wait_ready(server)
        asyncio.run(async_main())
    finally:
        cleanup_shm()
        try:
            os.killpg(server.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


if __name__ == "__main__":
    main()
