#!/usr/bin/env python3
"""Track 2 (Gravity-OCR Speed Run) High-Throughput Lossless Multi-Batch Server (V2).

Pure-BF16 lossless architecture verified against L40S (sm_89) + track2-base:v1:
1. Untimed All-in-VRAM preloading, SGLang overlay build (L40S ViT CUDA Graphs + SDPA),
   and multi-batch warmup before READY.
2. Zero-decode /dev/shm symlink IPC (`c_{idx}.png`): eliminates 1.2s client PIL decode/BMP
   re-encode AND eliminates URL-unescape bugs on crop IDs containing `%20` / `%28`.
3. Header-only (~4ms) LPT scheduling: `table -> text -> formula`, descending by pixel area
   (`W * H`) so the longest tables launch first (eliminating tail latency).
4. Persistent HTTP/1.1 Keep-Alive worker connections (eliminates 402 TCP handshakes).
5. Code-fence sanitizer + original manifest ID order atomic `fsync` writer.

Tuning switches (environment variables):
  TRACK2_STAGE        : "0" (reference order) | "1"+ (LPT + /dev/shm symlink + keep-alive, default "3")
  TRACK2_CONCURRENCY  : concurrent SGLang requests (default "96")
  TRACK2_SHM_IPC      : "1" (/dev/shm symlink zero-copy IPC, default) | "0" (base64 PNG)
  TRACK2_TRIM         : "0" (original pixels, default for 100% baseline quality) | "1"
  TRACK2_VIRTUAL_GRID : "0" (default — avoids synthetic table TEDS drop) | "1"
  TRACK2_RESCUE       : "1" (Cauchy-Schwarz lossless lm_head rescue, default) | "0"
"""
from __future__ import annotations

import base64
import http.client
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
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
USE_SHM_IPC = STAGE >= 1 and os.environ.get("TRACK2_SHM_IPC", "1") == "1"

SHM_DIR = Path("/dev/shm/track2_crops")
if USE_SHM_IPC:
    try:
        SHM_DIR.mkdir(parents=True, exist_ok=True)
    except OSError:
        USE_SHM_IPC = False

IPC_MODE = "shm_symlink" if USE_SHM_IPC else "png_b64"
_THREAD_LOCAL = threading.local()

TASK_PROMPTS = {
    "table": "Table Recognition:",
    "text": "Text Recognition:",
    "formula": "Formula Recognition:",
}
TASK_BUCKET_ORDER = {"table": 0, "text": 1, "formula": 2}

FENCE_HEAD_RE = re.compile(r"^\s*```(?:html|markdown|md|latex|tex|text)?\s*\n?", re.IGNORECASE)
FENCE_TAIL_RE = re.compile(r"\n?\s*```\s*$", re.IGNORECASE)


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
    """Strip only spurious markdown code fences (```html/```latex) if emitted."""
    if not text or STAGE == 0:
        return text or ""
    if "```" in text:
        cleaned = FENCE_HEAD_RE.sub("", text.strip())
        cleaned = FENCE_TAIL_RE.sub("", cleaned).strip()
        return cleaned
    return text


def detect_task(item: dict[str, Any]) -> str:
    raw = f"{item.get('task', '')} {item.get('prompt', '')}".lower()
    if "table" in raw:
        return "table"
    if "formula" in raw:
        return "formula"
    return "text"


def _get_conn() -> http.client.HTTPConnection:
    conn: http.client.HTTPConnection | None = getattr(_THREAD_LOCAL, "conn", None)
    if conn is None:
        conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=300)
        _THREAD_LOCAL.conn = conn
    return conn


def call_sglang(image_url: str, prompt: str) -> str:
    payload = json.dumps({
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
    headers = {"Content-Type": "application/json", "Connection": "keep-alive"}
    for attempt in range(2):
        conn = _get_conn()
        try:
            conn.request("POST", "/v1/chat/completions", body=payload, headers=headers)
            resp = conn.getresponse()
            raw = resp.read()
            if resp.status != 200:
                raise RuntimeError(f"HTTP {resp.status}: {raw[:200]!r}")
            return json.loads(raw)["choices"][0]["message"]["content"] or ""
        except (http.client.HTTPException, OSError):
            try:
                conn.close()
            except Exception:
                pass
            _THREAD_LOCAL.conn = None
            if attempt == 1:
                raise
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
        # Pure numeric filename avoids URL-unescape errors on `%20`, `%28`, `#`, etc.
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
    """Header-only (~4ms) LPT preparation: creates zero-copy symlinks and sorts table->text->formula by area."""
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


def infer_crop(prep: dict[str, Any]) -> tuple[str, str]:
    cid = prep["id"]
    task_key = prep["task_key"]
    try:
        raw = call_sglang(prep["image_url"], prep["prompt"])
        return cid, sanitize_output(raw, task_key)
    except Exception as e:
        log(f"crop {cid} failed: {e}")
        return cid, ""


def probe_ipc_mode(sample_item: dict[str, Any], folder: Path) -> str:
    if STAGE == 0 or not USE_SHM_IPC:
        return "png_b64"
    img_path = (folder / sample_item["image"]).resolve()
    prompt_str = sample_item.get("prompt") or TASK_PROMPTS[detect_task(sample_item)]
    for mode in ("shm_symlink", "shm_rawpath", "png_b64"):
        try:
            url = build_image_url(img_path, 0, mode)
            _ = call_sglang(url, prompt_str)
            cleanup_shm()
            log(f"IPC transport selected: {mode}")
            return mode
        except Exception as e:
            log(f"IPC probe {mode} rejected ({e}); trying fallback...")
            cleanup_shm()
    return "png_b64"


def run_manifest(folder: Path, gpu_pool: ThreadPoolExecutor, limit: int | None = None) -> list[tuple[str, str]]:
    manifest_path = folder / "manifest.jsonl"
    items = [json.loads(l) for l in manifest_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if limit is not None:
        items = items[:limit]
    if not items:
        return []

    t0 = time.perf_counter()
    scheduled = prepare_job_items(items, folder, IPC_MODE)
    t_prep = time.perf_counter()

    results_map: dict[str, str] = {}
    for cid, text in gpu_pool.map(infer_crop, scheduled):
        results_map[cid] = text
    t_infer = time.perf_counter()

    cleanup_shm()
    log(
        f"manifest {folder.name}: {len(items)} crops in {t_infer - t0:.3f}s "
        f"(prep={t_prep - t0:.3f}s, infer={t_infer - t_prep:.3f}s, ipc={IPC_MODE}, stage={STAGE})"
    )
    return [(m["id"], results_map.get(m["id"], "")) for m in items]


def warmup(gpu_pool: ThreadPoolExecutor) -> None:
    global IPC_MODE
    dev_dir = Path(DEV) if DEV else None
    if not dev_dir or not (dev_dir / "manifest.jsonl").exists():
        return
    items = [json.loads(l) for l in (dev_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    if not items:
        return
    t0 = time.monotonic()
    IPC_MODE = probe_ipc_mode(items[0], dev_dir)
    warm_limit = min(len(items), max(32, CONCURRENCY))
    run_manifest(dev_dir, gpu_pool, limit=warm_limit)
    log(f"untimed warmup ({warm_limit} dev crops) finished in {time.monotonic() - t0:.1f}s")


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
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as gpu_pool:
            warmup(gpu_pool)
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
                        results = run_manifest(job_dir, gpu_pool)
                        pred_path = job_dir / "predictions.jsonl"
                        out_lines = [json.dumps({"id": cid, "text": text}, ensure_ascii=False) for cid, text in results]
                        with open(pred_path, "w", encoding="utf-8") as f:
                            if out_lines:
                                f.write("\n".join(out_lines) + "\n")
                            f.flush()
                            os.fsync(f.fileno())
                        (job_dir / "DONE").touch()
                        seen.add(name)
                time.sleep(0.001)
    finally:
        cleanup_shm()
        try:
            os.killpg(server.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


if __name__ == "__main__":
    main()
