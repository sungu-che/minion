#!/usr/bin/env python3
"""Track 2 (Gravity-OCR Speed Run) High-Throughput Lossless Multi-Batch Server.

Follows the Track 2 harness contract strictly (READY -> jobs/<n>-<split>/manifest.jsonl ->
predictions.jsonl + DONE -> STOP) and preserves trida-stack's untimed SGLang rebuild path
(`gravityocr/sglang/` + `gravityocr/serve.sh`), while integrating:
1. Untimed All-in-VRAM preloading & multi-batch CUDA Graph warmup before READY.
2. Lock-free 64-worker AVX2 outer margin trimming (`text_px=None`, `damp=False`) + virtual grid
   for ruleless tables via `gravity_ocr`.
3. Task bucketing (`table -> text -> formula`) & patch-area descending sort for RadixAttention
   prompt KV cache sharing and zero tail-latency stragglers.
4. Auto-probed zero-compression `/dev/shm` BMP IPC (eliminates PNG zlib re-encoding & Base64
   round-trip, with automatic safe fallback to BMP/PNG data URLs if local file URLs are disabled).
5. Code-fence (```html, ```latex) & outer `$$` sanitizer + original manifest ID order `fsync`.

Tuning switches (environment variables):
  TRACK2_STAGE        : "0" (baseline pass-through) | "1" (preproc+bucket+shm)
                        | "2" (+ Q*K^T vocab slicing & rescue) | "3" (+ 1-hop RoPE pruning, default)
  TRACK2_CONCURRENCY  : concurrent SGLang requests (default "96", falls back to CONCURRENCY)
  TRACK2_SHM_IPC      : "1" (try /dev/shm BMP IPC, default) | "0" (in-memory data URL)
  TRACK2_VIRTUAL_GRID : "1" (Luma=180 virtual grid on ruleless tables, default) | "0"
  TRACK2_TRIM         : "1" (lossless outer whitespace trim with 6px halo, default) | "0"
  TRACK2_RESCUE       : "1" (Cauchy-Schwarz lossless rescue, default) | "0"
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
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
USE_VIRTUAL_GRID = STAGE >= 1 and os.environ.get("TRACK2_VIRTUAL_GRID", "1") == "1"
USE_TRIM = STAGE >= 1 and os.environ.get("TRACK2_TRIM", "1") == "1"
CPU_WORKERS = min(64, max(8, (os.cpu_count() or 8) * 2))

SHM_DIR = Path("/dev/shm/track2_crops")
if USE_SHM_IPC:
    try:
        SHM_DIR.mkdir(parents=True, exist_ok=True)
    except OSError:
        USE_SHM_IPC = False

# Active IPC transport decided during untimed warmup: "shm_file", "shm_path", "bmp_b64", or "png_b64"
IPC_MODE = "shm_file" if USE_SHM_IPC else ("bmp_b64" if STAGE >= 1 else "png_b64")


def log(msg: str) -> None:
    print(f"[track2-minion] {msg}", flush=True)


# ── Native AVX2 preprocessing module + lock-free Rayon thread pin (gk_set_threads(1)) ──
HAS_NATIVE_PREPROC = False
try:
    from gravity_ocr import native
    from gravity_ocr.drawing import crops

    threads = native.set_threads(1)
    HAS_NATIVE_PREPROC = True
    log(f"gravity_ocr loaded (native_threads={threads})")
except Exception as e:
    crops = None  # type: ignore[assignment]
    log(f"gravity_ocr fallback to numpy/PIL: {e}")

TASK_PROMPTS = {
    "table": "Table Recognition:",
    "text": "Text Recognition:",
    "formula": "Formula Recognition:",
}
TASK_BUCKET_ORDER = {"table": 0, "text": 1, "formula": 2}

FENCE_HEAD_RE = re.compile(r"^\s*```(?:html|markdown|md|latex|tex|text)?\s*\n?", re.IGNORECASE)
FENCE_TAIL_RE = re.compile(r"\n?\s*```\s*$", re.IGNORECASE)


def own(name: str, stock: str | None) -> str | None:
    """Return submission-local copy of `name` if present, else `stock`."""
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
        # All-in-VRAM & accelerator controls
        "CUDA_MODULE_LOADING": os.environ.get("CUDA_MODULE_LOADING", "EAGER"),
        "GRAVITY_OCR_RESIDENCY": os.environ.get("GRAVITY_OCR_RESIDENCY", "auto"),
        "GRAVITY_OCR_KV": os.environ.get("GRAVITY_OCR_KV", "f32"),
        "GRAVITY_OCR_WEIGHTS": os.environ.get("GRAVITY_OCR_WEIGHTS", "half"),
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
        for endpoint in ("/v1/models", "/health"):
            try:
                with urllib.request.urlopen(f"{URL}{endpoint}", timeout=3) as r:
                    if r.status == 200:
                        return
            except Exception:
                pass
        time.sleep(0.5)
    sys.exit("[track2-minion] server not ready in time")


def sanitize_output(text: str, task_key: str) -> str:
    """Strip spurious markdown code fences (```html/```latex) and outer $$ delimiters."""
    if not text or STAGE == 0:
        return text or ""
    cleaned = text.strip()
    if "```" in cleaned:
        cleaned = FENCE_HEAD_RE.sub("", cleaned)
        cleaned = FENCE_TAIL_RE.sub("", cleaned)
    if task_key == "formula" and cleaned.startswith("$$") and cleaned.endswith("$$") and len(cleaned) > 4:
        cleaned = cleaned[2:-2].strip()
    return cleaned.strip()


def trim_outer_whitespace(rgb: np.ndarray, pad: int = 6) -> np.ndarray:
    """Lossless outer paper whitespace trim (keeps original resolution & 6px halo)."""
    if HAS_NATIVE_PREPROC and crops is not None:
        return crops.trim_outer_whitespace(rgb, pad=pad)
    h, w, _ = rgb.shape
    if h <= 32 or w <= 32:
        return rgb
    ink_mask = rgb.min(axis=2) < 245
    if not np.any(ink_mask):
        return rgb
    rows = np.flatnonzero(ink_mask.any(axis=1))
    cols = np.flatnonzero(ink_mask.any(axis=0))
    r0, r1 = max(0, int(rows[0]) - pad), min(h, int(rows[-1]) + 1 + pad)
    c0, c1 = max(0, int(cols[0]) - pad), min(w, int(cols[-1]) + 1 + pad)
    if (r1 - r0) >= h - 4 and (c1 - c0) >= w - 4:
        return rgb
    return np.ascontiguousarray(rgb[r0:r1, c0:c1])


def inject_virtual_grid(rgb: np.ndarray) -> np.ndarray:
    """Draw 1px Luma=180 virtual grid in whitespace valleys of ruleless tables (`text_px=None, damp=False`)."""
    if not USE_VIRTUAL_GRID or not HAS_NATIVE_PREPROC or crops is None:
        return rgb
    try:
        return crops.prepare_crop(rgb, prompt="table", text_px=None, damp=False, grid=True).image
    except Exception:
        return rgb


def detect_task(item: dict[str, Any]) -> str:
    raw = f"{item.get('task', '')} {item.get('prompt', '')}".lower()
    if "table" in raw:
        return "table"
    if "formula" in raw:
        return "formula"
    return "text"


def encode_image_url(rgb: np.ndarray, img_path: Path, idx: int, cid: str, mode: str) -> str:
    if mode == "shm_file":
        shm_file = SHM_DIR / f"crop_{idx}_{cid}.bmp"
        Image.fromarray(rgb).save(shm_file, format="BMP")
        return f"file://{shm_file}"
    if mode == "shm_path":
        shm_file = SHM_DIR / f"crop_{idx}_{cid}.bmp"
        Image.fromarray(rgb).save(shm_file, format="BMP")
        return str(shm_file)
    if mode == "bmp_b64":
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="BMP")
        return "data:image/bmp;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    if STAGE == 0 and not USE_TRIM and not USE_VIRTUAL_GRID:
        raw = img_path.read_bytes()
        return "data:image/png;base64," + base64.b64encode(raw).decode("ascii")
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG", compress_level=1)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def preprocess_crop(item: dict[str, Any], job_dir: Path, idx: int, mode: str | None = None) -> dict[str, Any]:
    cid = item["id"]
    task_key = detect_task(item)
    prompt_str = item.get("prompt") or TASK_PROMPTS[task_key]
    img_path = job_dir / item["image"]
    active_mode = mode or IPC_MODE

    if STAGE == 0:
        b64 = base64.b64encode(img_path.read_bytes()).decode("ascii")
        return {
            "id": cid,
            "orig_idx": idx,
            "task_key": task_key,
            "bucket_rank": 0,
            "patch_area": 0,
            "prompt": prompt_str,
            "image_url": f"data:image/png;base64,{b64}",
        }

    with Image.open(img_path) as pil_img:
        rgb = np.asarray(pil_img.convert("RGB"))

    if USE_TRIM:
        rgb = trim_outer_whitespace(rgb, pad=6)
    if task_key == "table" and USE_VIRTUAL_GRID:
        rgb = inject_virtual_grid(rgb)

    h, w, _ = rgb.shape
    patch_area = ((h + 27) // 28) * ((w + 27) // 28)
    image_url = encode_image_url(rgb, img_path, idx, cid, active_mode)
    return {
        "id": cid,
        "orig_idx": idx,
        "task_key": task_key,
        "bucket_rank": TASK_BUCKET_ORDER[task_key],
        "patch_area": patch_area,
        "prompt": prompt_str,
        "image_url": image_url,
    }


def call_sglang(image_url: str, prompt: str, timeout: int = 300) -> str:
    payload = {
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
    }
    req = urllib.request.Request(
        f"{URL}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
        return body["choices"][0]["message"]["content"] or ""


def infer_crop(prep: dict[str, Any]) -> tuple[str, str]:
    cid = prep["id"]
    task_key = prep["task_key"]
    try:
        raw = call_sglang(prep["image_url"], prep["prompt"])
        return cid, sanitize_output(raw, task_key)
    except Exception as e:
        log(f"crop {cid} failed: {e}")
        return cid, ""


def cleanup_shm() -> None:
    if not SHM_DIR.is_dir():
        return
    for p in SHM_DIR.glob("crop_*.bmp"):
        try:
            p.unlink()
        except OSError:
            pass


def probe_ipc_mode(sample_item: dict[str, Any], folder: Path) -> str:
    """During untimed warmup, pick the fastest IPC transport accepted by SGLang."""
    if STAGE == 0:
        return "png_b64"
    candidates = ["shm_file", "shm_path", "bmp_b64", "png_b64"] if USE_SHM_IPC else ["bmp_b64", "png_b64"]
    for mode in candidates:
        try:
            prep = preprocess_crop(sample_item, folder, 0, mode=mode)
            _ = call_sglang(prep["image_url"], prep["prompt"], timeout=60)
            cleanup_shm()
            log(f"IPC transport selected: {mode}")
            return mode
        except Exception as e:
            log(f"IPC probe {mode} rejected ({e}); trying fallback...")
            cleanup_shm()
    return "png_b64"


def run_manifest(
    folder: Path,
    cpu_pool: ThreadPoolExecutor,
    gpu_pool: ThreadPoolExecutor,
    limit: int | None = None,
) -> list[tuple[str, str]]:
    manifest_path = folder / "manifest.jsonl"
    items = [json.loads(l) for l in manifest_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if limit is not None:
        items = items[:limit]
    if not items:
        return []

    t0 = time.perf_counter()
    prepped = list(cpu_pool.map(lambda pair: preprocess_crop(pair[1], folder, pair[0]), enumerate(items)))
    t_prep = time.perf_counter()

    if STAGE >= 1:
        scheduled = sorted(prepped, key=lambda x: (x["bucket_rank"], -x["patch_area"]))
    else:
        scheduled = prepped

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


def warmup(cpu_pool: ThreadPoolExecutor, gpu_pool: ThreadPoolExecutor) -> None:
    """Untimed warmup on $TRACK2_DEV_DIR: probe IPC transport and prime CUDA Graphs & RadixAttention."""
    global IPC_MODE
    dev_dir = Path(DEV) if DEV else None
    if not dev_dir or not (dev_dir / "manifest.jsonl").exists():
        return
    items = [json.loads(l) for l in (dev_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    if not items:
        return
    t0 = time.monotonic()
    IPC_MODE = probe_ipc_mode(items[0], dev_dir)
    # Pick balanced warmup samples across table / text / formula to warm up all prompt buckets and batch sizes
    warm_limit = min(len(items), max(32, CONCURRENCY))
    run_manifest(dev_dir, cpu_pool, gpu_pool, limit=warm_limit)
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
        with ThreadPoolExecutor(max_workers=CPU_WORKERS) as cpu_pool, ThreadPoolExecutor(max_workers=CONCURRENCY) as gpu_pool:
            warmup(cpu_pool, gpu_pool)
            Path(WORK, "READY").touch()
            log("READY signaled; polling jobs...")

            jobs_root = Path(WORK) / "jobs"
            seen: set[str] = set()
            stop_file = Path(WORK) / "STOP"
            while not stop_file.exists():
                if jobs_root.is_dir():
                    entries = sorted(os.listdir(jobs_root), key=lambda s: int(s.split("-")[0]) if s.split("-")[0].isdigit() else 0)
                    for name in entries:
                        if name in seen:
                            continue
                        job_dir = jobs_root / name
                        if not (job_dir / "manifest.jsonl").exists() or (job_dir / "DONE").exists():
                            continue
                        results = run_manifest(job_dir, cpu_pool, gpu_pool)
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
