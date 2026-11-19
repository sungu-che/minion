"""Build the native libraries and put them where ``gravity_ocr`` finds them.

    python tools/build_kernels.py                 # CPU kernels + GPU engine (every backend found)
    python tools/build_kernels.py --gpu none      # CPU kernels only
    python tools/build_kernels.py --gpu rocm,vulkan

``rust_kernels`` (CPU, rayon + SIMD) needs only cargo.  ``rust_gpu`` builds on
candle-rocm-vulkan: ``vulkan`` needs no SDK, ``rocm`` needs the HIP SDK / ROCm
(``HIP_PATH`` / ``ROCM_PATH``), ``cuda`` the CUDA toolkit.  The libraries are
copied to ``gravity_ocr/_native/`` (not tracked by git).

Windows note: the clang of the HIP SDK 7.x does not accept the C++ headers of
MSVC 14.50+ (``isgreater`` overload errors).  When such a toolset is the newest
one installed and an older one exists, ``VCToolsInstallDir`` is pointed at the
older toolset for the build (unless it is already set).  For CUDA the MSVC
environment, ``NVCC_CCBIN``, ``NVCC_PREPEND_FLAGS`` and ``CUDA_COMPUTE_CAP`` are
set up when missing (``cuda_env``).
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "gravity_ocr" / "_native"


def lib_name(stem: str) -> str:
    if sys.platform == "win32":
        return f"{stem}.dll"
    if sys.platform == "darwin":
        return f"lib{stem}.dylib"
    return f"lib{stem}.so"


def hip_root() -> Path | None:
    for var in ("ROCM_PATH", "HIP_PATH"):
        v = os.environ.get(var)
        if v and Path(v).is_dir():
            return Path(v)
    fixed = Path("C:/Program Files/AMD/ROCm") if sys.platform == "win32" else Path("/opt/rocm")
    if sys.platform == "win32" and fixed.is_dir():
        found = sorted(p for p in fixed.iterdir() if p.is_dir())
        return found[-1] if found else None
    return fixed if fixed.is_dir() else None


def cuda_root() -> Path | None:
    v = os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME")
    if v and Path(v).is_dir():
        return Path(v)
    return Path("/usr/local/cuda") if Path("/usr/local/cuda").is_dir() else None


def msvc_for_hip(env: dict) -> None:
    """Point clang at an MSVC toolset older than 14.50 when the newest one is 14.50+."""
    if sys.platform != "win32" or env.get("VCToolsInstallDir"):
        return
    toolsets = []
    for base in (Path("C:/Program Files/Microsoft Visual Studio"), Path("C:/Program Files (x86)/Microsoft Visual Studio")):
        if base.is_dir():
            toolsets += [p for p in base.glob("*/*/VC/Tools/MSVC/*") if p.is_dir()]

    def version(p: Path) -> tuple:
        return tuple(int(x) for x in re.findall(r"\d+", p.name)[:3])

    toolsets.sort(key=version)
    if not toolsets or version(toolsets[-1]) < (14, 50):
        return
    older = [p for p in toolsets if version(p) < (14, 50)]
    if older:
        env["VCToolsInstallDir"] = str(older[-1]) + os.sep
        print(f"  MSVC {toolsets[-1].name} is not accepted by HIP clang; using {older[-1].name}")


def _vcvars() -> Path | None:
    """``vcvars64.bat`` of the newest Visual Studio (or Build Tools) with the C++ tools."""
    found = []
    for base in (Path("C:/Program Files/Microsoft Visual Studio"), Path("C:/Program Files (x86)/Microsoft Visual Studio")):
        if base.is_dir():
            found += [p for p in base.glob("*/*/VC/Auxiliary/Build/vcvars64.bat") if p.is_file()]
    return sorted(found)[-1] if found else None


def cuda_env(env: dict) -> None:
    """Windows: what nvcc needs to compile the CUDA kernels of candle, unless already set.

    * the MSVC environment (``vcvars64.bat``) when ``cl.exe`` is not on the PATH;
    * ``NVCC_CCBIN``: the ``cl.exe`` of that environment;
    * ``NVCC_PREPEND_FLAGS=-Xcompiler /Zc:preprocessor``: CUDA 13's CCCL headers need the
      standard preprocessor of MSVC;
    * ``CUDA_COMPUTE_CAP`` from ``nvidia-smi`` (kernels only for the GPU that is there).
    """
    if not env.get("CUDA_COMPUTE_CAP"):
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
            cap = out.stdout.strip().splitlines()[0].replace(".", "") if out.returncode == 0 and out.stdout.strip() else ""
            if cap.isdigit():
                env["CUDA_COMPUTE_CAP"] = cap
                print(f"  CUDA_COMPUTE_CAP={cap} (nvidia-smi)")
        except (OSError, subprocess.TimeoutExpired):
            pass
    if sys.platform != "win32":
        return
    if shutil.which("cl", path=env.get("PATH")) is None:
        bat = _vcvars()
        if bat is None:
            print("  warning: cl.exe not found and no vcvars64.bat; nvcc will fail")
            return
        out = subprocess.run(f'"{bat}" >nul && set', shell=True, capture_output=True, text=True, errors="replace")
        for line in out.stdout.splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                env[k] = v
        print(f"  MSVC environment: {bat}")
    if not env.get("NVCC_CCBIN"):
        cl = shutil.which("cl", path=env.get("PATH"))
        if cl:
            env["NVCC_CCBIN"] = cl
            print(f"  NVCC_CCBIN={cl}")
    env.setdefault("NVCC_PREPEND_FLAGS", "-Xcompiler /Zc:preprocessor")


def cargo(crate: Path, features: list[str], env: dict) -> Path:
    cmd = ["cargo", "build", "--release"]
    if features:
        cmd += ["--features", ",".join(features)]
    print(f"$ ({crate.name}) {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=crate, env=env, check=True)
    return crate / "target" / "release" / lib_name(crate_lib(crate))


def crate_lib(crate: Path) -> str:
    return "gravity_kernels" if crate.name == "rust_kernels" else "gravity_gpu"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="auto", help="auto | none | comma list of rocm, vulkan, cuda")
    ap.add_argument("--no-cpu", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    built = []
    if not args.no_cpu:
        built.append(cargo(ROOT / "rust_kernels", [], env))
    if args.gpu != "none":
        if args.gpu == "auto":
            feats = ["vulkan"]
            if hip_root():
                feats.append("rocm")
            if cuda_root():
                feats.append("cuda")
        else:
            feats = [f.strip() for f in args.gpu.split(",") if f.strip()]
        if "cuda" in feats:
            cuda_env(env)
        if "rocm" in feats:
            msvc_for_hip(env)
        built.append(cargo(ROOT / "rust_gpu", sorted(feats), env))
    for lib in built:
        shutil.copy2(lib, OUT / lib.name)
        print(f"  -> {OUT / lib.name}")


if __name__ == "__main__":
    main()
