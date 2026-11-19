#!/usr/bin/env python3
"""Assemble a clean, submission-ready Track 2 team folder (e.g. ~/my-team).

Copies ONLY the runtime submission assets (`team.yaml`, `serve.py`, `gravity_ocr/`,
`gravityocr/`, `vocab_slices/`), normalizes shell scripts to LF, sets executable bits,
and verifies `team.yaml` sits at the root.

Usage:
  python3 tools/assemble_team.py --out ~/my-team [--stage 3]
"""
from __future__ import annotations

import argparse
import os
import shutil
import stat
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def ignore_junk(_dir: str, names: list[str]) -> set[str]:
    bad = {"__pycache__", ".pytest_cache", "target", ".git"}
    return {n for n in names if n in bad or n.endswith((".pyc", ".pyo", ".dll", ".pdb"))}


def normalize_lf(path: Path) -> None:
    data = path.read_bytes().replace(b"\r\n", b"\n")
    path.write_bytes(data)


def main() -> None:
    ap = argparse.ArgumentParser(description="Assemble clean ~/my-team folder for local_run.py / submit.py")
    ap.add_argument("--out", required=True, help="Destination team directory (e.g. ~/my-team)")
    ap.add_argument("--stage", type=int, default=None, help="Optional TRACK2_STAGE override written to team.yaml")
    ap.add_argument("--concurrency", type=int, default=None, help="Optional TRACK2_CONCURRENCY override")
    args = ap.parse_args()

    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    # Copy top-level entrypoint files
    shutil.copy2(ROOT / "serve.py", out / "serve.py")
    normalize_lf(out / "serve.py")

    if args.stage is not None or args.concurrency is not None:
        env_parts = []
        if args.stage is not None:
            env_parts.append(f"TRACK2_STAGE={args.stage}")
        if args.concurrency is not None:
            env_parts.append(f"TRACK2_CONCURRENCY={args.concurrency}")
        cmd = f"env {' '.join(env_parts)} python3 serve.py"
        (out / "team.yaml").write_text(f"serve: {cmd}\n", encoding="utf-8", newline="\n")
    else:
        shutil.copy2(ROOT / "team.yaml", out / "team.yaml")
        normalize_lf(out / "team.yaml")

    # Copy runtime directories
    for folder in ("gravity_ocr", "gravityocr", "vocab_slices"):
        src = ROOT / folder
        dst = out / folder
        if not src.is_dir():
            continue
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst, ignore=ignore_junk)

    # Ensure Linux LF and +x on shell scripts
    for sh in out.rglob("*.sh"):
        normalize_lf(sh)
        if os.name != "nt":
            sh.chmod(sh.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    assert (out / "team.yaml").is_file() and (out / "serve.py").is_file()
    print(f"Assembled clean Track 2 team folder at: {out}", flush=True)
    for p in sorted(out.iterdir()):
        kind = "dir " if p.is_dir() else "file"
        print(f"  [{kind}] {p.name}", flush=True)


if __name__ == "__main__":
    main()
