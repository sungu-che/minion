#!/usr/bin/env python3
"""Patch SGLang scheduler.py with deadlock-free Prefill Coalescing for Track 2.

Why:
  Without prefill coalescing, as TokenizerManager streams preprocessed images over ZMQ one by one,
  `_get_new_batch_prefill_raw` preempts active decoding whenever `len(waiting_queue) >= 1`:
    1. Runs ~380 single-image `bs=1` EXTEND passes (reading 2.50 GB of CogViT + GLM4 weights 380 times = 950 GB HBM!).
    2. Calls `ss_release_stash("prefill admission")` on every single arrival, discarding the pre-launched
       `SELFSPEC_DRAFT_AHEAD=1` speculative draft forward!

Fix (100% deadlock-free):
  - When `running_bs > 0` and `chunked_req is None`: if `len(waiting_queue) < min(TRACK2_PREFILL_COALESCE, free_slots)`
    and time since first deferred arrival `< TRACK2_PREFILL_MAX_WAIT_S` (default 25ms), return `None` BEFORE
    `ss_release_stash` so `running_batch` runs uninterrupted decode (preserving draft-ahead stash!) while
    `waiting_queue` accumulates a full wave of 16-32 images!
  - Never returns `None` when `running_bs == 0` (so the scheduler event loop can never block with queued items).
"""
from __future__ import annotations

import argparse
import py_compile
from pathlib import Path

MARKER = "# [TRACK2-PREFILL-COALESCE]"


def patch_scheduler(src: str) -> str:
    if MARKER in src:
        return src
    target = """        if (
            self.running_batch.batch_is_full or len(self.waiting_queue) == 0
        ) and self.chunked_req is None:
            return None

        running_bs = len(self.running_batch.reqs)"""
    repl = f"""        if (
            self.running_batch.batch_is_full or len(self.waiting_queue) == 0
        ) and self.chunked_req is None:
            return None

        running_bs = len(self.running_batch.reqs)

        {MARKER}
        _t2_coal = int(__import__("os").environ.get("TRACK2_PREFILL_COALESCE", "16"))
        if _t2_coal > 1 and self.chunked_req is None and running_bs > 0 and not self.running_batch.batch_is_full:
            _free = max(1, self.max_running_requests - running_bs)
            _need = min(_t2_coal, _free)
            if len(self.waiting_queue) < _need:
                _now = __import__("time").monotonic()
                _t0 = getattr(self, "_t2_wait_t0", None)
                if _t0 is None:
                    self._t2_wait_t0 = _now
                    return None
                _max_w = float(__import__("os").environ.get("TRACK2_PREFILL_MAX_WAIT_S", "0.025"))
                if (_now - _t0) < _max_w:
                    return None
        self._t2_wait_t0 = None"""
    if target not in src:
        raise RuntimeError("Could not locate running_bs anchor in scheduler.py")
    return src.replace(target, repl, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Patch SGLang scheduler.py with Prefill Coalescing")
    ap.add_argument("--scheduler-src", required=True)
    ap.add_argument("--out-sglang-py", required=True)
    args = ap.parse_args()

    out_path = Path(args.out_sglang_py) / "sglang/srt/managers/scheduler.py"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    patched = patch_scheduler(Path(args.scheduler_src).read_text(encoding="utf-8"))
    out_path.write_text(patched, encoding="utf-8", newline="\n")
    py_compile.compile(str(out_path), doraise=True)
    print(f"Patched & verified: {out_path}", flush=True)


if __name__ == "__main__":
    main()
