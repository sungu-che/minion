#!/usr/bin/env python3
"""Patch SGLang scheduler.py with slot-based Prefill Coalescing for Track 2.

Why:
  The client queues every crop at once, so SGLang's waiting queue is never empty. Upstream admits a prefill as soon as
  ONE running request finishes: dozens of tiny prefill batches (a few crops each), each one a separate ViT + LLM extend
  launch, a host sync, and a discarded SELFSPEC_DRAFT_AHEAD draft (the stash is released before every admission).

Fix (deadlock-free):
  While requests are decoding (running_bs > 0, no chunked request), a prefill is admitted only when at least
  min(TRACK2_PREFILL_COALESCE, len(waiting_queue)) slots are free, or when the first deferral is older than
  TRACK2_PREFILL_MAX_WAIT_S. With nothing running the prefill is never deferred, so the loop cannot stall.
  Scheduler._t2_prefill_deferred(running_bs) answers the same rule for the self-spec worker, which then keeps
  launching the next draft ahead (no admission will release it).
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
        if self.chunked_req is None and self._t2_prefill_deferred(running_bs, arm=True):
            return None
        self._t2_wait_t0 = None"""
    if target not in src:
        raise RuntimeError("Could not locate running_bs anchor in scheduler.py")
    src = src.replace(target, repl, 1)

    # [TRACK2-VIT-PREFETCH] high-priority schedule stream (decode wins the SMs; the ViT prefetch stream is low priority)
    target_stream = "        self.schedule_stream = self.device_module.Stream(priority=0)"
    repl_stream = """        # [TRACK2-VIT-PREFETCH] decode / prefill run on a HIGH-priority stream; the ViT prefetch stream (gravity_ocr/vit_prefetch.py)
        # keeps the default (lowest) priority and fills the SMs the decode kernels leave idle
        _t2_prio = 0
        if __import__("os").environ.get("TRACK2_VIT_PREFETCH", "1") == "1" and self.device == "cuda":
            _t2_prio = int(__import__("os").environ.get("TRACK2_SCHED_STREAM_PRIORITY", "-10"))
        self.schedule_stream = self.device_module.Stream(priority=_t2_prio)"""
    if target_stream not in src:
        raise RuntimeError("Could not locate schedule_stream creation in scheduler.py")
    src = src.replace(target_stream, repl_stream, 1)

    # [TRACK2-VIT-PREFETCH] one tick per event-loop iteration, right after new requests are queued
    target_loop = """    def event_loop_normal(self):
        \"\"\"A normal scheduler loop.\"\"\"
        while True:
            # Receive requests
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)"""
    repl_loop = """    def event_loop_normal(self):
        \"\"\"A normal scheduler loop.\"\"\"
        _t2_tick = None
        self._t2_recv_buf = []
        _t2_recv_lim = int(__import__("os").environ.get("TRACK2_RECV_LIMIT", "24"))
        if __import__("os").environ.get("TRACK2_VIT_PREFETCH", "1") == "1":
            try:
                from gravity_ocr import vit_prefetch as _t2_vp
                _t2_vp.register_sched(self)
                _t2_tick = _t2_vp.tick
            except Exception as _e:
                logger.warning(f"[track2-vit-prefetch] tick unavailable: {_e}")
        while True:
            # Receive requests (prepend any already unpickled behind GPU decode, and cap burst while GPU is busy)
            if (_t2_recv_lim > 0 and self.running_batch.reqs
                    and self.pp_rank == 0 and self.attn_tp_rank == 0 and self.attn_cp_rank == 0):
                import zmq as _t2_zmq
                recv_reqs = self._t2_recv_buf
                self._t2_recv_buf = []
                _rem = _t2_recv_lim
                while _rem > 0:
                    try:
                        _r = self.recv_from_tokenizer.recv_pyobj(_t2_zmq.NOBLOCK)
                        unwrap_shm_features(_r)
                        recv_reqs.append(_r)
                        _rem -= 1
                    except _t2_zmq.ZMQError:
                        break
                while True:
                    try:
                        _r = self.recv_from_rpc.recv_pyobj(_t2_zmq.NOBLOCK)
                        unwrap_shm_features(_r)
                        recv_reqs.append(_r)
                    except _t2_zmq.ZMQError:
                        break
            else:
                recv_reqs = self.recv_requests()
                if self._t2_recv_buf:
                    recv_reqs = self._t2_recv_buf + recv_reqs
                    self._t2_recv_buf = []
            self.process_input_requests(recv_reqs)
            if _t2_tick is not None:
                _t2_tick(self)"""
    if target_loop not in src:
        raise RuntimeError("Could not locate event_loop_normal head in scheduler.py")
    src = src.replace(target_loop, repl_loop, 1)

    # [TRACK2-VIT-PREFETCH] module-level readiness probe (non-blocking cudaEventQuery on the request's ViT batch)
    target_logger = "logger = logging.getLogger(__name__)\n"
    repl_logger = target_logger + """
# [TRACK2-VIT-PREFETCH]
_t2_vit_ready = None
if __import__("os").environ.get("TRACK2_VIT_PREFETCH", "1") == "1":
    try:
        from gravity_ocr.vit_prefetch import ready as _t2_vit_ready
    except Exception:
        _t2_vit_ready = None
"""
    if target_logger not in src:
        raise RuntimeError("Could not locate the module logger in scheduler.py")
    src = src.replace(target_logger, repl_logger, 1)

    # admit only requests whose ViT embeddings are done while others are decoding (FCFS order kept: stop at the first)
    target_admit = """        # Get requests from the waiting queue to a new prefill batch
        for req in self.waiting_queue:
"""
    repl_admit = """        # Get requests from the waiting queue to a new prefill batch
        for req in self.waiting_queue:
            # [TRACK2-VIT-PREFETCH] a request without its ViT embeddings would stall this prefill (and decode) on the ViT
            if _t2_vit_ready is not None and (self.running_batch.reqs or adder.can_run_list) and not _t2_vit_ready(req):
                break
"""
    if target_admit not in src:
        raise RuntimeError("Could not locate the waiting-queue admission loop in scheduler.py")
    src = src.replace(target_admit, repl_admit, 1)

    anchor = "    def _get_new_batch_prefill_raw("
    helper = f"""    {MARKER}
    _t2_coal = int(__import__("os").environ.get("TRACK2_PREFILL_COALESCE", "1"))
    _t2_max_wait = float(__import__("os").environ.get("TRACK2_PREFILL_MAX_WAIT_S", "0.02"))
    _t2_wait_t0 = None

    def _t2_prefill_deferred(self, running_bs: int, arm: bool = False) -> bool:
        \"\"\"True while a prefill admission should wait (requests still decoding): the head of the waiting queue has no
        ViT embeddings yet ([TRACK2-VIT-PREFETCH]), or fewer than TRACK2_PREFILL_COALESCE slots are free.\"\"\"
        if running_bs <= 0 or self.chunked_req is not None:
            return False
        if _t2_vit_ready is not None and self.waiting_queue and not _t2_vit_ready(self.waiting_queue[0]):
            return True
        if self._t2_coal <= 1:
            return False
        free = self.max_running_requests - running_bs
        need = min(self._t2_coal, len(self.waiting_queue))
        if need <= 0 or free >= need:
            return False
        now = __import__("time").monotonic()
        t0 = self._t2_wait_t0
        if t0 is None:
            if arm:
                self._t2_wait_t0 = now
            return True
        return (now - t0) < self._t2_max_wait

"""
    if anchor not in src:
        raise RuntimeError("Could not locate _get_new_batch_prefill_raw in scheduler.py")
    return src.replace(anchor, helper + anchor, 1)


def patch_output_processor(src: str) -> str:
    """Hide ViT kernel launch and ZMQ unpickling behind EXTEND's GPU wait in process_batch_result_prefill."""
    marker = "# [TRACK2-EXTEND-HIDE]"
    if marker in src:
        return src
    hdr_anchor = "logger = logging.getLogger(__name__)\n"
    if hdr_anchor not in src:
        raise RuntimeError("Could not locate logger in scheduler_output_processor_mixin.py")
    hdr = hdr_anchor + """
# [TRACK2-EXTEND-HIDE] hide ViT launch & ZMQ unpickling behind EXTEND's GPU completion wait
_T2_EXTEND_HIDE = __import__("os").environ.get("TRACK2_EXTEND_HIDE", "1") == "1"
try:
    from gravity_ocr.vit_prefetch import pump_behind_decode as _t2_ext_pump
except Exception:
    _t2_ext_pump = None
"""
    src = src.replace(hdr_anchor, hdr, 1)
    old = """        if self.is_generation:
            if result.copy_done is not None:
                result.copy_done.synchronize()"""
    new = """        if self.is_generation:
            if _T2_EXTEND_HIDE and _t2_ext_pump is not None:  # [TRACK2-EXTEND-HIDE]
                _ev = result.copy_done
                if _ev is None:
                    _ev = getattr(self, "_t2_ext_ev", None)
                    if _ev is None:
                        _ev = self._t2_ext_ev = torch.cuda.Event()
                    _ev.record(torch.cuda.current_stream())
                _t2_ext_pump(_ev)
            if result.copy_done is not None:
                result.copy_done.synchronize()"""
    if src.count(old) != 1:
        raise RuntimeError(f"Expected 1 process_batch_result_prefill anchor, found {src.count(old)}")
    return src.replace(old, new, 1)


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
