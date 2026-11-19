"""Track 2: GLM-OCR ViT prefetch on a low-priority CUDA stream (SGLang scheduler process only).

Why: SGLang runs the vision tower inside every prefill forward, once per request
(mm_utils._get_chunked_embedding_by_item): each small crop pays its own graph replay / eager pass, and every prefill
stalls the decode batch for the whole ViT time (~45% of a run's GPU time on the dev set).

What:
  * The scheduler calls tick() once per event-loop iteration. Every newly queued request's image descriptors (the
    shm-ring descriptors written by the tokenizer manager, see processors/glm4v.py) are submitted in waiting-queue
    order and encoded in large packed batches (<= TRACK2_VIT_PF_PATCHES patches) on a dedicated LOW-priority stream.
    The schedule stream is created HIGH-priority (scheduler.py), so decode kernels win the SMs and the ViT fills the
    gaps between them (CPU bookkeeping between rounds, attention phases, small kernels).
  * get_image_feature() (models/glm_ocr.py) asks get_embeddings(items): the current stream waits on the batch's event
    and gets the stored embedding. An image that was never submitted is encoded on demand, on the same ViT stream.
  * Every ViT launch (graph replays included) runs on that ONE stream, from the scheduler thread, in submission order,
    so the graph runner's static input buffers are never shared across streams.
Embeddings are identical in math to the per-request path (packed FA2 varlen vs per-image SDPA: bf16 rounding only).

Env: TRACK2_VIT_PREFETCH (1 = on), TRACK2_VIT_PF_PATCHES (patches per packed batch), TRACK2_VIT_PF_IMAGES.
"""
from __future__ import annotations

import collections
import os
import time
from typing import Callable, List, Optional

import torch

ENABLED = os.environ.get("TRACK2_VIT_PREFETCH", "1") == "1"
_TRACE = os.environ.get("T2_PROF", "")  # diagnostic: per-tick timing lines (profiling runs only)
BATCH_PATCHES = int(os.environ.get("TRACK2_VIT_PF_PATCHES", "16384"))
BATCH_IMAGES = int(os.environ.get("TRACK2_VIT_PF_IMAGES", "64"))
# Batches queued on the ViT stream at once. Queuing everything fills the driver's launch queue: cudaLaunchKernel then
# blocks the scheduler thread (and every decode launch) until the GPU drains it.
MAX_INFLIGHT = int(os.environ.get("TRACK2_VIT_PF_INFLIGHT", "2"))
RANK_ORDER = os.environ.get("TRACK2_RANK_ORDER", "1") == "1"
# Pacing (0 = off): while the LM is decoding and at least this many waiting requests already have their image encoded
# (or in flight), no new ViT batch is launched. The ViT then stays just ahead of the LM's admissions instead of racing
# to the end of the queue, so its kernels are left to fill the LM's host-bound gaps late in the job as well.
READY_TARGET = int(os.environ.get("TRACK2_VIT_PF_READY_TARGET", "0"))
SCHED_PRIORITY = int(os.environ.get("TRACK2_SCHED_STREAM_PRIORITY", "-10"))

_PF: Optional["VitPrefetcher"] = None

try:  # multi-layer run log, ViT layer (L4)
    from gravity_ocr import t2log as _LOG
    if _LOG.LEVEL < 1:
        _LOG = None
except Exception:
    _LOG = None


def req_rank(req) -> int:
    """the client's scheduling rank, carried in the request id ("t2r<rank:05d>j<job>"); unranked requests go last."""
    rid = getattr(req, "rid", None)
    if isinstance(rid, str) and rid.startswith("t2r"):
        try:
            return int(rid[3:8])
        except ValueError:
            pass
    return 1 << 30


def is_desc(item) -> bool:
    f = getattr(item, "feature", None)
    return (
        isinstance(f, torch.Tensor)
        and f.dtype == torch.int64
        and f.dim() == 2
        and f.shape[0] == 1
        and f.shape[1] == 7
        and int(f[0, 0]) == -99
    )


def desc(item):
    _, off, nbytes, h, w, gh, gw = (int(v) for v in item.feature[0].tolist())
    return off, nbytes, h, w, gh, gw


class VitPrefetcher:
    def __init__(self, model, preprocess: Callable, log: Callable[[str], None] = print):
        """model: GlmOcrForConditionalGeneration (uses model.visual); preprocess(items) -> (pixel_values, grid_thw)
        enqueues the H2D copy / resize / patchify of the items on the CURRENT stream."""
        self.model = model
        self.preprocess = preprocess
        self.log = log
        # default 0 = the lowest priority (decode wins); TRACK2_VIT_STREAM_PRIORITY < 0 raises it
        self.stream = torch.cuda.Stream(device=model.visual.device,
                                        priority=int(os.environ.get("TRACK2_VIT_STREAM_PRIORITY", "0")))
        self.pending: collections.deque = collections.deque()
        self.inflight: collections.deque = collections.deque()  # events of launched, possibly unfinished batches
        self._ondemand_now = False
        self.n_batches = 0
        self.n_items = 0
        self.n_ondemand = 0
        self.n_patches = 0
        self.t_launch = 0.0
        self.n_paced = 0  # ticks on which the ready-pool target held a launch back
        self.gc_done = False

    # ------------------------------------------------------------------ submission (scheduler thread)
    def submit_reqs(self, reqs) -> None:
        added = False
        for req in reqs:
            if getattr(req, "_t2_pf", False):
                continue
            req._t2_pf = True
            mm = getattr(req, "multimodal_inputs", None)
            items = getattr(mm, "mm_items", None) if mm is not None else None
            if not items:
                continue
            rank = req_rank(req)
            for it in items:
                if getattr(it, "_t2_state", 0) == 0 and is_desc(it):
                    it._t2_state = 1  # queued
                    it._t2_rank = rank
                    self.pending.append(it)
                    added = True
        if added and len(self.pending) > 1:
            # encode in the client's rank order, not in the order the tokenizer manager happened to finish requests
            self.pending = collections.deque(sorted(self.pending, key=lambda it: it._t2_rank))

    def reap(self) -> None:
        while self.inflight and self.inflight[0].query():
            self.inflight.popleft()

    def pump(self) -> None:
        """Keep up to MAX_INFLIGHT packed batches queued on the ViT stream (non-blocking: completion is polled)."""
        self.reap()
        while self.pending and len(self.inflight) < MAX_INFLIGHT:
            batch, patches = [], 0
            while self.pending and len(batch) < BATCH_IMAGES:
                it = self.pending[0]
                gh, gw = desc(it)[4:6]
                m = gh * gw
                if batch and patches + m > BATCH_PATCHES:
                    break
                batch.append(self.pending.popleft())
                patches += m
            self.inflight.append(self._encode(batch))

    # ------------------------------------------------------------------ encoding (ViT stream)
    def _encode(self, items: List) -> None:
        # No wait on the schedule stream: the ViT reads only weights (loaded at startup), the page-locked shm ring and
        # its own stream's buffers (the caching allocator never hands this stream a block freed by another stream).
        t0 = time.perf_counter()
        with torch.cuda.stream(self.stream):
            ev0 = torch.cuda.Event(enable_timing=True)
            ev0.record(self.stream)
            pixel_values, grid_thw = self.preprocess(items)
            out = self.model.visual(pixel_values, grid_thw=grid_thw)
            counts = [int(g[1] * g[2]) // 4 for g in grid_thw.tolist()] if grid_thw.dim() == 2 else [out.shape[0]]
            parts = torch.split(out, counts, dim=0) if len(items) > 1 else (out,)
            ev = torch.cuda.Event(enable_timing=True)
            ev.record(self.stream)
        for it, e in zip(items, parts):
            it._t2_emb = e
            it._t2_ev = ev
            it._t2_state = 2  # encoded (launched)
        self.n_batches += 1
        self.n_items += len(items)
        self.n_patches += sum(int(g[1] * g[2]) for g in grid_thw.tolist())
        dt = time.perf_counter() - t0
        self.t_launch += dt
        if _LOG is not None:
            _LOG.SCHED.vit_batch(len(items), sum(int(g[1] * g[2]) for g in grid_thw.tolist()), 1e3 * dt, ev0, ev,
                                 self._ondemand_now)
        if self.n_batches in (1, 10, 100) or self.n_batches % 1000 == 0:
            self.log(f"[track2-vit-prefetch] batches={self.n_batches} images={self.n_items} patches={self.n_patches} "
                     f"on_demand={self.n_ondemand} host_launch={self.t_launch:.3f}s")
        return ev

    # ------------------------------------------------------------------ consumption (prefill forward)
    def get_embeddings(self, items: List) -> torch.Tensor:
        todo = [it for it in items if getattr(it, "_t2_state", 0) != 2]
        if todo:
            ids = {id(it) for it in todo}
            if self.pending:
                self.pending = collections.deque(it for it in self.pending if id(it) not in ids)
            self.n_ondemand += len(todo)
            self._ondemand_now = True
            try:
                for i in range(0, len(todo), BATCH_IMAGES):
                    self.inflight.append(self._encode(todo[i : i + BATCH_IMAGES]))
            finally:
                self._ondemand_now = False
        cur = torch.cuda.current_stream(self.model.visual.device)
        seen = set()
        embs = []
        for it in items:
            ev = it._t2_ev
            if id(ev) not in seen:
                seen.add(id(ev))
                cur.wait_event(ev)
            e = it._t2_emb
            e.record_stream(cur)
            embs.append(e)
        return embs[0] if len(embs) == 1 else torch.cat(embs, dim=0)

    def run_on_stream(self, fn: Callable, *args):
        """Any other ViT call (non-descriptor items): same stream, the caller's stream waits for it."""
        cur = torch.cuda.current_stream(self.model.visual.device)
        self.stream.wait_stream(cur)
        with torch.cuda.stream(self.stream):
            out = fn(*args)
            ev = torch.cuda.Event()
            ev.record(self.stream)
        cur.wait_event(ev)
        if isinstance(out, torch.Tensor):
            out.record_stream(cur)
        return out


def ready(req) -> bool:
    """True when every descriptor image of the request has been encoded (non-blocking cudaEventQuery)."""
    mm = getattr(req, "multimodal_inputs", None)
    items = getattr(mm, "mm_items", None) if mm is not None else None
    if not items:
        return True
    for it in items:
        st = getattr(it, "_t2_state", 0)
        if st == 0:
            continue  # not a prefetched item (never submitted): computed in the forward as before
        if st != 2 or not it._t2_ev.query():
            return False
    return True


def wait_first(req) -> None:
    """Nothing is running: block until the first waiting request's ViT batch is done (it is admitted next)."""
    mm = getattr(req, "multimodal_inputs", None)
    for it in (getattr(mm, "mm_items", None) or []) if mm is not None else []:
        if getattr(it, "_t2_state", 0) == 2:
            it._t2_ev.synchronize()


def launched(req) -> bool:
    """True when every descriptor image of the request has been launched on the ViT stream (done or in flight)."""
    mm = getattr(req, "multimodal_inputs", None)
    items = getattr(mm, "mm_items", None) if mm is not None else None
    if not items:
        return True
    return all(getattr(it, "_t2_state", 0) in (0, 2) for it in items)


def register(pf: VitPrefetcher) -> None:
    global _PF
    _PF = pf


def get() -> Optional[VitPrefetcher]:
    return _PF


def tick(sched) -> None:
    """Scheduler hook (event loop, after process_input_requests): submit newly queued requests, launch their ViT."""
    pf = _PF
    if pf is None:
        return
    if getattr(pf.model, "_t2_capture_deferred", False):
        pf.model._t2_capture_if_deferred()  # quantized ViT: capture now (before the first request, untimed)
    t0 = time.perf_counter() if _TRACE else 0.0
    wq = getattr(sched, "waiting_queue", None)
    n0 = pf.n_items
    if wq and not pf.gc_done:
        pf.gc_done = True  # startup (load, graph capture) is over: no cyclic-GC pauses inside jobs from here on
        try:
            from gravity_ocr import gcguard
            gcguard.freeze("scheduler")
        except Exception as e:
            pf.log(f"[track2-gc] scheduler: not applied ({e})")
    if wq:
        if RANK_ORDER and len(wq) > 1:
            # admission follows the client's rank (FCFS without priority scheduling never reorders the queue itself)
            wq.sort(key=req_rank)
        pf.submit_reqs(wq)
    if pf.pending:
        if (READY_TARGET > 0 and wq and sched.running_batch.reqs
                and sum(1 for r in wq if launched(r)) >= READY_TARGET):
            pf.reap()
            pf.n_paced += 1  # enough encoded requests are waiting for LM slots: leave the GPU to the decode for now
        else:
            pf.pump()
    t1 = time.perf_counter() if _TRACE else 0.0
    waited = False
    if wq and sched.chunked_req is None and not sched.running_batch.reqs and not ready(wq[0]):
        tw = time.perf_counter()
        wait_first(wq[0])  # idle GPU queue otherwise: nothing to decode until the first embeddings exist
        waited = True
        if _LOG is not None:
            _LOG.SCHED.vit_wait(1e3 * (time.perf_counter() - tw))
    if _TRACE:
        t2 = time.perf_counter()
        if pf.n_items != n0 or waited or (t2 - t0) > 0.005:
            pf.log(f"[t2-vit-tick] t={t0:.4f} launched={pf.n_items - n0} launch_ms={1e3 * (t1 - t0):.1f} "
                   f"wait_first={int(waited)} wait_ms={1e3 * (t2 - t1):.1f} wq={len(wq) if wq else 0} "
                   f"run={len(sched.running_batch.reqs)}")
