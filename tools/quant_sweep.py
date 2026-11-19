"""Decide the precision of every part of the model by measuring it.

    python tools/quant_sweep.py screen --schemes q4,q3          # one group at a time
    python tools/quant_sweep.py config plan uniform:q8 uniform:q4 build/quant/alloc.json

``screen`` swaps one group of tensors (a band of layers × one projection) to a
scheme, keeps everything else at the original weights, and scores the model by
teacher forcing on the calibration crops: the crops' reference outputs (the
greedy reading with the original weights) are fed back and the mean negative
log-likelihood of each reference token and the share of positions whose argmax
changes ("flips") are recorded.  ``config`` scores a complete assignment the
same way.  Everything runs on the GPU engine; the quantised tensors come from
``build/quant/<scheme>.gocr`` (``tools/quant_cache.py``).

The reference outputs come from ``build/eval/<set>-rocm-up30-slice.json`` (the
original weights, output head sliced to 16,384 rows).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gravity_ocr import language, native, preprocess, vision  # noqa: E402
from gravity_ocr.container import Package, PackageWriter  # noqa: E402
from gravity_ocr.drawing import crops  # noqa: E402
from gravity_ocr.model import TABLE_PROMPT, TEXT_PROMPT, GravityOCR  # noqa: E402

QUANT = ROOT / "build" / "quant"
PROMPTS = {"text": TEXT_PROMPT, "table": TABLE_PROMPT}
V, L = vision.PREFIX, language.PREFIX
VIS_BANDS = [(0, 0), (1, 7), (8, 15), (16, 22), (23, 23)]
VIS_KINDS = {"qkv": ["attn.qkv.weight"], "proj": ["attn.proj.weight"], "mlp_in": ["mlp.gate_proj.weight", "mlp.up_proj.weight"],
             "mlp_out": ["mlp.down_proj.weight"]}  # fmt: skip
LM_BANDS = [(0, 1), (2, 5), (6, 9), (10, 13), (14, 15)]
LM_KINDS = {"q": ["self_attn.q_proj.weight"], "k": ["self_attn.k_proj.weight"], "v": ["self_attn.v_proj.weight"],
            "o": ["self_attn.o_proj.weight"], "gate_up": ["mlp.gate_up_proj.weight"], "down": ["mlp.down_proj.weight"]}  # fmt: skip


def groups() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for a, b in VIS_BANDS:
        for kind, names in VIS_KINDS.items():
            out[f"vis{a}-{b}.{kind}"] = [f"{V}blocks.{i}.{n}" for i in range(a, b + 1) for n in names]
    out["vis.downsample"] = [f"{V}downsample.weight"]
    out["vis.merger"] = [f"{V}merger.{n}.weight" for n in ("proj", "gate_proj", "up_proj", "down_proj")]
    for a, b in LM_BANDS:
        for kind, names in LM_KINDS.items():
            out[f"lm{a}-{b}.{kind}"] = [f"{L}layers.{i}.{n}" for i in range(a, b + 1) for n in names]
    out["lm.head"] = [language.HEAD]
    return out


def plan_config() -> dict[str, str]:
    """The four tiers of the plan (Part 4.2), tensor by tensor; tensors the plan does not name → q6."""
    cfg = {}
    for name in sum(groups().values(), []):
        cfg[name] = "q6"
    for name in cfg:
        if name.startswith(V + "merger.") or name == language.HEAD:
            cfg[name] = "q8"  # tier 2 (vision_align and token_embd are not matrices of this model / stored apart)
        elif name.startswith(V + "blocks."):
            i = int(name.split(".")[3])
            if i in (0, 23) or ".attn.qkv." in name:
                cfg[name] = "q6"  # tier 3
            elif ".mlp.gate_proj." in name or ".mlp.up_proj." in name:
                cfg[name] = "q4"  # tier 4: "fc1" of the mid blocks
        elif name.startswith(L + "layers."):
            i = int(name.split(".")[3])
            if i in (0, 1, 14, 15) or ".v_proj." in name or ".down_proj." in name:
                cfg[name] = "q6"  # tier 3
            elif any(k in name for k in (".q_proj.", ".k_proj.", ".gate_up_proj.")):
                cfg[name] = "q4"  # tier 4 (gate_proj and up_proj are one fused matrix here)
    return cfg


class Bench:
    """The reference model on the GPU, the calibration samples and their reference outputs."""

    def __init__(self, sets: list[str], device: str = "rocm", limit: int = 0, every: int = 1):
        t0 = time.perf_counter()
        self.model = GravityOCR.from_pretrained(ROOT / "models", device=device)
        if self.model.gpu is None:
            raise SystemExit("the sweep needs a GPU device")
        self.ids = np.load(QUANT / "head_ids.npy")
        self.model.slice_head(self.ids)
        src = self.model.src
        self.original = {}  # tensor → (parts, scheme, shape) of the original weights
        for name in sum(groups().values(), []):
            if name == language.HEAD:
                parts, scheme, shape = native.parts_of(src, name)
                rows = np.ascontiguousarray(np.asarray(parts["w"]).reshape(shape[0], -1)[self.ids])
                self.original[name] = ({"w": rows}, scheme, (self.ids.size, shape[1]))
            else:
                self.original[name] = native.parts_of(src, name)
        self.packages = {}
        self.samples = []
        for set_name in sets:
            ref = {r["id"]: r for r in json.loads((ROOT / f"build/eval/{set_name}-rocm-up30-slice.json").read_text(encoding="utf-8"))["rows"]}
            info = json.loads((ROOT / f"build/synth/{set_name}/samples.json").read_text(encoding="utf-8"))
            if limit:
                info = info[:limit]
            info = info[::every]
            for s in info:
                img = Image.open(ROOT / f"build/synth/{set_name}" / s["image"]).convert("RGB")
                prep = crops.prepare_crop(img, prompt=s["prompt"], text_px=30)
                patches, grid = preprocess.to_patches(preprocess.fit_image(prep.image))
                self.samples.append({"id": s["id"], "patches": patches, "grid": grid, "prompt": prep.prompt, "target": ref[s["id"]]["tokens"]})
        self.rows = [self.model.head_ids.searchsorted(np.asarray(s["target"], np.int64)) for s in self.samples]
        for s, r in zip(self.samples, self.rows):
            if not np.array_equal(self.model.head_ids[r], s["target"]):
                raise SystemExit(f"{s['id']}: a reference token is outside the sliced head")
        self.ref_tokens = self.encode_all()
        self.base = self.score(self.ref_tokens)
        print(f"bench: {len(self.samples)} samples, {sum(len(t) for t in self.rows)} target tokens, "
              f"base nll {self.base['nll']:.5f} flips {self.base['flips']:.4f}, ready in {time.perf_counter() - t0:.0f}s", flush=True)  # fmt: skip

    def package(self, scheme: str) -> Package:
        if scheme not in self.packages:
            self.packages[scheme] = Package(QUANT / f"{scheme}.gocr")
        return self.packages[scheme]

    def set(self, name: str, scheme: str) -> None:
        if scheme == "orig":
            parts, sch, shape = self.original[name]
        else:
            pkg = self.package(scheme)
            parts, sch, shape = pkg.parts(name), pkg.scheme(name), pkg.shape(name)
        target = self.model.vision if name.startswith(V) else self.model.lm
        target.set_parts(name, parts, sch, shape)

    def encode_all(self, batch_patches: int = 12_288) -> list[np.ndarray]:
        """Image tokens of every sample; consecutive crops share one batched vision pass on a GPU."""
        vis = self.model.vision
        if not hasattr(vis, "forward_batch"):
            return [vis.forward(s["patches"], s["grid"]) for s in self.samples]
        out, i = [], 0
        while i < len(self.samples):
            j, total = i, 0
            while j < len(self.samples) and (j == i or total + self.samples[j]["patches"].shape[0] <= batch_patches):
                total += self.samples[j]["patches"].shape[0]
                j += 1
            group = self.samples[i:j]
            if len(group) == 1:
                out.append(vis.forward(group[0]["patches"], group[0]["grid"]))
            else:
                out += vis.forward_batch([(s["patches"], s["grid"], None) for s in group])
            i = j
        return out

    def score(self, image_tokens: list[np.ndarray], inject: list | None = None) -> dict:
        """Teacher-forced NLL / flips; ``inject[i]`` (rows of the outputs) is added after layer 0."""
        m = self.model
        nll = flips = n = 0.0
        for idx, (s, toks, rows) in enumerate(zip(self.samples, image_tokens, self.rows)):
            ids = m.prompt_ids(s["prompt"], toks.shape[0])
            head = ids.index(m.vocab.image_id)
            tail = len(ids) - head - toks.shape[0]
            x = m.embed(ids + s["target"][:-1])
            x[head : head + toks.shape[0]] = toks
            grid = (s["grid"][0] // preprocess.MERGE, s["grid"][1] // preprocess.MERGE)
            pos, nxt = m.prompt_positions(head, grid, tail)
            k = len(s["target"]) - 1
            pos = np.concatenate([pos, (nxt + np.arange(k))[:, None].repeat(3, axis=1)])
            cache = m.lm.new_cache(len(ids) + k + 16)
            extra = None
            if inject is not None and inject[idx] is not None:
                extra = np.zeros((len(ids) + k, x.shape[1]), np.float32)
                extra[len(ids) :] = inject[idx]
            logits = m.run(x, pos, cache, start=len(ids) - 1, inject=extra).astype(np.float64)
            logits -= logits.max(axis=1, keepdims=True)
            lse = np.log(np.exp(logits).sum(axis=1))
            nll += float((lse - logits[np.arange(len(rows)), rows]).sum())
            flips += float((logits.argmax(axis=1) != rows).sum())
            n += len(rows)
        return {"nll": nll / n, "flips": flips / n, "tokens": int(n)}

    def evaluate(self, assignment: dict[str, str], vision_changed: bool) -> dict:
        for name, scheme in assignment.items():
            self.set(name, scheme)
        toks = self.encode_all() if vision_changed else self.ref_tokens
        if self.model.gpu is not None:
            self.model.gpu.trim()
        res = self.score(toks)
        if self.model.gpu is not None:
            self.model.gpu.trim()
        for name in assignment:
            self.set(name, "orig")
        res["dnll"] = res["nll"] - self.base["nll"]
        return res


def stored_bytes(name: str, scheme: str, original: dict) -> int:
    if scheme == "orig":
        parts = original[name][0]
        return int(sum(np.asarray(p).nbytes for p in parts.values()))
    return Package(QUANT / f"{scheme}.gocr").stored_bytes(name)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sc = sub.add_parser("screen")
    sc.add_argument("--schemes", default="q4,q3")
    sc.add_argument("--groups", default="", help="comma list of group names (default all)")
    sc.add_argument("--sets", default="calib,calib-dc")
    sc.add_argument("--limit", type=int, default=0)
    sc.add_argument("--every", type=int, default=1, help="score every n-th sample (screening)")
    sc.add_argument("--out", default=str(QUANT / "screen.json"))
    cf = sub.add_parser("config", help="score whole configurations: plan, uniform:<scheme>, or a JSON file")
    cf.add_argument("specs", nargs="+", help="plan | uniform:q4 | path/to/assign.json (several in one bench)")
    cf.add_argument("--sets", default="calib,calib-dc")
    cf.add_argument("--every", type=int, default=1)
    cf.add_argument("--out", default=str(QUANT / "configs.json"))
    em = sub.add_parser("embed", help="the input embedding (its rows are gathered per token) in each scheme")
    em.add_argument("--schemes", default="q8,q6,q5,q4")
    em.add_argument("--sets", default="calib,calib-dc")
    em.add_argument("--every", type=int, default=1)
    em.add_argument("--out", default=str(QUANT / "screen.json"))
    args = ap.parse_args()

    bench = Bench(args.sets.split(","), every=getattr(args, "every", 1))
    out_path = Path(args.out)
    results = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    results["_base"] = bench.base
    gs = groups()
    if args.cmd == "embed":
        # rows of the embedding come from a package holding only it, in the scheme under test
        original = bench.model.embed
        for scheme in args.schemes.split(","):
            key = f"lm.embed@{scheme}"
            path = QUANT / f"embed-{scheme}.gocr"
            if not path.exists():
                with PackageWriter(path.with_suffix(".part"), {"scheme": scheme}) as w:
                    w.add(language.EMBED, bench.model.src.get(language.EMBED), scheme)
                path.with_suffix(".part").replace(path)
            pkg = Package(path)
            t0 = time.perf_counter()
            bench.model.embed = lambda ids, pkg=pkg: pkg.rows(language.EMBED, ids)
            res = bench.score(bench.ref_tokens)
            bench.model.embed = original
            res["dnll"] = res["nll"] - bench.base["nll"]
            res["bytes"] = pkg.stored_bytes(language.EMBED)
            res["seconds"] = round(time.perf_counter() - t0, 1)
            results[key] = res
            out_path.write_text(json.dumps(results, indent=1), encoding="utf-8")
            print(f"{key:<24} dnll {res['dnll']:+.5f} flips {res['flips']:.4f} bytes {res['bytes'] / 2**20:7.1f} MB ({res['seconds']}s)", flush=True)
        return
    if args.cmd == "screen":
        names = args.groups.split(",") if args.groups else list(gs)
        for scheme in args.schemes.split(","):
            for g in names:
                key = f"{g}@{scheme}"
                if key in results:
                    continue
                t0 = time.perf_counter()
                res = bench.evaluate({n: scheme for n in gs[g]}, vision_changed=g.startswith("vis"))
                res["bytes"] = sum(stored_bytes(n, scheme, bench.original) for n in gs[g])
                res["bytes_orig"] = sum(stored_bytes(n, "orig", bench.original) for n in gs[g])
                res["seconds"] = round(time.perf_counter() - t0, 1)
                results[key] = res
                out_path.write_text(json.dumps(results, indent=1), encoding="utf-8")
                print(f"{key:<24} dnll {res['dnll']:+.5f} flips {res['flips']:.4f} bytes {res['bytes'] / 2**20:7.1f} MB ({res['seconds']}s)", flush=True)
    else:
        for spec in args.specs:
            if spec == "plan":
                name, assignment = "plan", plan_config()
            elif spec.startswith("uniform:"):
                name, scheme = spec, spec.split(":", 1)[1]
                assignment = {n: scheme for n in sum(gs.values(), [])}
            else:
                name = Path(spec).stem
                assignment = {}
                for key, scheme in json.loads(Path(spec).read_text(encoding="utf-8")).items():
                    for n in gs.get(key, [key]):
                        assignment[n] = scheme
            t0 = time.perf_counter()
            res = bench.evaluate(assignment, vision_changed=any(n.startswith(V) and s != "orig" for n, s in assignment.items()))
            res["bytes"] = sum(stored_bytes(n, s, bench.original) for n, s in assignment.items())
            res["seconds"] = round(time.perf_counter() - t0, 1)
            res["assignment"] = assignment
            results[name] = res
            out_path.write_text(json.dumps(results, indent=1), encoding="utf-8")
            print(f"{name}: dnll {res['dnll']:+.5f} flips {res['flips']:.4f} matrices {res['bytes'] / 2**20:.1f} MB ({res['seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
