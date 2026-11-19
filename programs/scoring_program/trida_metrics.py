"""Per-crop quality from GravityOCR's own scorer: trida-stack benchmark/gravityocr/score.py (OmniDocBench metrics).

  text    : 1 - Edit_dist     OmniDocBench normalized_text, Levenshtein / max len
  table   : TEDS              OmniDocBench normalize_table_match_content (Markdown pipe tables -> HTML first)
  formula : CDM F1            both LaTeX strings rendered with pdflatex, rendered characters matched

Runs in the image's GravityOCR evaluation venv ($GRAVITYOCR_EVAL/venv, built by trida-stack's setup.sh), not the
SGLang venv: scoring.py calls it as a subprocess.

  python trida_metrics.py jobs.jsonl scores.json
    jobs.jsonl : {"key", "id", "task", "target", "pred"} per line (key = which prediction file)
    scores.json: {key: {id: score 0-100}}

pdflatex compiles team-controlled LaTeX (formula predictions) inside the scoring container, next to the answers, and
TeX can read any file it can name (\\input, \\openin; kpathsea's openin_any=p does not stop this in TeX Live 2026).
So the prediction's pdflatex runs as an unprivileged user (uid 65534, like the harness's team user) that can reach
only its own temp directory, under a CPU-time, wall-time and file-size limit: the crop's ground-truth files, the jobs file and the reference are root-only. Before
scoring, main() checks that this user cannot read the protected paths and refuses to score if it can. The ground
truth is organiser LaTeX and compiles as root. Shell escape is off. CDM's own timeouts (30 s per pdflatex call,
tokenizer, render) bound a hostile formula; a crash scores the crop 0.

  python trida_metrics.py jobs.jsonl scores.json [protected path ...]
"""
import glob
import importlib.util
import json
import os
import re
import resource
import shlex
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor

EVAL = os.environ.get("GRAVITYOCR_EVAL", "/opt/gravityocr-eval")
TRIDA = os.environ.get("TRIDA_STACK", "/opt/trida-stack")
SCORE_PY = os.path.join(TRIDA, "benchmark", "gravityocr", "score.py")
ODB = os.path.join(EVAL, "OmniDocBench")
TASK_TYPE = {"text": "text", "table": "table", "formula": "formula"}
SANDBOX_UID = int(os.environ.get("TRACK2_TEAM_UID", "65534"))     # nobody
FONT_CACHE = os.path.join(tempfile.gettempdir(), "track2_texmfvar")   # writable by the sandbox, one per scoring run

_score = None


def _sandbox_env():
    texbin = sorted(glob.glob(os.path.join(EVAL, "texlive", "bin", "*")))
    path = [os.path.join(EVAL, "bin")] + texbin[:1] + [os.environ.get("PATH", "")]
    os.environ.update({
        "PATH": os.pathsep.join(path),
        "CDM_SAVE_VIS": "0", "CDM_CJK_FONT": "mj",          # as trida-stack run.sh scores
        "OMNIDOCBENCH_ROOT": ODB,
        "shell_escape": "f",
    })


def _drop():
    os.setsid()
    os.setgroups([])
    os.setgid(SANDBOX_UID)
    os.setuid(SANDBOX_UID)


def _drop_limited(timeout_sec):
    def f():
        _drop()
        resource.setrlimit(resource.RLIMIT_CPU, (timeout_sec, timeout_sec))
        resource.setrlimit(resource.RLIMIT_FSIZE, (64 << 20, 64 << 20))
    return f


def _sandboxed_run_cmd(l2b):
    """Replaces CDM's run_cmd (latex2bbox_color.py) for the prediction's pdflatex: same command, same timeout and
    kill, but run as SANDBOX_UID with only its own temp directory reachable."""
    original = l2b.run_cmd

    def run_cmd(cmd, timeout_sec=30):
        m = re.search(r"-output-directory=(\S+)", cmd)
        out = shlex.split(m.group(1))[0] if m else ""
        if os.path.basename(out) != "temp_pred":
            return original(cmd, timeout_sec)          # ground truth (organiser LaTeX) or not pdflatex
        root = os.path.dirname(out)                     # CDM's per-crop dir: gt/, temp_gt/, pred/, temp_pred/
        for d in os.listdir(root):
            os.chmod(os.path.join(root, d), 0o700)      # ground-truth files root-only from here on
        os.chmod(root, 0o711)
        os.chmod(out, 0o777)    # root-owned: the scoring container's root has no DAC_OVERRIDE to write a uid-65534 dir
        # The time limit is enforced inside the sandbox: that root also has no CAP_KILL for a uid-65534 process.
        env = l2b.build_tex_env()
        env["TEXMFVAR"] = FONT_CACHE   # fonts mktexpk makes on the fly (e.g. tcrm1200 for \\text{}); the image's is root-only
        proc = subprocess.Popen(f"timeout -s KILL {int(timeout_sec)} sh -c {shlex.quote(cmd)}", shell=True, cwd=out,
                                stdin=subprocess.DEVNULL, preexec_fn=_drop_limited(timeout_sec), env=env)
        try:
            proc.communicate(timeout=timeout_sec + 10)
            return int(proc.returncode or 0)
        except subprocess.TimeoutExpired:
            return -1
    l2b.run_cmd = run_cmd


def _check_protected(paths):
    """Fail closed: the sandbox user must not be able to read or list any protected path."""
    probe = "import os,sys\nfor p in sys.argv[1:]:\n try:\n  os.listdir(p) if os.path.isdir(p) else open(p).read(1)\n" \
            "  print(p)\n except OSError: pass\n"
    r = subprocess.run([sys.executable, "-c", probe, *paths], preexec_fn=_drop, capture_output=True, text=True)
    if r.returncode != 0 or r.stdout.strip():
        sys.exit(f"refusing to score: the LaTeX sandbox user can read {r.stdout.split() or r.stderr[-200:]}")


def _init_worker():
    global _score
    _sandbox_env()
    spec = importlib.util.spec_from_file_location("trida_score", SCORE_PY)
    _score = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_score)
    _score.ODB = ODB
    _score._odb()                                       # imports OmniDocBench (and CDM) from ODB
    from src.metrics.cdm.modules import latex2bbox_color
    _sandboxed_run_cmd(latex2bbox_color)


def _one(job):
    rec = {"id": job["id"], "task_type": TASK_TYPE[job["task"]], "target": job["target"]}
    try:
        _, out = _score.score_one((rec, job["pred"] or "", job["cdm_root"]))
        return job["key"], job["id"], 100.0 * max(0.0, float(out["score"]))
    except Exception as e:  # never let one crop fail the submission
        print(f"crop {job['key']}/{job['id']}: {type(e).__name__}", file=sys.stderr)
        return job["key"], job["id"], 0.0


def main():
    jobs_path, out_path = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
    _check_protected([jobs_path, *sys.argv[3:]])
    jobs = [json.loads(l) for l in open(jobs_path, encoding="utf-8") if l.strip()]
    work = tempfile.mkdtemp(prefix="track2_cdm_")       # CDM's tex/pdf files
    os.chmod(work, 0o711)                               # the sandbox user may pass through, not list
    os.makedirs(FONT_CACHE, exist_ok=True)
    os.chmod(FONT_CACHE, 0o1777)
    os.chdir(work)
    for j in jobs:
        j["cdm_root"] = os.path.join(work, j["key"])
    scores = {}
    workers = max(1, (os.cpu_count() or 2) // 2)
    with ProcessPoolExecutor(workers, initializer=_init_worker) as ex:
        for key, cid, s in ex.map(_one, jobs, chunksize=4):
            scores.setdefault(key, {})[cid] = s
    json.dump(scores, open(out_path, "w"))


if __name__ == "__main__":
    main()
