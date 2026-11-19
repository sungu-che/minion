"""Run a Track 2 submission locally exactly as our eval server does: same image, same harness and scorer, the
same container restrictions (network off, worker capabilities minus SETUID/SETGID, no-new-privileges), the GPU.

Usage (on a Linux GPU box with Docker and the image):
  teams:      python3 local_run.py --team <dir with team.yaml> --dev-set <unzipped dev-set.zip>
  organisers: python3 local_run.py --team <dir> --data <dir with input_data/{test,dev} and reference_data/>
  options:    [--image track2-base:v1] [--out <dir>]
With --dev-set, dev stands in for the private test set (3 timed runs) and is also the untimed dev run.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
CAP_DROP = ["AUDIT_WRITE", "CHOWN", "DAC_OVERRIDE", "FOWNER", "FSETID", "KILL", "MKNOD", "NET_BIND_SERVICE",
            "NET_RAW", "SETFCAP", "SETPCAP", "SYS_CHROOT"]   # the eval server's list (SETUID/SETGID kept)


def sh(cmd, **kw):
    print("+", " ".join(cmd[:6]), "...", flush=True)
    return subprocess.run(cmd, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", help="input_data/{test,dev} + reference_data/ (organisers)")
    src.add_argument("--dev-set", help="the unzipped dev-set.zip: manifest.jsonl, targets.jsonl, images/ (teams)")
    ap.add_argument("--image", default="track2-base:v1")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if subprocess.run(["docker", "image", "inspect", a.image], capture_output=True).returncode != 0:
        base = os.path.join(HERE, "track2-base")    # starter kit; in the repo it is images/track2-base
        if not os.path.isdir(base):
            base = os.path.normpath(os.path.join(HERE, "..", "..", "images", "track2-base"))
        sys.exit(f"Docker image {a.image} not found. Build it (~5 min, downloads ~20 GB):\n"
                 f"  docker build -t {a.image} {base}")
    if a.dev_set:   # accept the unzipped download, or a folder with dev/ inside (older bundles)
        if not os.path.exists(os.path.join(a.dev_set, "manifest.jsonl")) and \
                os.path.exists(os.path.join(a.dev_set, "dev", "manifest.jsonl")):
            a.dev_set = os.path.join(a.dev_set, "dev")
        if not os.path.exists(os.path.join(a.dev_set, "manifest.jsonl")):
            sys.exit(f"--dev-set {a.dev_set}: no manifest.jsonl here or in dev/ (unzip the dev set download first)")

    # absolute: docker -v reads a relative path ("out") as a volume name and refuses it
    work = os.path.abspath(a.out) if a.out else tempfile.mkdtemp(prefix="track2-local-")
    for d in ("program", "scoring_program", "team", "output", "score", "input_data", "reference_data"):
        shutil.rmtree(os.path.join(work, d), ignore_errors=True)
    progs = os.path.join(HERE, "programs") if os.path.isdir(os.path.join(HERE, "programs")) else os.path.join(HERE, "bundle")
    shutil.copytree(os.path.join(progs, "ingestion_program"), os.path.join(work, "program"))
    shutil.copytree(os.path.join(progs, "scoring_program"), os.path.join(work, "scoring_program"))
    # like `zip -r`: symlinks are copied as their targets, broken ones skipped (SGLang's source has one)
    shutil.copytree(a.team, os.path.join(work, "team"), ignore_dangling_symlinks=True)
    os.makedirs(os.path.join(work, "output"))
    os.makedirs(os.path.join(work, "score"))
    # Like the worker: a fresh copy of the data for this run (root-owned below), never the original folder
    if a.data:
        shutil.copytree(os.path.join(a.data, "input_data"), os.path.join(work, "input_data"))
        shutil.copytree(os.path.join(a.data, "reference_data"), os.path.join(work, "reference_data"))
    else:   # the dev download stands in for both splits
        for split in ("test", "dev"):
            d = os.path.join(work, "input_data", split)
            shutil.copytree(os.path.join(a.dev_set, "images"), os.path.join(d, "images"))
            shutil.copy(os.path.join(a.dev_set, "manifest.jsonl"), os.path.join(d, "manifest.jsonl"))
        os.makedirs(os.path.join(work, "reference_data"))
        with open(os.path.join(work, "reference_data", "reference.jsonl"), "w") as out:
            targets = [json.loads(l) for l in open(os.path.join(a.dev_set, "targets.jsonl")) if l.strip()]
            for split in ("test", "dev"):
                for t in targets:
                    out.write(json.dumps({**t, "split": split}, ensure_ascii=False) + "\n")
    data = work
    gpu = ["--device", "nvidia.com/gpu=all"]
    # The eval server runs as root and creates run folders as root: copy that ownership
    sh(["docker", "run", "--rm", "-v", f"{work}:/w", a.image, "chown", "-R", "0:0", "/w"], check=True,
       capture_output=True)
    ing = sh(["docker", "run", "--rm", "--network", "none", "--security-opt", "no-new-privileges", "--shm-size=16g", *gpu,   # /dev/shm as on our eval server
              *[x for c in CAP_DROP for x in ("--cap-drop", c)],
              "-v", f"{work}/program:/app/program", "-v", f"{work}/team:/app/ingested_program:ro",
              "-v", f"{data}/input_data:/app/input_data", "-v", f"{work}/output:/app/output",
              "-w", "/app/program", a.image, "python3", "ingestion.py"])
    if ing.returncode != 0:
        sys.exit(f"harness failed (exit {ing.returncode}); logs in {work}/output")
    sc = sh(["docker", "run", "--rm", "--network", "none", "-v", f"{work}/scoring_program:/app/program",
             "-v", f"{data}/reference_data:/app/input/ref:ro", "-v", f"{work}/output:/app/input/res:ro",
             "-v", f"{work}/score:/app/output", "-w", "/app/program", a.image, "python3", "scoring.py"])
    # Hand the work folder back to whoever ran this (the harness and scorer wrote it as root), so the scores and
    # the team logs in output/ can be read
    sh(["docker", "run", "--rm", "-v", f"{work}:/w", "--entrypoint", "sh", a.image, "-c",
        f"chown -R {os.getuid()}:{os.getgid()} /w && chmod -R u+rwX /w"], check=True, capture_output=True)
    if sc.returncode != 0:
        sys.exit(f"scorer failed (exit {sc.returncode}); logs in {work}/score")
    print(json.dumps(json.load(open(os.path.join(work, "score", "scores.json"))), indent=2))
    print(f"outputs: {work}")


if __name__ == "__main__":
    main()
