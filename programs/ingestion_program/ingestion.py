"""Track 2 harness (organiser code). Runs as root inside the submission container.

Team contract (team.yaml in the submission zip):
    serve: <command>      long-running; started as an unprivileged user with $TRACK2_DIR set.

Protocol, per timed run (a fresh team process each run when restart_between_runs is true):
  1. The team process loads its model and warms up (it may read $TRACK2_DEV_DIR), then creates
     $TRACK2_DIR/READY. Untimed, up to ready_timeout_seconds.
  2. The harness moves a job folder into $TRACK2_DIR/jobs/<n>-<split>/ (manifest.jsonl + images/).
     The clock starts at that move.
  3. The team writes jobs/<n>-<split>/predictions.jsonl ({"id", "text"} per line), then creates DONE in that
     folder. The clock stops at DONE's modification time.
  4. After the last job of a process the harness creates $TRACK2_DIR/STOP; the team process should exit.

Test (private) is timed test_runs times; the ranked wall time is the median. Dev runs once, untimed, in the last
process. Test images are copied to a root-only folder before any team code starts, so the team process cannot
read them until handover. If that protection cannot be set up, the harness stops instead of running unprotected.
"""
import json
import os
import shutil
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
INPUT = os.environ.get("TRACK2_INPUT", "/app/input_data")
SUB = os.environ.get("TRACK2_SUBMISSION", "/app/ingested_program")
OUT = os.environ.get("TRACK2_OUTPUT", "/app/output")
WORK = os.environ.get("TRACK2_WORK", "/tmp/track2")
STAGE = os.environ.get("TRACK2_STAGE", "/var/lib/track2-stage")
TEAM_UID = int(os.environ.get("TRACK2_TEAM_UID", "65534"))     # nobody
CFG = json.load(open(os.path.join(HERE, "timing.json")))

# Pinned toolchain: python here; in track2-base also the full manifest (check_versions below).
PINNED = {"python": "3.12"}


def log(msg):
    print(f"[harness] {msg}", flush=True)


def fail(msg):
    log(f"STOPPED: {msg}")
    sys.exit(1)


def drop_privileges():
    os.setgroups([])
    os.setgid(TEAM_UID)
    os.setuid(TEAM_UID)


def as_team(cmd, **kw):
    try:
        return subprocess.run(cmd, preexec_fn=drop_privileges, **kw)
    except PermissionError as e:   # e.g. the image's Python lives under /root, which the team user cannot enter
        fail(f"the team user (uid {TEAM_UID}) cannot run {cmd[0]}: {e}. Fix the image, not the submission.")


def read_team_yaml():
    cfg = {}
    for line in open(os.path.join(SUB, "team.yaml")):
        if ":" in line and not line.strip().startswith("#"):
            k, v = line.split(":", 1)
            cfg[k.strip()] = v.strip()
    if not cfg.get("serve"):
        fail("team.yaml has no 'serve:' command")
    return cfg


PINNED_MANIFEST = "/etc/track2/pinned.json"          # written by track2-base's check_versions.py at build time
PINNED_CHECKER = "/opt/track2/check_versions.py"


def check_versions():
    got = {"python": f"{sys.version_info.major}.{sys.version_info.minor}"}
    bad = {k: (v, got.get(k)) for k, v in PINNED.items() if got.get(k) != v}
    if bad:
        fail(f"pinned versions changed: {bad}")
    if os.path.exists(PINNED_MANIFEST):              # in track2-base: torch, triton, flashinfer, sglang, ...
        r = subprocess.run([sys.executable, PINNED_CHECKER, "--check", PINNED_MANIFEST], capture_output=True, text=True)
        if r.returncode != 0:
            fail(f"pinned toolchain changed: {r.stdout.strip()}")
        log("pinned toolchain unchanged (track2-base)")
    log(f"version check OK: {got}")


def protect_test_set():
    """Copy test into a root-only folder and lock the original; then prove the team user cannot read either."""
    src = os.path.join(INPUT, "test")
    shutil.rmtree(STAGE, ignore_errors=True)
    os.makedirs(os.path.dirname(STAGE), exist_ok=True)
    shutil.copytree(src, os.path.join(STAGE, "test"))
    os.chmod(STAGE, 0o700)
    try:
        os.chmod(src, 0o700)
    except PermissionError:
        fail(f"cannot lock {src} (not owned by root in this container)")
    probe = "import os,sys\nfor p in sys.argv[1:]:\n    try:\n        os.listdir(p); print(p)\n    except PermissionError:\n        pass"
    r = as_team([sys.executable, "-c", probe, src, os.path.join(STAGE, "test")], capture_output=True, text=True)
    if r.stdout.strip():
        fail(f"team user can still read: {r.stdout.split()}")
    log("test set locked: the team process cannot read it before handover")


SHARED_TMP = []   # every folder the team user can write to, found by find_writable() before any team code runs

# Read-only for the team: the code it was given and the harness's own folders are checked separately.
_SCAN = r"""import os, sys
skip = ("/proc", "/sys", "/dev")
for root in ("/", "/dev/shm"):
    for r, ds, fs in os.walk(root, followlinks=False):
        if root == "/" and r.startswith(skip):
            ds[:] = []; continue
        if os.access(r, os.W_OK | os.X_OK): print("d", r)
        for f in fs:
            p = os.path.join(r, f)
            if not os.path.islink(p) and os.path.isfile(p) and os.access(p, os.W_OK): print("f", p)
"""


def find_writable():
    """Scan the whole filesystem as the team user. Writable folders are emptied before every run; writable files
    are made read-only (root owns them) or the harness stops, since data written into them could outlive a run."""
    r = as_team([sys.executable, "-c", _SCAN], capture_output=True, text=True)
    dirs, files = [], []
    for line in r.stdout.splitlines():
        kind, path = line.split(" ", 1)
        (dirs if kind == "d" else files).append(path)
    for f in files:
        try:
            os.chmod(f, os.stat(f).st_mode & ~0o022)
        except OSError as e:
            fail(f"team user can write {f} and it cannot be made read-only ({e})")
    left = [f for f in files if os.access(f, os.W_OK) and os.stat(f).st_mode & 0o002]
    if left:
        fail(f"world-writable files remain: {left}")
    SHARED_TMP[:] = sorted(d for d in dirs if not d.startswith(WORK))
    log(f"team-writable folders (emptied before every run): {SHARED_TMP}; files made read-only: {len(files)}")


def fresh_workdir():
    """Empty every folder the team user can write to before each team process starts. Two passes: the team user
    removes its own files (root here lacks DAC_OVERRIDE), then root removes the rest (e.g. root-owned files a team
    could have written into). The harness's WORK is recreated."""
    if SHARED_TMP:
        dirs = " ".join(f"'{d}'" for d in SHARED_TMP)
        as_team(["sh", "-c", f"find {dirs} -user $(id -u) -mindepth 1 -delete 2>/dev/null; true"])
    for d in SHARED_TMP:
        if not os.path.isdir(d):
            continue
        for name in os.listdir(d):
            path = os.path.join(d, name)
            if path == WORK:
                continue
            try:
                shutil.rmtree(path) if os.path.isdir(path) and not os.path.islink(path) else os.remove(path)
            except OSError:
                pass
    shutil.rmtree(WORK, ignore_errors=True)
    os.makedirs(os.path.join(WORK, "jobs"))
    os.makedirs(os.path.join(WORK, "home"))
    for d in (WORK, os.path.join(WORK, "jobs"), os.path.join(WORK, "home")):
        os.chmod(d, 0o777)


# The image's own settings (CUDA_HOME, GRAVITYOCR_WEIGHTS, SGLANG_SRC, HF_HUB_OFFLINE, ...) reach the team process;
# nothing else from the harness does. The container holds no secrets, but names are listed to keep it explicit.
PASS_ENV = ("PATH", "CUDA_HOME", "CUDA_VERSION", "NVIDIA_VISIBLE_DEVICES", "NVIDIA_DRIVER_CAPABILITIES",
            "LD_LIBRARY_PATH", "SGLANG_SRC", "TRIDA_STACK", "GRAVITYOCR_WEIGHTS", "HF_HUB_OFFLINE",
            "VLLM_USE_FLASHINFER_SAMPLER", "UV_PYTHON_INSTALL_DIR", "LANG", "LC_ALL")


def start_team(cfg, run):
    env = {k: os.environ[k] for k in PASS_ENV if k in os.environ}
    env.update({"PYTHONUNBUFFERED": "1", "TRACK2_DIR": WORK, "TRACK2_DEV_DIR": os.path.join(INPUT, "dev"),
                "HOME": os.path.join(WORK, "home"), "TMPDIR": os.path.join(WORK, "home")})
    logf = open(os.path.join(OUT, f"team_process_{run}.log"), "w")
    sup = os.path.join(HERE, "supervisor.py")
    proc = subprocess.Popen([sys.executable, sup, WORK, cfg["serve"]], cwd=SUB, env=env, stdout=logf,
                            stderr=subprocess.STDOUT, preexec_fn=drop_privileges)
    return proc, logf


def wait_for(path, timeout, proc, what):
    deadline = time.monotonic() + timeout
    while not os.path.exists(path):
        if proc.poll() is not None:
            fail(f"team process exited (code {proc.returncode}) before {what}; see team_process_*.log")
        if time.monotonic() > deadline:
            fail(f"timed out after {timeout}s waiting for {what}")
        time.sleep(0.002)


def hand_over(split, n, src_dir):
    """Prepare the job out of sight, then move it into place. Returns (job_dir, t0_ns)."""
    hidden = os.path.join(WORK, f".staging-{n}")
    shutil.copytree(src_dir, hidden)
    for root, dirs, files in os.walk(hidden):
        for d in dirs:
            os.chmod(os.path.join(root, d), 0o755)
        for f in files:
            os.chmod(os.path.join(root, f), 0o644)
    os.chmod(hidden, 0o700)                        # invisible to the team until the move
    job = os.path.join(WORK, "jobs", f"{n}-{split}")
    t0 = time.time_ns()
    os.chmod(hidden, 0o777)
    os.rename(hidden, job)
    return job, t0


def run_job(proc, split, n, src_dir, timed):
    job, t0 = hand_over(split, n, src_dir)
    done = os.path.join(job, "DONE")
    wait_for(done, CFG["run_timeout_seconds"] if timed else CFG["ready_timeout_seconds"], proc, f"{split} DONE")
    t1 = os.stat(done).st_mtime_ns
    wall = max(t1 - t0, 0) / 1e9
    preds = os.path.join(job, "predictions.jsonl")
    if not os.path.exists(preds):
        fail(f"{split} job finished without predictions.jsonl")
    return preds, wall


def reap():
    """The harness is PID 1 in the container, so orphaned team processes become its children: collect the exited
    ones (zombies) so they disappear from /proc."""
    while True:
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return
        if pid == 0:
            return


def team_pids():
    """Live (non-zombie) processes owned by the team user."""
    reap()
    pids = []
    for p in os.listdir("/proc"):
        if p.isdigit():
            try:
                if os.stat(f"/proc/{p}").st_uid != TEAM_UID:
                    continue
                state = open(f"/proc/{p}/stat").read().rsplit(")", 1)[1].split()[0]
                if state != "Z":
                    pids.append(int(p))
            except OSError:      # exited meanwhile
                pass
    return pids


def stop_team(proc, logf):
    """Stop the team's code completely: every process of the team user, whatever group or session it moved to
    (a server started with setsid would otherwise survive the restart and could carry answers between runs)."""
    open(os.path.join(WORK, "STOP"), "w").close()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        log("team process did not stop within 30 s")
    deadline = time.monotonic() + 30
    while team_pids():
        as_team(["sh", "-c", "kill -KILL -1 2>/dev/null; true"])   # as the team user: only its own processes
        if time.monotonic() > deadline:
            fail(f"team processes still running after stop: {team_pids()}")
        time.sleep(0.2)
    try:
        proc.wait(timeout=5)
    except (subprocess.TimeoutExpired, ChildProcessError):
        pass
    logf.close()


def main():
    cfg = read_team_yaml()
    check_versions()
    protect_test_set()
    find_writable()
    runs, restart = int(CFG["test_runs"]), bool(CFG["restart_between_runs"])
    walls, proc, logf, n = [], None, None, 0
    for run in range(1, runs + 1):
        if proc is None or restart:
            if proc is not None:
                stop_team(proc, logf)
            fresh_workdir()
            log(f"run {run}: starting team process (untimed load)")
            proc, logf = start_team(cfg, run)
            wait_for(os.path.join(WORK, "READY"), CFG["ready_timeout_seconds"], proc, "READY")
        n += 1
        preds, wall = run_job(proc, "test", n, os.path.join(STAGE, "test"), timed=True)
        shutil.copy(preds, os.path.join(OUT, f"predictions_test_{run}.jsonl"))
        walls.append(wall)
        log(f"run {run}: test wall time {wall:.3f} s")
    # Dev runs in a fresh process that never saw test data. Teams see none of this output (phase hide_output):
    # only their scores reach them, on the leaderboard. Logs stay with the organisers.
    stop_team(proc, logf)
    fresh_workdir()
    log("dev: starting a fresh team process (untimed)")
    proc, logf = start_team(cfg, "dev")
    wait_for(os.path.join(WORK, "READY"), CFG["ready_timeout_seconds"], proc, "READY (dev)")
    n += 1
    preds, dev_wall = run_job(proc, "dev", n, os.path.join(INPUT, "dev"), timed=False)
    shutil.copy(preds, os.path.join(OUT, "predictions_dev.jsonl"))
    log(f"dev (untimed): {dev_wall:.3f} s")
    stop_team(proc, logf)
    wall = statistics.median(walls)
    json.dump({"wall_time": wall, "wall_times": walls, "test_runs": runs, "dev_wall_time": dev_wall},
              open(os.path.join(OUT, "metadata.json"), "w"))
    log(f"ranked wall time (median of {runs}): {wall:.3f} s  runs={['%.3f' % w for w in walls]}")


if __name__ == "__main__":
    main()
