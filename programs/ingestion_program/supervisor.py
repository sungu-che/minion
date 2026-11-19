"""Runs as the team user. Starts the team's serve command in its own process group and kills that group once
the harness creates $TRACK2_DIR/STOP (root cannot signal another user's processes: KILL is dropped).

Usage: supervisor.py <track2_dir> <serve command>
"""
import os
import signal
import subprocess
import sys
import time

work, cmd = sys.argv[1], sys.argv[2]
team = subprocess.Popen(cmd, shell=True, start_new_session=True)
stop = os.path.join(work, "STOP")
while team.poll() is None and not os.path.exists(stop):
    time.sleep(0.05)
if team.poll() is None:
    time.sleep(2)                                   # give the team a moment to exit on its own
    try:
        os.killpg(team.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
team.wait()
sys.exit(0)
