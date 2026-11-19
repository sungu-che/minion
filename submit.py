#!/usr/bin/env python3
"""Submit a Track 2 zip from the command line (e.g. from a GPU box with no browser). Standard library only.

  python3 submit.py submission.zip      submit
  python3 submit.py --status            your recent submissions: status, quality, wall time, eligible

The first run asks for your Submit server (제출 서버) username and password and keeps the API token it gets back in
~/.submit-server-token (readable by you only); later runs reuse it. The submission is made as that account, exactly
like uploading on the Submit tab: same owner, same quota (10 per clock hour, 2 waiting or running).

  --relogin       forget the saved token and ask again
  --site URL      competition site (default: below)
  --competition N competition id (default: below)

Large zips (model weights) are fine: the file goes straight to the competition's storage, not through the site's
web proxy, which limits a single request to 100 MB.
"""
import argparse
import datetime
import getpass
import json
import os
import sys
import time
import urllib.error
import urllib.request

SITE = "https://codabench-rw1y11331.gobrev.dev"                 # filled in by build.py from config/track2.yaml (leaderboard.site)
COMPETITION = "1"   # filled in by build.py (leaderboard.competition)
TOKEN_FILE = os.path.expanduser("~/.submit-server-token")
# The site sits behind Cloudflare, which rejects Python's default user agent (error 1010)
USER_AGENT = "track2-submit/1.0 (+starter kit)"


def api(site, method, path, body=None, token=None):
    headers = {"Content-Type": "application/json", "Accept": "application/json", "User-Agent": USER_AGENT}
    if token:
        headers["Authorization"] = f"Token {token}"
    req = urllib.request.Request(f"{site}/api/{path}", method=method, headers=headers,
                                 data=json.dumps(body).encode() if body is not None else None)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            txt = r.read().decode()
            return json.loads(txt) if txt else {}
    except urllib.error.HTTPError as e:
        msg = e.read().decode(errors="replace")[:500]
        if e.code == 401:
            sys.exit("Login rejected or token expired. Run again with --relogin.")
        sys.exit(f"The site refused: {method} {path} -> {e.code}: {msg}")


def get_token(site, relogin):
    if not relogin and os.path.exists(TOKEN_FILE):
        return open(TOKEN_FILE).read().strip()
    user = input("Submit server username: ").strip()
    pw = getpass.getpass("Submit server password: ")
    token = api(site, "POST", "api-token-auth/", {"username": user, "password": pw}).get("token")
    if not token:
        sys.exit("No token returned: check your username and password.")
    fd = os.open(TOKEN_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(token)
    return token


class Progress:
    """A file wrapper that prints upload progress as urllib reads it."""

    def __init__(self, path):
        self.f, self.size, self.done, self.t0, self.last = open(path, "rb"), os.path.getsize(path), 0, time.time(), 0

    def read(self, n=-1):
        chunk = self.f.read(1 << 20 if n is None or n < 0 else n)
        self.done += len(chunk)
        if time.time() - self.last > 1 or self.done == self.size:
            self.last = time.time()
            rate = self.done / max(1e-6, self.last - self.t0) / 1e6
            print(f"\r  uploaded {self.done / 1e6:,.0f} / {self.size / 1e6:,.0f} MB ({rate:.1f} MB/s)", end="", flush=True)
        return chunk


def show_status(site, competition, token, me, n=10):
    """Your last n submissions (the API also returns other teams' for organisers; keep your own)."""
    comp = api(site, "GET", f"competitions/{competition}/", token=token)
    phase = sorted(comp["phases"], key=lambda p: p["index"])[0]
    rows = api(site, "GET", f"submissions/?phase={phase['id']}&page_size=100", token=token)
    rows = [r for r in (rows.get("results", rows) if isinstance(rows, dict) else rows) if r.get("owner") == me]
    kst = datetime.timezone(datetime.timedelta(hours=9))
    print(f"{'id':>5}  {'submitted (KST)':16}  {'status':10}  {'quality':>7}  {'wall s':>7}  eligible")
    for r in sorted(rows, key=lambda r: r["id"], reverse=True)[:n]:
        sc = {x["column_key"]: float(x["score"]) for x in r.get("scores") or []}
        when = datetime.datetime.fromisoformat(r["created_when"].replace("Z", "+00:00")).astimezone(kst)
        q = f"{sc['quality']:.2f}" if "quality" in sc else "-"
        w = f"{sc['wall_time']:.3f}" if "wall_time" in sc else "-"
        e = {1.0: "yes", 0.0: "no"}.get(sc.get("eligible"), "-")
        print(f"{r['id']:>5}  {when:%m-%d %H:%M}       {r['status']:10}  {q:>7}  {w:>7}  {e}")
    if not rows:
        print("  (no submissions yet)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("zip", nargs="?")
    ap.add_argument("--status", action="store_true", help="list your recent submissions and their scores")
    ap.add_argument("--site", default=SITE)
    ap.add_argument("--competition", type=int, default=int(COMPETITION) if COMPETITION.isdigit() else None)
    ap.add_argument("--relogin", action="store_true")
    a = ap.parse_args()
    site = a.site.rstrip("/")
    if a.competition is None:
        sys.exit("--competition is required")
    if not a.status and (not a.zip or not os.path.isfile(a.zip) or not a.zip.endswith(".zip")):
        sys.exit(f"{a.zip or 'missing'}: give a .zip file to submit, or --status")

    token = get_token(site, a.relogin)
    me = api(site, "GET", "my_profile/", token=token)
    if a.status:
        print(f"Account: {me.get('username', '?')}")
        return show_status(site, a.competition, token, me.get("username"))
    print(f"Submitting as: {me.get('username', '?')}")
    comp = api(site, "GET", f"competitions/{a.competition}/", token=token)
    phase = sorted(comp["phases"], key=lambda p: p["index"])[0]
    name, size = os.path.basename(a.zip), os.path.getsize(a.zip)

    ds = api(site, "POST", "datasets/", {"type": "submission", "competition": a.competition,
                                          "request_sassy_file_name": name, "file_name": name, "file_size": size},
             token=token)
    print(f"Uploading {name} ({size / 1e6:,.1f} MB) ...")
    body = Progress(a.zip)
    req = urllib.request.Request(ds["sassy_url"], data=body, method="PUT",
                                 headers={"Content-Type": "application/zip", "Content-Length": str(size),
                                          "User-Agent": USER_AGENT})
    try:
        urllib.request.urlopen(req, timeout=3600).read()
    except urllib.error.HTTPError as e:
        sys.exit(f"\nUpload failed: {e.code} {e.read().decode(errors='replace')[:300]}")
    print()
    api(site, "PUT", f"datasets/completed/{ds['key']}/", token=token)
    sub = api(site, "POST", "submissions/", {"data": ds["key"], "phase": phase["id"],
                                             "tasks": [t["id"] for t in phase["tasks"]]}, token=token)
    print(f"Submitted: submission {sub.get('id')}. Check it with: python3 submit.py --status "
          f"(or the Submit tab: {site}/competitions/{a.competition}/)")


if __name__ == "__main__":
    main()
