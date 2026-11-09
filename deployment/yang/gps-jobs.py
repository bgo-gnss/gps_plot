#!/mnt/data/gpsops/venv/bin/python
"""gps-jobs: one line per automated run, from jobs.yaml + systemd + output ages.

    gps-jobs                 every job, problems first
    gps-jobs --group gps     one group (gps, epos, personal)
    gps-jobs --problems      only what needs a look
    gps-jobs --all           also the ignored (·) system timers
    gps-jobs -v              also the last error line of each problem

Status: ✓ ok, ✗ failed (the last run's result), ⌛ stale (an output older than its
max_age, or a service down), ⏸ failed and waiting to retry, ▶ running, ? a --user timer missing from
jobs.yaml, · ignored. Exit 1 when anything is ✗, ⌛ or ?.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
JOBS = HERE / "jobs.yaml"
UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400}
ORDER = {"✗": 0, "⌛": 1, "?": 2, "⏸": 3, "▶": 4, "✓": 5, "·": 6}


def seconds(text: str) -> int:
    """'20m' → 1200 (s, m, h, d)."""
    m = re.fullmatch(r"(\d+)([smhd])", str(text).strip())
    if not m:
        raise ValueError(f"bad max_age {text!r} (use e.g. 20m, 26h, 32d)")
    return int(m[1]) * UNITS[m[2]]


def ago(sec: float | None) -> str:
    """A short age: 45s, 12m, 3h05, 2d04."""
    if sec is None:
        return "-"
    sec = max(0, int(sec))
    if sec < 60:
        return f"{sec}s"
    if sec < 3600:
        return f"{sec // 60}m"
    if sec < 86400:
        return f"{sec // 3600}h{sec % 3600 // 60:02d}"
    return f"{sec // 86400}d{sec % 86400 // 3600:02d}"


def show(unit: str, props: list[str]) -> dict[str, str]:
    """systemctl --user show UNIT -p … as a dict."""
    out = subprocess.run(
        ["systemctl", "--user", "show", unit, *(f"-p{p}" for p in props)],
        capture_output=True,
        text=True,
    ).stdout
    return dict(ln.split("=", 1) for ln in out.splitlines() if "=" in ln)


def user_timers() -> dict[str, tuple[float | None, float | None]]:
    """Every --user timer → (last trigger, next run) in epoch s (None = never)."""
    out = subprocess.run(
        ["systemctl", "--user", "list-timers", "--all", "-o", "json"],
        capture_output=True,
        text=True,
    ).stdout

    def epoch(us: int | None) -> float | None:
        return us / 1e6 if us else None

    return {
        t["unit"].removesuffix(".timer"): (epoch(t.get("last")), epoch(t.get("next")))
        for t in json.loads(out or "[]")
    }


def last_error(unit: str) -> str:
    out = subprocess.run(
        [
            "journalctl",  # the system journal: yang files gpsops' units there
            f"_SYSTEMD_USER_UNIT={unit}.service",
            "-n",
            "60",
            "--no-pager",
            "-o",
            "cat",
        ],
        capture_output=True,
        text=True,
    ).stdout
    lines = [
        ln.strip()
        for ln in out.splitlines()
        if re.search(r"error|fatal|failed|stale|gap|timeout", ln, re.I)
        and not re.search(r"Failed (to start|with result)|Main process exited", ln)
    ]
    return lines[-1][:160] if lines else ""


@dataclass
class Row:
    name: str
    group: str
    what: str
    status: str = "✓"
    last: float | None = None  # s since the last run ended
    result: str = ""
    next_in: float | None = None
    output: float | None = None  # s, the oldest checked output's age
    notes: list[str] = field(default_factory=list)


def output_age(spec: dict) -> tuple[float | None, str]:
    """Newest mtime under the glob → (age s, note when too old or missing)."""
    files = glob.glob(os.path.expanduser(spec["path"]))
    if not files:
        return None, f"no file {spec['path']}"
    age = time.time() - max(os.path.getmtime(f) for f in files)
    limit = seconds(spec["max_age"])
    note = (
        f"{Path(spec['path']).name} {ago(age)} old > {spec['max_age']}"
        if age > limit
        else ""
    )
    return age, note


def url_ok(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def check(name: str, job: dict, timers: dict) -> Row:
    row = Row(name, job.get("group", "?"), job.get("what", ""))
    if job.get("ignore"):
        row.status = "·"
        return row
    kind = job.get("kind", "timer")
    if kind in ("timer", "service"):
        s = show(f"{name}.service", ["ActiveState", "SubState", "Result", "LoadState"])
        if s.get("LoadState") == "not-found":
            row.status, row.result = "?", "no unit"
            return row
        row.result = s.get("Result", "")
        if kind == "timer":
            last, nxt = timers.get(name, (None, None))
            now = time.time()
            row.last = now - last if last else None
            row.next_in = nxt - now if nxt else None
            if s.get("SubState") == "auto-restart":
                row.status = "⏸"
                row.notes.append("failed, waiting to retry (offline?)")
            elif s.get("ActiveState") == "activating":
                row.status = "▶"
            elif row.result not in ("success", ""):
                row.status = "✗"
        else:
            if s.get("ActiveState") != "active":
                row.status = "⌛"
                row.notes.append(f"service {s.get('ActiveState')}")
            row.result = s.get("ActiveState", "")
    for spec in job.get("outputs", ()):
        age, note = output_age(spec)
        row.output = age if row.output is None else max(row.output, age or 0)
        if note:
            if row.status == "▶":
                note += " (running now)"
            row.notes.append(note)
            if row.status in ("✓", "▶"):
                row.status = "⌛"
    if "url" in job and not url_ok(job["url"]):
        row.notes.append(f"{job['url']} not answering")
        if row.status == "✓":
            row.status = "⌛"
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--group", help="gps, epos or personal")
    ap.add_argument("--problems", action="store_true", help="only ✗ ⌛ ?")
    ap.add_argument("--all", action="store_true", help="also the ignored (·) units")
    ap.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="the last error line of each failed job",
    )
    ap.add_argument("--jobs", default=str(JOBS), help="jobs.yaml")
    a = ap.parse_args()

    jobs = (yaml.safe_load(Path(a.jobs).read_text()) or {}).get("jobs", {})
    timers = user_timers()
    rows = [check(n, j or {}, timers) for n, j in jobs.items()]
    for name in timers:
        if name not in jobs:
            r = Row(name, "?", "not in jobs.yaml", status="?")
            rows.append(r)
    if a.group:
        rows = [r for r in rows if r.group == a.group]
    if a.problems:
        rows = [r for r in rows if r.status in ("✗", "⌛", "?")]
    elif not a.all:
        rows = [r for r in rows if r.status != "·"]
    rows.sort(key=lambda r: (ORDER[r.status], r.group, r.name))

    w = max((len(r.name) for r in rows), default=4)
    print(
        f"  {'job':<{w}}  {'group':<8} {'ran':>6} {'result':<10} {'next':>6} {'output':>6}  what / note"
    )
    for r in rows:
        nxt = ago(r.next_in) if r.next_in is not None else "-"
        note = "; ".join(r.notes) or r.what
        print(
            f"{r.status} {r.name:<{w}}  {r.group:<8} {ago(r.last):>6} "
            f"{r.result[:10]:<10} {nxt:>6} {ago(r.output) if r.output is not None else '-':>6}  {note}"
        )
        if a.verbose and r.status == "✗":
            err = last_error(r.name)
            if err:
                print(f"  {'':<{w}}  └ {err}")
    return 1 if any(r.status in ("✗", "⌛", "?") for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
