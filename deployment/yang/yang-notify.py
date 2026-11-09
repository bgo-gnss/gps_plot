#!/mnt/data/gpsops/venv/bin/python
"""yang-notify: one mail when a timer's run starts failing, one when it recovers.

    yang-notify.py fail UNIT   (OnFailure=, via yang-notify@.service)
    yang-notify.py ok UNIT     (ExecStartPost=, after a good run)

Once per failure streak, not every 5 minutes. A mail the relay refuses is
reported on stderr and the streak still counts: the journal and gps-jobs show
the failure either way.
"""

from __future__ import annotations

import os
import smtplib
import subprocess
import sys
from datetime import UTC, datetime
from email.message import EmailMessage
from pathlib import Path

SMTP_HOST = "mail.vedur.is"
SENDER = "gpsops@yang-d01.vedur.is"
MOUNTS = ("/mnt_data/okada/D/SNX", "/mnt_data/okada/WEEKS", "/mnt_data/rtgpsdata")


def state_dir() -> Path:
    base = os.environ.get("XDG_STATE_HOME") or str(Path.home() / ".local/state")
    path = Path(base) / "yang-notify"
    path.mkdir(parents=True, exist_ok=True)
    return path


def unreachable_mounts() -> list[str]:
    """NFS sources a timer depends on that do not answer (stale or down)."""
    bad = []
    for mount in MOUNTS:
        try:
            subprocess.run(
                ["ls", mount], capture_output=True, check=True, timeout=10
            )
        except (subprocess.SubprocessError, OSError):
            bad.append(mount)
    return bad


def send(subject: str, body: str) -> None:
    to = os.environ.get("NOTIFY_TO", "")
    print(f"notify: {subject}: {body}")
    if not to:
        return
    msg = EmailMessage()
    msg["Subject"], msg["From"], msg["To"] = subject, SENDER, to
    msg.set_content(body)
    try:
        with smtplib.SMTP(SMTP_HOST, timeout=20) as smtp:
            smtp.send_message(msg)
    except (OSError, smtplib.SMTPException) as exc:
        print(f"notify: mail not sent: {exc}", file=sys.stderr)


def main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[0] not in ("fail", "ok"):
        print(__doc__, file=sys.stderr)
        return 64
    what, unit = argv
    name = unit.removesuffix(".service")
    flag = state_dir() / f"{name}.failing"
    if what == "fail":
        if flag.exists():
            return 0
        flag.write_text(datetime.now(UTC).strftime("%Y-%m-%dT%H:%MZ"))
        bad = unreachable_mounts()
        body = f"{unit} failed on yang-d01. journalctl _SYSTEMD_USER_UNIT={unit}"
        if bad:
            body += f"\nNot answering: {', '.join(bad)} (stale NFS: sudo umount -l <path>, then sudo mount -a)"
        send(f"GPS yang: {name} stopped", body)
    elif flag.exists():
        since = flag.read_text()
        flag.unlink()
        send(f"GPS yang: {name} back", f"{unit} runs again (failing since {since}).")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
