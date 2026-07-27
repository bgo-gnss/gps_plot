#!/usr/bin/env python3
"""Probe which Kitty graphics transfer media the current terminal supports.

Ranger's KittyImageDisplayer decides how to send images from ONE query at
startup (`img_display.py::_late_init`): it asks with `t=f` and accepts only two
answers — `OK` means "shared filesystem, use temp files" (`t=t` at draw time),
`EBADF` means "cannot read my file, send inline" (`t=d`). Anything else and it
raises ImgDisplayUnsupportedException and previews are silently off.

herdr answers `EINVAL: unsupported medium` — it renders on the attached CLIENT,
so a server-side file path is meaningless to it. That is neither OK nor EBADF,
so ranger gives up even though inline transfer would work fine.

This probes each medium separately under a timeout, so nothing can hang the way
ranger's unguarded `stdin.read(1)` loop would.
"""

import base64
import os
import select
import sys
import tempfile
import termios
import tty

TIMEOUT = 3.0
START, END = b"\x1b_G", b"\x1b\\"

# 1x1 RGB pixel, the smallest payload every medium can carry.
PIXEL = bytes([0xFF, 0x00, 0x00])


def ask(params, payload):
    """Send one query APC and read the reply, or None on timeout."""
    out = getattr(sys.stdout, "buffer", sys.stdout)
    out.write(START + params.encode("ascii") + b";" + payload + END)
    out.flush()

    resp = b""
    while not resp.endswith(END):
        if not select.select([sys.stdin], [], [], TIMEOUT)[0]:
            return None
        chunk = os.read(sys.stdin.fileno(), 1)
        if not chunk:
            return None
        resp += chunk
    return resp


def verdict(resp):
    if resp is None:
        return "NO RESPONSE (would hang ranger)"
    body = resp.replace(START, b"").replace(END, b"")
    if b"OK" in body:
        return "OK — supported"
    return body.decode("ascii", "replace").split(";", 1)[-1] or "empty reply"


def main():
    fd = sys.stdin.fileno()
    try:
        old = termios.tcgetattr(fd)
    except termios.error:
        print("RESULT: stdin is not a tty — run this from a real pane")
        return 2

    results = {}
    path = None
    tty.setcbreak(fd)
    try:
        # t=d: raw pixel data inline, base64. What ranger uses when stream=True.
        results["d (inline data)"] = ask(
            "a=q,i=1,f=24,s=1,v=1,t=d", base64.standard_b64encode(PIXEL)
        )

        # t=f / t=t: a path the terminal must open itself. What ranger uses when
        # stream=False (t=t at draw time; the startup query asks with t=f).
        with tempfile.NamedTemporaryFile(suffix=".rgb", delete=False) as tmpf:
            tmpf.write(PIXEL)
            path = tmpf.name
        enc = base64.standard_b64encode(path.encode("utf-8"))
        results["f (regular file)"] = ask("a=q,i=2,f=24,s=1,v=1,S=3,t=f", enc)
        results["t (temp file)"] = ask("a=q,i=3,f=24,s=1,v=1,S=3,t=t", enc)

        # The decisive one. ranger's draw() blocks reading a reply after EVERY
        # image it sends (`a=T`, not `a=q`), so a terminal that answers queries
        # but stays silent on display commands still hangs it on the first
        # preview. This paints one red pixel, then deletes it.
        results["T (real display)"] = ask(
            "a=T,i=4,f=24,s=1,v=1,t=d", base64.standard_b64encode(PIXEL)
        )
        sys.stdout.buffer.write(START + b"a=d,i=4" + END)
        sys.stdout.buffer.flush()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        # t=t asks the terminal to delete the file; it may already be gone.
        if path and os.path.exists(path):
            os.unlink(path)

    print()
    print("TERM      =", os.environ.get("TERM"))
    print("HERDR_ENV =", os.environ.get("HERDR_ENV"))

    # Ranger sizes every image from TIOCGWINSZ pixel fields. Multiplexers often
    # report 0 there, which collapses the scale factor and yields a blank or
    # zero-sized image — so this is as decisive as the medium support above.
    try:
        import fcntl
        import struct

        packed = fcntl.ioctl(
            sys.stdout, termios.TIOCGWINSZ, struct.pack("HHHH", 0, 0, 0, 0)
        )
        rows, cols, xpix, ypix = struct.unpack("HHHH", packed)
        print("winsize   = %d rows x %d cols, %dx%d px" % (rows, cols, xpix, ypix))
        if xpix == 0 or ypix == 0:
            print(
                "            ^ NO pixel size reported — the plugin must supply a "
                "cell size (RANGER_KITTY_CELL_PX)"
            )
        else:
            print("            cell = %dx%d px" % (xpix // cols, ypix // rows))
    except (OSError, ValueError) as exc:
        print("winsize   = unavailable (%s)" % exc)
    print()
    for medium, resp in results.items():
        print("  %-20s %s" % (medium, verdict(resp)))
    print()

    def ok(key):
        return results[key] is not None and b"OK" in results[key]

    # A silent display command is fatal regardless of which media are supported:
    # ranger's draw() has no timeout around its reply read.
    if results["T (real display)"] is None:
        print("RESULT: display commands go unanswered — ranger WOULD HANG on the")
        print("        first preview. Do not set RANGER_KITTY_FORCE_STREAM here.")
    elif ok("f (regular file)"):
        print("RESULT: ranger works unmodified (stream=False, temp files).")
    elif ok("d (inline data)"):
        print("RESULT: inline transfer works, but ranger will NOT find it on its own —")
        print("        its startup query uses t=f and only understands OK/EBADF.")
        print("        Install the kitty-stream ranger plugin (see README).")
    else:
        print("RESULT: no usable medium — previews cannot work here.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
