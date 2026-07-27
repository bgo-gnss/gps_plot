#!/usr/bin/env python3
"""Display one image three ways and report which the terminal actually renders.

Diagnostic for blank previews. Ranger's inline mode (`stream=True`) sends RAW
uncompressed RGB — a 1000x1000 preview is ~3 MB, ~4 MB once base64'd — and
herdr logs `dropping oversized graphics payload for client frame`, so an image
can be accepted and then silently dropped on the way to the client. The Kitty
protocol also accepts PNG bytes inline (`f=100,t=d`), which is ~50x smaller for
a plot; ranger has no code path for that, but a plugin can add one.

Run it in the pane where previews are broken and say which blocks appear:

    python3 kitty-transfer-test.py ../../tmp-figdir/SKHA-plate-year.png

Each attempt prints its payload size and the terminal's reply, then draws.
"""

import base64
import io
import os
import select
import sys
import termios
import tty

TIMEOUT = 4.0
START, END = b"\x1b_G", b"\x1b\\"
CHUNK = 2048

# Keep the test image small enough that size alone is never the problem for the
# PNG case, while still being large enough to be visibly a plot.
TARGET_PX = (600, 400)


def send(cmds, payload):
    """Chunked APC transmission, mirroring ranger's _format_cmd_str."""
    out = getattr(sys.stdout, "buffer", sys.stdout)
    central = ",".join("%s=%s" % kv for kv in cmds.items()).encode("ascii")
    if payload is None:
        out.write(START + central + b";" + END)
    else:
        rest = payload
        while len(rest) > CHUNK:
            blk, rest = rest[:CHUNK], rest[CHUNK:]
            out.write(START + central + b",m=1;" + blk + END)
        out.write(START + central + b",m=0;" + rest + END)
    out.flush()


def reply():
    resp = b""
    while not resp.endswith(END):
        if not select.select([sys.stdin], [], [], TIMEOUT)[0]:
            return None
        chunk = os.read(sys.stdin.fileno(), 1)
        if not chunk:
            return None
        resp += chunk
    return resp


def describe(resp):
    if resp is None:
        return "NO REPLY (ranger would hang here)"
    body = resp.replace(START, b"").replace(END, b"")
    text = body.decode("ascii", "replace")
    return "OK" if "OK" in text else text


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    path = sys.argv[1]
    if not os.path.exists(path):
        print("no such file:", path)
        return 2

    try:
        import PIL.Image
    except ImportError:
        print("needs pillow")
        return 2

    image = PIL.Image.open(path)
    image.thumbnail(TARGET_PX, PIL.Image.LANCZOS)
    if image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGB")

    buf = io.BytesIO()
    image.save(buf, format="png", compress_level=6)
    png = base64.standard_b64encode(buf.getvalue())
    raw = base64.standard_b64encode(bytearray().join(map(bytes, image.getdata())))

    attempts = [
        ("A  PNG inline   (f=100,t=d)", {"a": "T", "i": 91, "f": 100, "t": "d"}, png),
        (
            "B  raw RGB inline (f=%d,t=d)  <- what ranger sends today"
            % (len(image.getbands()) * 8),
            {
                "a": "T",
                "i": 92,
                "f": len(image.getbands()) * 8,
                "t": "d",
                "s": image.width,
                "v": image.height,
            },
            raw,
        ),
    ]

    fd = sys.stdin.fileno()
    try:
        old = termios.tcgetattr(fd)
    except termios.error:
        print("RESULT: stdin is not a tty — run this from a real pane")
        return 2

    print("image     : %s" % path)
    print("scaled to : %dx%d px" % (image.width, image.height))
    print()

    results = []
    tty.setcbreak(fd)
    try:
        for label, cmds, payload in attempts:
            send(cmds, payload)
            resp = reply()
            results.append((label, len(payload), resp))
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)

    print()
    for label, size, resp in results:
        print("  %-46s %7.1f KB payload -> %s" % (label, size / 1024.0, describe(resp)))
    print()
    print("Two images should be drawn above. Say which of A / B you can actually SEE —")
    print("a reply of OK only means herdr accepted it, not that it reached the screen.")
    print('Clear leftovers with:  printf "\\033_Ga=d\\033\\\\"')
    return 0


if __name__ == "__main__":
    sys.exit(main())
