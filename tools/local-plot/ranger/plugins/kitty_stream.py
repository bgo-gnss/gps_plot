"""Make ranger's kitty image previews work inside herdr.

NOT installed by this repo — copy it into the ranger stow package, see
../../README.md ("Image previews inside herdr").

Ranger picks its Kitty graphics transfer mode from a single startup query
(`ranger/ext/img_display.py::KittyImageDisplayer._late_init`): it asks the
terminal about `t=f` (read this file yourself) and understands exactly two
answers — `OK` -> use temp files, `EBADF` -> send pixels inline. herdr renders
graphics on the attached *client*, so a path produced inside a pane means
nothing to it and it correctly replies `EINVAL: unsupported medium`. That is
neither answer, so ranger raises ImgDisplayUnsupportedException and turns image
previews off — even though inline transfer (`t=d`) works perfectly.

This forces the inline path and skips the query. Activated only when
RANGER_KITTY_FORCE_STREAM=1, so a plain kitty terminal keeps ranger's own
autodetection.
"""

import base64
import io
import os
import select
import struct
import sys
import warnings

import ranger.ext.img_display as img_display

_DEFAULT_CELL = (8, 16)  # px, only used if the terminal reports no pixel size
_REPLY_TIMEOUT = 4.0  # seconds; ranger's own read has no timeout at all

# Upper bound on the pixels in one preview, independent of encoded size.
# herdr/ghostty stores decoded images as RGBA, so cost is width*height*4, and it
# answers ENOMEM past roughly 8 MB. Measured on this setup: 1302x2075 (2.70 MP,
# 10.9 MB) -> ENOMEM even at a 68 KB payload, while 981x1556 (1.53 MP, 6.1 MB)
# -> OK at 170 KB. Encoded size is NOT the constraint. 1.5 MP reproduces the
# known-good geometry with a margin.
_DEFAULT_MAX_PIXELS = 1500000

_original_late_init = img_display.KittyImageDisplayer._late_init
_original_draw = img_display.KittyImageDisplayer.draw


def _log(message):
    """Append a diagnostic line when RANGER_KITTY_DEBUG names a file.

    ranger routes preview failures through `fm.notify`, which flashes in the
    status bar and is easy to miss, so trace the draw path to a file instead.
    """
    target = os.environ.get("RANGER_KITTY_DEBUG")
    if not target:
        return
    try:
        with open(target, "a") as handle:
            handle.write(message + "\n")
    except OSError:
        pass


def _cell_size():
    """Pixel size of one cell, as (width, height).

    Ranger derives this from TIOCGWINSZ, but multiplexers commonly report
    0x0 pixels. A zero here silently collapses ranger's scale factor and the
    preview comes out blank, so fall back to an explicit value instead.
    """
    override = os.environ.get("RANGER_KITTY_CELL_PX")
    if override:
        try:
            width, height = (int(part) for part in override.split("x", 1))
            if width > 0 and height > 0:
                return width, height
        except ValueError:
            pass

    try:
        import fcntl
        import termios

        packed = fcntl.ioctl(
            sys.stdout, termios.TIOCGWINSZ, struct.pack("HHHH", 0, 0, 0, 0)
        )
        rows, cols, xpix, ypix = struct.unpack("HHHH", packed)
        if xpix and ypix and rows and cols:
            return xpix // cols, ypix // rows
    except (OSError, ValueError):
        pass

    return _DEFAULT_CELL


def _max_pixels():
    """Pixel ceiling for one preview; RANGER_KITTY_MAX_PIXELS overrides."""
    override = os.environ.get("RANGER_KITTY_MAX_PIXELS")
    if override:
        try:
            value = int(override)
            if value > 0:
                return value
        except ValueError:
            pass
    return _DEFAULT_MAX_PIXELS


def _late_init(self):
    if os.environ.get("RANGER_KITTY_FORCE_STREAM") != "1":
        _original_late_init(self)
        return

    # Inline base64 pixel data — no capability query, so nothing to hang on.
    self.stream = True

    try:
        import PIL.Image
    except ImportError:
        raise img_display.ImageDisplayError(
            "Image previews in kitty require PIL (pillow)"
        )
    self.backend = PIL.Image

    # Ranger's own names for these two are transposed relative to struct
    # winsize; keep its convention so draw() computes the same box it always
    # did: pix_row is px-per-column, pix_col is px-per-row.
    self.pix_row, self.pix_col = _cell_size()
    self.needs_late_init = False


def _draw(self, path, start_x, start_y, width, height):
    """Send the preview as inline PNG instead of inline raw RGB.

    Ranger's stream mode transmits the flattened pixel buffer, so a modest
    1000x1000 preview becomes ~3 MB (~4 MB base64). herdr caps the frame it
    forwards to the attached client and logs `dropping oversized graphics
    payload for client frame`, which shows up as a blank preview: the terminal
    replies OK, then the image never reaches the screen. The Kitty protocol
    takes PNG bytes inline too (`f=100`), which is ~50x smaller for a plot.
    """
    if os.environ.get("RANGER_KITTY_FORCE_STREAM") != "1":
        _original_draw(self, path, start_x, start_y, width, height)
        return

    self.image_id += 1
    if self.needs_late_init:
        self._late_init()

    # Logged before PIL touches the file: if a path shows up here with no
    # matching "draw" line below, the failure is decoding, not transfer.
    _log("enter %s (id=%d)" % (path, self.image_id))

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("ignore", self.backend.DecompressionBombWarning)
        image = self.backend.open(path)

        box = (width * self.pix_row, height * self.pix_col)
        if image.width > box[0] or image.height > box[1]:
            scale = min(box[0] / image.width, box[1] / image.height)
            # A zero dimension makes PIL raise; clamp so a tiny pane degrades
            # to a 1px image rather than killing the preview.
            image = image.resize(
                (max(1, int(scale * image.width)), max(1, int(scale * image.height))),
                self.backend.LANCZOS,
            )
        # Ranger only shrinks an image that overflows the preview box. On a
        # HiDPI cell size the box is large enough that a full-page PDF render
        # fits inside it untouched, so it is sent at native resolution and the
        # terminal rejects it. Cap the pixel count regardless of the box.
        max_pixels = _max_pixels()
        if image.width * image.height > max_pixels:
            scale = (max_pixels / float(image.width * image.height)) ** 0.5
            image = image.resize(
                (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
                self.backend.LANCZOS,
            )

        if image.mode not in ("RGB", "RGBA"):
            image = image.convert("RGB")

        buf = io.BytesIO()
        image.save(buf, format="png", compress_level=6)

    payload = base64.standard_b64encode(buf.getvalue())
    cmds = {"a": "T", "i": self.image_id, "f": 100, "t": "d"}

    _log(
        "draw %s\n     cells=%sx%s at (%s,%s) cell_px=%sx%s "
        "box=%sx%s img=%sx%s payload=%.1fKB"
        % (
            path,
            width,
            height,
            start_x,
            start_y,
            self.pix_row,
            self.pix_col,
            box[0],
            box[1],
            image.width,
            image.height,
            len(payload) / 1024.0,
        )
    )

    with img_display.temporarily_moved_cursor(int(start_y), int(start_x)):
        for cmd_str in self._format_cmd_str(cmds, payload=payload):
            self.stdbout.write(cmd_str)

    # Ranger reads the reply with a bare, unbounded stdin.read(1) loop, so a
    # terminal that stays silent wedges the whole UI. Bound it instead.
    resp = b""
    while not resp.endswith(self.protocol_end):
        if not select.select([self.stdbin], [], [], _REPLY_TIMEOUT)[0]:
            _log("     -> TIMEOUT after %.0fs, partial=%r" % (_REPLY_TIMEOUT, resp))
            raise img_display.ImageDisplayError(
                "no reply to a kitty graphics draw within %.0fs "
                "(terminal accepted the image but never acknowledged it)"
                % _REPLY_TIMEOUT
            )
        chunk = self.stdbin.read(1)
        if not chunk:
            break
        resp += chunk

    _log("     -> reply %r" % resp)
    if b"OK" not in resp:
        raise img_display.ImageDisplayError('kitty replied "{}"'.format(resp))


img_display.KittyImageDisplayer._late_init = _late_init
img_display.KittyImageDisplayer.draw = _draw
