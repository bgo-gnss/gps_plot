"""Station names that keep off each other: positions on paper [cm].

A dense cluster (Svartsengi: ~20 stations within ~6 km) cannot take every
name at its default spot, up and right of the symbol. :func:`place` tries,
for each name, the spots around its symbol, then further out on rings
(drawn with a leader line), and takes the first whose box overlaps no
name placed before it, no symbol and nothing outside the map. Crowded
points go first, so the open ones adapt to them. A name with no free spot
keeps the default one.

Pure numpy on paper coordinates: the caller projects (``mapproject``).
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence

import numpy as np

#: the default spot first (up and right: GMT justify BL), then the others
#: around the symbol: (dx, dy) direction and the justify that puts the
#: text on that side
NEAR: tuple[tuple[float, float, str], ...] = (
    (1, 1, "BL"), (1, -1, "TL"), (-1, 1, "BR"), (-1, -1, "TR"),
    (1, 0, "ML"), (-1, 0, "MR"), (0, 1, "BC"), (0, -1, "TC"),
)  # fmt: skip
#: rings further out, with a leader line [cm], and directions on each
RINGS_CM = (0.45, 0.75, 1.1, 1.5)
RING_DIRECTIONS = 16
#: a symbol's half size kept clear of names [cm]
SYMBOL_CM = 0.1
#: the gap between a symbol and its name at the near spots [cm]
GAP_CM = 0.07
#: text box per character and its height, at 1 pt [cm]: Helvetica-Bold
#: averages ~0.6 em, plus the white box's padding
CHAR_CM_PER_PT = 0.6 * 2.54 / 72
HEIGHT_CM_PER_PT = 1.25 * 2.54 / 72
PAD_CM = 0.06


@dataclasses.dataclass(frozen=True)
class Label:
    """Where a name goes: its anchor (x, y) [cm] and GMT justify; ``leader``
    when away from its symbol (a line from the symbol to the anchor)."""

    index: int
    x: float
    y: float
    justify: str
    leader: bool


def text_size(text: str, size_pt: float) -> tuple[float, float]:
    """The box of ``text`` at ``size_pt`` [cm]: width, height."""
    return (
        len(text) * CHAR_CM_PER_PT * size_pt + 2 * PAD_CM,
        HEIGHT_CM_PER_PT * size_pt + 2 * PAD_CM,
    )


def centred_box(
    x: float, y: float, text: str, size_pt: float
) -> tuple[float, float, float, float]:
    """The box of ``text`` centred at (x, y) [cm] (a place name)."""
    w, h = text_size(text, size_pt)
    return _box(x, y, "MC", w, h)


def _box(
    x: float, y: float, justify: str, w: float, h: float
) -> tuple[float, float, float, float]:
    """(x0, x1, y0, y1) of a w x h box anchored at (x, y) by ``justify``."""
    vert, horiz = justify[0], justify[1]
    x0 = {"L": x, "C": x - w / 2, "R": x - w}[horiz]
    y0 = {"B": y, "M": y - h / 2, "T": y - h}[vert]
    return x0, x0 + w, y0, y0 + h


def _justify(dx: float, dy: float) -> str:
    """The justify that puts text on the (dx, dy) side of its anchor."""
    vert = "B" if dy > 0.38 else "T" if dy < -0.38 else "M"
    horiz = "L" if dx > 0.38 else "R" if dx < -0.38 else "C"
    return vert + horiz


def place(
    xy: np.ndarray,
    texts: Sequence[str],
    *,
    size_pt: float,
    width: float,
    height: float,
    obstacles: np.ndarray | None = None,
    fixed: Sequence[tuple[float, float, float, float]] = (),
) -> list[Label]:
    """A :class:`Label` per point of ``xy`` (n x 2, cm), in input order.

    ``obstacles`` are further symbol positions (n x 2, cm) to keep clear,
    ``fixed`` boxes (x0, x1, y0, y1) already taken (place names); the map
    is ``width`` x ``height`` cm.
    """
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    symbols = xy if obstacles is None else np.vstack([xy, obstacles])
    n = len(xy)
    if n == 0:
        return []
    # crowded first: the number of other points within 1 cm
    d = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])
    order = np.argsort(-(d < 1.0).sum(axis=1), kind="stable")
    taken: list[tuple[float, float, float, float]] = list(fixed)
    out: dict[int, Label] = {}

    def free(box: tuple[float, float, float, float], own: int) -> bool:
        x0, x1, y0, y1 = box
        if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
            return False
        for t0, t1, u0, u1 in taken:
            if x0 < t1 and t0 < x1 and y0 < u1 and u0 < y1:
                return False
        near = (
            (symbols[:, 0] > x0 - SYMBOL_CM)
            & (symbols[:, 0] < x1 + SYMBOL_CM)
            & (symbols[:, 1] > y0 - SYMBOL_CM)
            & (symbols[:, 1] < y1 + SYMBOL_CM)
        )
        near[own] = False
        return not near.any()

    for i in (int(k) for k in order):
        x, y = xy[i]
        w, h = text_size(texts[i], size_pt)
        candidates: list[tuple[float, float, str, bool]] = []
        for dx, dy, just in NEAR:
            r = SYMBOL_CM + GAP_CM
            candidates.append((x + dx * r, y + dy * r, just, False))
        for radius in RINGS_CM:
            for k in range(RING_DIRECTIONS):
                a = 2 * math.pi * k / RING_DIRECTIONS + math.pi / 4
                dx, dy = math.cos(a), math.sin(a)
                candidates.append((x + dx * radius, y + dy * radius,
                                   _justify(dx, dy), True))  # fmt: skip
        chosen = None
        for cx, cy, just, leader in candidates:
            box = _box(cx, cy, just, w, h)
            if free(box, i):
                chosen = Label(i, cx, cy, just, leader)
                taken.append(box)
                break
        if chosen is None:  # no room: the default spot
            cx, cy, just, _ = candidates[0]
            taken.append(_box(cx, cy, just, w, h))
            chosen = Label(i, cx, cy, just, False)
        out[i] = chosen
    return [out[i] for i in range(n)]
