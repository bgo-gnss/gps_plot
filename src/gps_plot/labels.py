"""Plot labels in Icelandic and English, one table for every figure.

File names stay English; what a figure SAYS comes from here, so a component
or a note is worded once. Each figure keeps its language: the classic RTK
plots are Icelandic, the gpsplot frames English.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

Lang = Literal["is", "en"]

#: component key → (Icelandic, English)
COMPONENTS: dict[str, tuple[str, str]] = {
    "north": ("Norður", "North"),
    "east": ("Austur", "East"),
    "up": ("Upp", "Up"),
    "along": ("Lengd", "Along"),
    "across": ("Þvert", "Across"),
    "normal": ("Þverlægt", "Normal"),
    "downslope": ("Niður hlíð", "Downslope"),
    "motion": ("Hreyfing", "Motion"),
}

#: words used in labels and notes → (Icelandic, English)
WORDS: dict[str, tuple[str, str]] = {
    "apart": ("+ fjær", "+ apart"),
    "raw": ("A − B", "A - B"),
}


def word(key: str, lang: Lang) -> str:
    """``key`` from :data:`WORDS` in ``lang``."""
    return WORDS[key][0 if lang == "is" else 1]


def component(key: str, lang: Lang) -> str:
    """Component name in ``lang`` (an unknown key is shown as given)."""
    pair = COMPONENTS.get(key)
    return key if pair is None else pair[0 if lang == "is" else 1]


def axis_labels(
    keys: Sequence[str],
    lang: Lang,
    unit: str,
    *,
    apart: bool | Sequence[bool] = False,
    raw: Sequence[bool] = (False, False, False),
) -> tuple[str, ...]:
    """Y-axis labels, e.g. ``"North [mm] (+ apart)"``, ``"Norður [cm] (+ fjær)"``.

    ``apart``: the components (all, or per component) are signed so that +
    is lengthening;
    ``raw[i]``: component i kept as rover − reference (too short to carry a
    sign), said instead.
    """
    marks = [apart] * len(keys) if isinstance(apart, bool) else list(apart)
    out = []
    for key, kept, mark in zip(keys, raw, marks, strict=True):
        text = f"{component(key, lang)} [{unit}]"
        if kept:
            text += f" ({word('raw', lang)})"
        elif mark:
            text += f" ({word('apart', lang)})"
        out.append(text)
    return tuple(out)
