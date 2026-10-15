"""Colour maps of ``gpsplot map`` DEM backgrounds: lookup and stitching.

A map background has up to three colour layers, each named on the
command line (``--dem-cmap``, ``--bathymetry-cmap``, ``--glacier-cmap``):
land (z >= 0), the sea floor (z < 0) and the glaciers. A name resolves,
first match wins, to

1. an existing CPT file path;
2. ``<name>.cpt`` in the user's directory, :func:`user_dir`
   (``$XDG_CONFIG_HOME/gps_plot/cpt``);
3. ``<name>.cpt`` shipped with gpsplot (:data:`GPSPLOT_DIR`: lightgrey,
   slate, relief, sea);
4. a GMT built-in CPT (``<GMT share>/cpt/**``: geo, sealand, batlow, …);
5. else a GMT colour or colour list (``white``, ``200/200/200``,
   ``white,lightblue``), checked by GMT.

A CPT file keeps its own z (metres; GMT's rule for files) and is cut to
the layer's range; a GMT built-in or a colour is stretched over it.
A hinged GMT CPT (geo, sealand, …) splits at 0, so ``-T0/max`` gives its
land half and ``-Tmin/0`` its sea half.

:func:`stitch` joins the layers into one CPT whose slices meet exactly —
GMT rejects a CPT with a gap and then aborts ``grdimage`` without failing
the figure. Every boundary is written from one Python float, never
re-formatted from makecpt's text.
"""

from __future__ import annotations

import contextlib
import dataclasses
import difflib
import os
import re
import tempfile
from collections.abc import Iterable, Sequence
from functools import cache
from pathlib import Path
from typing import Any

#: CPTs shipped with gpsplot
GPSPLOT_DIR = Path(__file__).with_name("cpt")
#: sea-floor colours when the land colour map has no sea half
DEFAULT_SEA = "sea"
#: ``--bathymetry-cmap none``: no sea-floor colours or relief
NONE = "none"

FILE, GMT, COLOUR = "file", "gmt", "colour"
_NUMBER = re.compile(r"[-+.\deE]+")


@dataclasses.dataclass(frozen=True)
class Palette:
    """A resolved colour-map name.

    Attributes:
        name: The name as given.
        kind: :data:`FILE` (absolute z), :data:`GMT` (stretched) or
            :data:`COLOUR` (a colour or list, stretched).
        source: The CPT path (files), the GMT name or the colour text.
    """

    name: str
    kind: str
    source: str


@dataclasses.dataclass(frozen=True)
class Slice:
    """One CPT slice: colours ``lo_colour`` at ``lo`` to ``hi_colour`` at ``hi``."""

    lo: float
    lo_colour: str
    hi: float
    hi_colour: str


def user_dir() -> Path:
    """The user's CPT directory, ``$XDG_CONFIG_HOME/gps_plot/cpt``."""
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    return Path(base) / "gps_plot" / "cpt"


@cache
def gmt_cpt_dir() -> Path | None:
    """GMT's built-in CPT directory, or None when GMT cannot say."""
    from pygmt.clib import Session

    with Session() as lib:
        share = lib.info.get("share dir")
    return Path(share, "cpt") if share else None


def _cpts(directory: Path | None) -> dict[str, Path]:
    if directory is None or not directory.is_dir():
        return {}
    return {p.stem.lower(): p for p in sorted(directory.rglob("*.cpt"))}


def resolve(name: str, *, gmt_dir: Path | None = None) -> Palette:
    """Resolve a colour-map name (see the module doc for the order).

    ``gmt_dir`` defaults to :func:`gmt_cpt_dir`. A GMT name keeps its
    modifiers (``geo+h0``).
    """
    path = Path(os.path.expanduser(name))
    if path.is_file():
        return Palette(name, FILE, str(path.resolve()))  # makecpt runs elsewhere
    key = name.lower()
    for directory in (user_dir(), GPSPLOT_DIR):
        found = _cpts(directory).get(key)
        if found is not None:
            return Palette(name, FILE, str(found))
    gmt_dir = gmt_dir if gmt_dir is not None else gmt_cpt_dir()
    if key.split("+", 1)[0] in _cpts(gmt_dir):
        return Palette(name, GMT, name)
    return Palette(name, COLOUR, name)


def is_hinged(palette: Palette, *, gmt_dir: Path | None = None) -> bool:
    """Whether a colour map also colours the sea: a GMT CPT with a hinge
    (geo, sealand, …) or a CPT file reaching below 0."""
    if palette.kind == GMT:
        gmt_dir = gmt_dir if gmt_dir is not None else gmt_cpt_dir()
        path = _cpts(gmt_dir).get(palette.source.lower().split("+", 1)[0])
        return path is not None and "HINGE" in path.read_text(errors="replace")
    if palette.kind == FILE:
        slices = parse(Path(palette.source).read_text())
        return bool(slices) and slices[0].lo < 0
    return False


def available(*, gmt_dir: Path | None = None) -> dict[str, list[str]]:
    """Colour-map names by source: gpsplot's, the user's, GMT's by group."""
    gmt_dir = gmt_dir if gmt_dir is not None else gmt_cpt_dir()
    out = {
        f"gpsplot ({GPSPLOT_DIR})": sorted(_cpts(GPSPLOT_DIR)),
        f"yours ({user_dir()})": sorted(_cpts(user_dir())),
    }
    groups: dict[str, list[str]] = {}
    for key, path in _cpts(gmt_dir).items():
        group = path.parent.name if gmt_dir and path.parent != gmt_dir else "gmt"
        groups.setdefault(f"GMT {group}", []).append(key)
    out.update({k: sorted(v) for k, v in sorted(groups.items())})
    return out


def suggest(name: str, *, gmt_dir: Path | None = None) -> list[str]:
    """Known colour-map names close to ``name``."""
    known = [n for names in available(gmt_dir=gmt_dir).values() for n in names]
    return difflib.get_close_matches(name.lower(), known, n=4)


def parse(text: str) -> list[Slice]:
    """The slices of a CPT: ``z0 c0 z1 c1`` (``r/g/b``, a name, …; as
    makecpt ``-Fr`` writes it) or ``z0 r g b z1 r g b``. Comments and
    B/F/N lines are skipped."""
    out = []
    for line in text.splitlines():
        t = line.split()
        if not t or t[0].startswith("#") or t[0] in ("B", "F", "N"):
            continue
        if len(t) >= 8 and all(_NUMBER.fullmatch(v) for v in t[:8]):
            t = [t[0], "/".join(t[1:4]), t[4], "/".join(t[5:8])]
        out.append(Slice(float(t[0]), t[1], float(t[2]), t[3]))
    return out


def _makecpt(pygmt_mod: Any, **kwargs: Any) -> list[Slice]:
    # from inside the temporary directory: GMT writes a colour list
    # ("white,white") as gmtapi_colors2cpt_<pid>.cpt into the working one
    with tempfile.TemporaryDirectory() as tmp, contextlib.chdir(tmp):
        path = os.path.join(tmp, "s.cpt")
        pygmt_mod.makecpt(color_model="r", output=path, **kwargs)
        return parse(Path(path).read_text())


def segment(pygmt_mod: Any, palette: Palette, lo: float, hi: float) -> list[Slice]:
    """``palette``'s slices over ``lo``..``hi``, the end slices pinned to
    exactly ``lo`` and ``hi``.

    A file is cut to the range (its z is absolute); its end colour
    stretches over whatever part of the range it does not reach. A GMT
    CPT or colour is sampled over the range.

    Raises:
        ValueError: ``lo >= hi``.
        pygmt.exceptions.GMTCLibError: GMT does not know the CPT/colour.
    """
    if not lo < hi:
        raise ValueError(f"empty colour range {lo!r}..{hi!r}")
    if palette.kind == FILE:
        whole = parse(Path(palette.source).read_text())
        if not whole:
            raise ValueError(f"no colour slices in {palette.source}")
        cut_lo, cut_hi = max(lo, whole[0].lo), min(hi, whole[-1].hi)
        if cut_lo < cut_hi:
            slices = _makecpt(pygmt_mod, cmap=palette.source, truncate=[cut_lo, cut_hi])
        else:  # the file lies wholly outside: its nearest end colour
            end = whole[0] if hi <= whole[0].lo else whole[-1]
            colour = end.lo_colour if end is whole[0] else end.hi_colour
            slices = [Slice(lo, colour, hi, colour)]
    else:
        cmap = palette.source
        if palette.kind == COLOUR and "," not in cmap:
            cmap = f"{cmap},{cmap}"
        slices = _makecpt(pygmt_mod, cmap=cmap, series=[lo, hi])
    slices = [s for s in slices if s.hi > lo and s.lo < hi]
    slices[0] = dataclasses.replace(slices[0], lo=lo)
    slices[-1] = dataclasses.replace(slices[-1], hi=hi)
    return slices


def shift(slices: Iterable[Slice], dz: float) -> list[Slice]:
    """``slices`` moved up by ``dz``."""
    return [dataclasses.replace(s, lo=s.lo + dz, hi=s.hi + dz) for s in slices]


def stitch(parts: Sequence[Sequence[Slice]]) -> str:
    """CPT text of ``parts`` (each ascending, bottom part first). A gap
    between two parts is filled with the upper part's first colour; the
    slices meet exactly, with z written at full precision.

    Raises:
        ValueError: overlapping parts.
    """
    out: list[Slice] = []
    for part in parts:
        for s in part:
            if out and s.lo != out[-1].hi:
                if s.lo < out[-1].hi:
                    raise ValueError(f"colour slices overlap at {s.lo!r}")
                out.append(Slice(out[-1].hi, s.lo_colour, s.lo, s.lo_colour))
            out.append(s)
    return "".join(f"{s.lo!r} {s.lo_colour} {s.hi!r} {s.hi_colour}\n" for s in out)


def check_gaps(text: str) -> None:
    """Raise when a CPT's slices do not join exactly (GMT would abort
    grdimage, but the figure is still saved, without its background)."""
    slices = parse(text)
    for prev, s in zip(slices, slices[1:], strict=False):
        if s.lo != prev.hi:
            raise ValueError(f"CPT slices do not join: {prev.hi!r} then {s.lo!r}")
