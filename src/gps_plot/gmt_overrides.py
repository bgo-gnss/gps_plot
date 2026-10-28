"""GMT arguments for the drawing steps of a map (``gpsplot map --gmt``).

Every map is drawn in named steps, the :data:`LAYERS` (``coast``, ``dem``,
``glaciers``, ``stations``, …). ``--gmt LAYER:KEY=VALUE`` adds a GMT
argument to every PyGMT call of that step, or replaces gpsplot's own:

    --gmt coast:area_thresh=500/0/1     drop small islands (long name)
    --gmt coast:A=500/0/1               the same, GMT's letter
    --gmt glaciers:pen=0.3p,blue        glacier outline pen
    --gmt dem:t=20                      20 % transparent DEM

A KEY is a PyGMT long name of that call or a single GMT letter (PyGMT
warns about letters but passes them). It replaces gpsplot's value given
under either name: :data:`LAYERS` maps the long names gpsplot passes to
their letters, and a GMT-gated test proves each pair conflicts in PyGMT.
``text`` folds ``font`` and ``justify`` into one ``F`` (and does not
complain about both), so ``labels:F=…`` drops both.

Keys gpsplot must set itself are refused (:data:`OWNED`): the data and
the map frame (``region``/``projection``) and, on ``dem``, the colour map,
whose glacier slices sit :data:`gps_plot.maps.GLACIER_Z_OFFSET` up (use
``--dem-cmap``/``--glacier-cmap``).

The overrides live in a context variable for the extent of
:func:`overrides`, so the map functions need no extra argument;
:func:`layer_kw` merges them into each call.
"""

from __future__ import annotations

import contextlib
import contextvars
import difflib
import inspect
import re
from collections.abc import Iterator, Mapping, Sequence
from types import MappingProxyType
from typing import Any

#: drawing step -> (the PyGMT call it makes, gpsplot's long names -> letter)
LAYERS: dict[str, tuple[str, dict[str, str]]] = {
    "coast": ("Figure.coast", {"land": "G", "water": "S", "shorelines": "W",
                               "resolution": "D", "frame": "B"}),
    "smooth": ("grdfilter", {"filter": "F", "distance": "D"}),
    "sea-smooth": ("grdfilter", {"filter": "F", "distance": "D"}),
    "gradient": ("grdgradient", {"azimuth": "A", "normalize": "N"}),
    "glacier-gradient": ("grdgradient", {"azimuth": "A", "normalize": "N"}),
    "dem": ("Figure.grdimage", {"shading": "I", "nan_transparent": "Q"}),
    "contours": ("Figure.grdcontour", {"levels": "C", "annotation": "A",
                                       "pen": "W"}),
    "zones": ("Figure.plot", {"pen": "W", "fill": "G"}),
    "glaciers": ("Figure.plot", {"pen": "W", "fill": "G"}),
    "lakes": ("Figure.plot", {"pen": "W", "fill": "G"}),
    "coastline": ("Figure.plot", {"pen": "W"}),
    "overlay": ("Figure.plot", {"pen": "W", "fill": "G"}),
    # the named --layers (gps_plot.gis_layers), one step each
    "roads": ("Figure.plot", {"pen": "W", "fill": "G"}),
    "towns": ("Figure.plot", {"pen": "W", "fill": "G"}),
    "lavas": ("Figure.plot", {"pen": "W", "fill": "G"}),
    "graben": ("Figure.plot", {"pen": "W", "fill": "G"}),
    "fissures": ("Figure.plot", {"pen": "W", "fill": "G"}),
    "barriers": ("Figure.plot", {"pen": "W", "fill": "G"}),
    "volcanoes": ("Figure.plot", {"pen": "W", "fill": "G"}),
    "calderas": ("Figure.plot", {"pen": "W", "fill": "G"}),
    "swarms": ("Figure.plot", {"pen": "W", "fill": "G"}),
    "localroads": ("Figure.plot", {"pen": "W", "fill": "G"}),
    "buildings": ("Figure.plot", {"style": "S", "pen": "W", "fill": "G"}),
    "place-labels": ("Figure.text", {"font": "F", "justify": "F"}),
    "leaders": ("Figure.plot", {"pen": "W"}),
    "stations": ("Figure.plot", {"style": "S", "pen": "W", "fill": "G"}),
    "labels": ("Figure.text", {"font": "F", "justify": "F", "fill": "G",
                               "offset": "D"}),
    "vectors": ("Figure.velo", {"spec": "S", "pen": "W", "vector": "A",
                                "line": "L", "uncertainty_fill": "E"}),
    "legend": ("Figure.legend", {"position": "D", "box": "F"}),
    "grid": ("Figure.basemap", {"frame": "B"}),
    "frame": ("Figure.basemap", {"frame": "B"}),
    "logo": ("Figure.image", {"position": "D", "box": "F", "transparency": "t"}),
    "scalebar": ("Figure.scalebar", {}),
    # the Iceland locator inset and the zoom inset (gps_plot.insets)
    "locator": ("Figure.plot", {"pen": "W", "fill": "G"}),
    "inset": ("Figure.inset", {}),
    "zoom-box": ("Figure.plot", {"pen": "W"}),
}  # fmt: skip
#: keys gpsplot always sets itself
OWNED = frozenset({
    "grid", "outgrid", "data", "x", "y", "text", "textfiles", "imagefile",
    "region", "projection", "R", "J", "label", "incols", "i",
})  # fmt: skip
OWNED_BY_LAYER = {"dem": frozenset({"cmap", "C"})}

_LETTER = re.compile(r"[A-Za-z]")
_CURRENT: contextvars.ContextVar[Mapping[str, Mapping[str, Any]]] = (
    contextvars.ContextVar("gmt_overrides", default=MappingProxyType({}))
)


def parse(specs: Sequence[str]) -> dict[str, dict[str, Any]]:
    """``LAYER:KEY=VALUE`` (or ``LAYER:KEY`` for a bare flag) specs, by
    layer. A key given twice keeps the last value.

    Raises:
        ValueError: a malformed spec.
    """
    out: dict[str, dict[str, Any]] = {}
    for spec in specs:
        layer, sep, arg = spec.partition(":")
        key, eq, value = arg.partition("=")
        if not (sep and layer.strip() and key.strip()):
            raise ValueError(f"--gmt {spec!r}: use LAYER:KEY=VALUE")
        out.setdefault(layer.strip(), {})[key.strip()] = value if eq else True
    return out


def _signature_names(call: str, pygmt_mod: Any) -> set[str]:
    obj = pygmt_mod
    for part in call.split("."):
        obj = getattr(obj, part)
    return set(inspect.signature(obj).parameters) - {"self", "kwargs"}


def check(overrides: Mapping[str, Mapping[str, Any]], pygmt_mod: Any) -> None:
    """Refuse unknown layers, unknown or owned keys, before drawing (a
    misspelt key would otherwise reach GMT as a bogus option, and GMT
    can fail a step without failing the figure).

    Raises:
        ValueError: naming the closest valid layer or key.
    """
    for layer, kwargs in overrides.items():
        if layer not in LAYERS:
            close = difflib.get_close_matches(layer, LAYERS, n=3)
            hint = f"; did you mean {', '.join(close)}?" if close else ""
            raise ValueError(
                f"--gmt: no drawing step {layer!r}{hint} (steps: {', '.join(LAYERS)})"
            )
        call, _letters = LAYERS[layer]
        names = _signature_names(call, pygmt_mod)
        owned = OWNED | OWNED_BY_LAYER.get(layer, frozenset())
        for key in kwargs:
            if key in owned:
                extra = (
                    " (use --dem-cmap / --bathymetry-cmap / --glacier-cmap)"
                    if layer == "dem" else ""
                )  # fmt: skip
                raise ValueError(
                    f"--gmt {layer}:{key}: gpsplot sets {key!r} itself{extra}"
                )
            if _LETTER.fullmatch(key) or key in names:
                continue
            close = difflib.get_close_matches(key, sorted(names), n=3)
            hint = f"; did you mean {', '.join(close)}?" if close else ""
            raise ValueError(
                f"--gmt {layer}:{key}: not an option of {call.split('.')[-1]}"
                f" (a PyGMT long name or one GMT letter){hint}"
            )


@contextlib.contextmanager
def overrides(
    specs: Mapping[str, Mapping[str, Any]], pygmt_mod: Any | None = None
) -> Iterator[None]:
    """Apply ``specs`` (layer -> kwargs) to the maps drawn inside.

    Checked first when ``pygmt_mod`` is given (:func:`check`).
    """
    if pygmt_mod is not None:
        check(specs, pygmt_mod)
    frozen = MappingProxyType({k: MappingProxyType(dict(v)) for k, v in specs.items()})
    token = _CURRENT.set(frozen)
    try:
        yield
    finally:
        _CURRENT.reset(token)


def layer_kw(
    layer: str, *, exclude: Sequence[str] = (), **defaults: Any
) -> dict[str, Any]:
    """gpsplot's ``defaults`` for one call of ``layer``, merged with the
    user's overrides: a user key replaces a default given under either its
    long name or its letter. ``exclude`` drops user keys this call must
    not take (the second, shoreline-only coast call takes no fill)."""
    user = {k: v for k, v in _CURRENT.get().get(layer, {}).items() if k not in exclude}
    if not user:
        return defaults
    letters = LAYERS[layer][1]
    kept = {
        k: v
        for k, v in defaults.items()
        if k not in user and letters.get(k) not in user
    }
    return {**kept, **user}
