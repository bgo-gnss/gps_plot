"""Baseline series along other axes: re-signed N/E/U, along/across, …

:func:`orient` applies one :class:`geofunc.axes.Axes` to everything a
:class:`~gps_plot.series.Series` carries — values, sigmas, the overlays, the
two ends of a baseline, the RTK zero level — so a figure drawn from the
oriented series is self-consistent. :func:`pair_axes` builds the default for
a pair: N/E/U signed so that + is the baseline lengthening.
"""

from __future__ import annotations

import dataclasses
import sys
import warnings
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
from geofunc.axes import Axes, rotate, sign_axes

from gps_plot.series import Series

#: --sign: "apart" = + is lengthening per component; "raw" = rover − reference
SignMode = Literal["apart", "raw"]
SIGN_MODES: tuple[str, ...] = ("apart", "raw")


def pair_axes(pair: str, *, coord_file: str | None = None) -> Axes | None:
    """N/E/U re-signed for ``pair`` (``ROVER-REF``), or None without coordinates."""
    from geo_dataread.station_catalog import apriori_baseline

    try:
        b0 = apriori_baseline(pair[:4], pair[5:], coord_file)
    except OSError:  # no coordinate file configured
        return None
    return None if b0 is None else sign_axes(b0)


def _is_pair(spec: str) -> bool:
    from geo_dataread.baseline import split_baseline

    return split_baseline(spec) is not None


def sign_pair(spec: str, s: Series, sign: str = "apart") -> Series:
    """A baseline ``s`` on its re-signed N/E/U (``sign="apart"``).

    A single station, ``sign="raw"``, or a pair without a-priori
    coordinates (with a warning) is returned unchanged.
    """
    if sign == "raw" or not _is_pair(spec):
        return s
    ax = pair_axes(spec)
    if ax is None:
        warnings.warn(f"{spec}: no a-priori coordinates, plotted A - B", stacklevel=2)
        return s
    return orient(s, ax)


def label_flags(
    series: Sequence[Series],
) -> tuple[bool, tuple[bool, bool, bool]]:
    """What the axis labels of a figure of ``series`` may say.

    (every series is a re-signed baseline, per component: every one kept it
    as A − B). A component kept raw in only some of them is reported on
    stderr; the label then says "+ apart" for the rest.
    """
    signed = [s for s in series if s.meta.get("apart")]
    apart = bool(signed) and len(signed) == len(series)
    if not apart:
        return False, (False, False, False)
    raws = [tuple(s.meta.get("raw", (False,) * 3)) for s in signed]
    raw = [all(r[i] for r in raws) for i in range(3)]
    for s, r in zip(signed, raws, strict=True):
        short = [k for k, kept, all_ in zip(s.components, r, raw, strict=True)
                 if kept and not all_]  # fmt: skip
        if short:
            print(f"{s.label}: {'/'.join(short)} under 1 m, plotted A - B",
                  file=sys.stderr)  # fmt: skip
    return True, (raw[0], raw[1], raw[2])


def _rows(y: Any, ax: Axes) -> Any:
    """(3, N) rows along ``ax`` (geofunc works on (..., 3))."""
    return rotate(np.asarray(y, dtype=float).T, ax).values.T


def _overlay(pair: tuple[Any, Any] | None, ax: Axes) -> tuple[Any, Any] | None:
    if pair is None:
        return None
    values, sigmas = pair
    r = rotate(
        np.asarray(values, dtype=float).T,
        ax,
        sigma=np.asarray(sigmas, dtype=float).T,
    )
    assert r.sigma is not None
    return r.values.T, r.sigma.T


def orient(s: Series, ax: Axes) -> Series:
    """``s`` along ``ax``: values, σ (as uncorrelated), overlays, ends, level.

    A component is ``aborted`` when any component it mixes was. ``meta``
    gains ``axes`` (the component keys), ``raw`` (components left as rover −
    reference) and ``apart`` (+ is lengthening).
    """
    if tuple(s.components) != ("north", "east", "up") or "axes" in s.meta:
        # a re-signed series is still N/E/U: signing it again would undo it
        raise ValueError(f"{s.label}: orient wants raw N/E/U, got {s.components}")
    r = rotate(
        np.asarray(s.y, dtype=float).T, ax, sigma=np.asarray(s.sigma, dtype=float).T
    )
    assert r.sigma is not None
    uses = ax.matrix != 0.0
    aborted = None
    if s.aborted is not None:
        bad = np.asarray(s.aborted, dtype=bool)
        aborted = [bool((uses[i] & bad).any()) for i in range(3)]
    meta = dict(s.meta)
    if "level" in meta:
        meta["level"] = _rows(np.asarray(meta["level"])[:, None], ax)[:, 0]
    meta.update(axes=ax.keys, raw=ax.raw, apart=ax.keys == ("north", "east", "up"))
    ends = None
    if s.ends is not None:
        ends = (orient(s.ends[0], ax), orient(s.ends[1], ax))
    return dataclasses.replace(
        s,
        y=r.values.T,
        sigma=r.sigma.T,
        components=ax.keys,
        outliers=_overlay(s.outliers, ax),
        provisional=_overlay(s.provisional, ax),
        iced=_overlay(s.iced, ax),
        buried=_overlay(s.buried, ax),
        netday=_overlay(s.netday, ax),
        aborted=aborted,
        ends=ends,
        meta=meta,
    )


class Orienter:
    """Puts each series on its pair's re-signed N/E/U (``--sign apart``).

    For a stream of series read piece by piece (RTK): called on every series
    as it is read, before any alignment, so shifts and notes are in the
    plotted signs. One a-priori lookup per pair; a pair without coordinates
    stays rover − reference, with one warning; ``--sign raw`` changes nothing.
    """

    def __init__(self, sign: str) -> None:
        self.sign = sign
        self.found: dict[str, Axes | None] = {}

    def __call__(self, s: Series) -> Series:
        if self.sign == "raw":
            return s
        pair = s.meta.get("pair", s.label)
        if not _is_pair(pair):
            self.found.setdefault(pair, None)
            return s
        if pair not in self.found:
            self.found[pair] = pair_axes(pair)
            if self.found[pair] is None:
                print(f"{pair}: no a-priori coordinates, plotted A - B",
                      file=sys.stderr)  # fmt: skip
        ax = self.found[pair]
        return s if ax is None else orient(s, ax)

    def summary(self) -> tuple[bool, tuple[bool, bool, bool]]:
        """(every pair re-signed, per component: every pair kept it raw).

        What the axis labels say; a component kept raw (under 1 m) in only
        some pairs of a group is reported on stderr.
        """
        if not self.found or any(ax is None for ax in self.found.values()):
            return False, (False, False, False)
        signed = [ax for ax in self.found.values() if ax is not None]
        raw = [all(ax.raw[i] for ax in signed) for i in range(3)]
        for pair, ax in self.found.items():
            assert ax is not None
            short = [k for k, kept, all_ in zip(ax.keys, ax.raw, raw, strict=True)
                     if kept and not all_]  # fmt: skip
            if short:
                print(f"{pair}: {'/'.join(short)} under 1 m, plotted A - B",
                      file=sys.stderr)  # fmt: skip
        return True, (raw[0], raw[1], raw[2])
