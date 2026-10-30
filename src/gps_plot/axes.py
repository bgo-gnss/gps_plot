"""Baseline series along other axes: re-signed N/E/U, along/across, …

:func:`orient` applies one :class:`geofunc.axes.Axes` to everything a
:class:`~gps_plot.series.Series` carries — values, sigmas, the overlays, the
two ends of a baseline, the RTK zero level — so a figure drawn from the
oriented series is self-consistent. :func:`pair_axes` builds the default for
a pair: N/E/U signed so that + is the baseline lengthening.
"""

from __future__ import annotations

import dataclasses
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
    if tuple(s.components) != ("north", "east", "up"):
        raise ValueError(f"{s.label}: orient wants N/E/U, got {s.components}")
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
