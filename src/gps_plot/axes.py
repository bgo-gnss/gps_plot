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
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np
from geofunc.axes import (
    Axes,
    along_axes,
    baseline_azimuth,
    rotate,
    baseline_elevation,
    sign_axes,
    slant_axes,
    tilted_axes,
)

from gps_plot.series import Series

#: --sign: "apart" = + is lengthening per component; "raw" = rover − reference
SignMode = Literal["apart", "raw"]
SIGN_MODES: tuple[str, ...] = ("apart", "raw")

#: --axes: N/E/U (signed per --sign), along/across/up (horizontal along the
#: baseline), or along the 3-D line between the antennas
AXES_MODES: tuple[str, ...] = ("neu", "along", "3d", "motion", "slope")

#: the modes whose direction is not the baseline's but given (config
#: ``directions:``, ``--direction``/``--slope`` or ``--direction-from``)
GIVEN_MODES: tuple[str, ...] = ("motion", "slope")

#: the a-priori direction may be this far [°] off the RTK data vector; past
#: it the data vector is used (a station coordinate is off, or the rover moved
#: metres on a short baseline)
DIRECTION_TOLERANCE_DEG = 0.5

#: components whose + is the baseline lengthening when signed: N/E/U by
#: :func:`sign_axes`, along by construction (across and normal: a direction)
APART_KEYS = frozenset({"north", "east", "up", "along"})


def apriori(pair: str, *, coord_file: str | None = None) -> Any:
    """The a-priori N/E/U baseline [m] of ``pair`` (``ROVER-REF``), or None."""
    from geo_dataread.station_catalog import apriori_baseline

    try:
        return apriori_baseline(pair[:4], pair[5:], coord_file)
    except OSError:  # no coordinate file configured
        return None


def pair_axes(pair: str, *, coord_file: str | None = None) -> Axes | None:
    """N/E/U re-signed for ``pair`` (``ROVER-REF``), or None without coordinates."""
    b0 = apriori(pair, coord_file=coord_file)
    return None if b0 is None else sign_axes(b0)


def _up_signed(ax: Axes, b: Any) -> Axes:
    """``ax`` with its Up row signed as in ``neu`` (raw without ``b``)."""
    if b is None:
        return Axes(ax.matrix, ax.keys, (False, False, True))
    signed = sign_axes(b)
    m = np.array(ax.matrix, dtype=float)
    m[2] *= signed.matrix[2, 2]
    return Axes(m, ax.keys, (False, False, signed.raw[2]))


def pair_frame(b: Any, mode: str, direction: Mapping[str, Any] | None = None) -> Axes:
    """The axes of ``--axes mode`` for a baseline vector ``b`` (N/E/U, m).

    ``neu``: :func:`sign_axes`. ``along``: along/across/up, Up signed as in
    ``neu`` (so switching ``--axes`` changes only the horizontal panels).
    ``3d``: :func:`slant_axes` (along the line, across, normal). ``motion``:
    motion/across/up toward ``direction["motion"]`` [°] (+ = the rover
    moving that way), Up as in ``neu``. ``slope``: downslope/across/normal
    of ``direction["slope"]`` = (downslope azimuth, dip) [°], the normal
    out of the slope. ``b`` may be None for the last two (Up then A − B).

    Raises:
        ValueError: an unknown mode, or ``motion``/``slope`` without its
            direction.
    """
    if mode in GIVEN_MODES:
        got = (direction or {}).get(mode)
        if got is None:
            raise ValueError(f"no {mode} direction")
        if mode == "slope":
            az, dip = got
            return tilted_axes(
                np.radians(az), np.radians(dip), keys=("downslope", "across", "normal")
            )
        ax = tilted_axes(np.radians(got), 0.0, keys=("motion", "across", "up"))
        return _up_signed(ax, b)
    signed = sign_axes(b)
    if mode == "neu":
        return signed
    if mode == "3d":
        return slant_axes(b)
    if mode != "along":
        raise ValueError(f"--axes {mode!r}: choose from {', '.join(AXES_MODES)}")
    return _up_signed(along_axes(baseline_azimuth(b)), b)


def direction_note(b: Any, mode: str, direction: Mapping[str, Any] | None) -> str:
    """The direction of a pair's axes for its legend: ``"334°"``, ``"210°/28°"``.

    Empty for ``neu``.
    """
    if mode == "neu":
        return ""
    if mode == "motion":
        return f"{(direction or {})['motion']:.0f}°"
    if mode == "slope":
        az, dip = (direction or {})["slope"]
        return f"{az:.0f}°/{dip:.0f}°"
    az = np.degrees(baseline_azimuth(b))
    if mode == "3d":
        return f"{az:.0f}°/{np.degrees(-baseline_elevation(b)):.0f}°"
    return f"{az:.0f}°"


def angle_off(b0: Any, b: Any, *, horizontal: bool) -> float:
    """Angle [°] between two baseline vectors (their horizontal parts if asked)."""
    u, v = np.asarray(b0, dtype=float), np.asarray(b, dtype=float)
    if horizontal:
        u, v = u[:2], v[:2]
    c = float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def _is_pair(spec: str) -> bool:
    from geo_dataread.baseline import split_baseline

    return split_baseline(spec) is not None


def pair_orient(
    spec: str,
    s: Series,
    sign: str = "apart",
    axes: str = "neu",
    data: Any = None,
    direction: Mapping[str, Any] | None = None,
) -> Series:
    """A baseline ``s`` on its pair's axes (``--sign``/``--axes``).

    The direction is the a-priori baseline, or ``data`` (a measured vector,
    N/E/U m) when that is more than :data:`DIRECTION_TOLERANCE_DEG` off it
    or there are no coordinates (with a warning); ``motion``/``slope`` take
    theirs from ``direction`` (the pair's ``directions:`` entry). A single
    station is returned unchanged; so is a pair with no direction in ``neu``
    (A − B, with a warning), while the other modes refuse it (ValueError).
    ``meta["direction"]`` says the direction (:func:`direction_note`).
    """
    if not _is_pair(spec) or (axes == "neu" and sign == "raw"):
        return s
    b0 = apriori(spec)
    b = b0 if b0 is not None else data
    if b0 is None and data is not None:
        warnings.warn(
            f"{spec}: no a-priori coordinates, axes from the data", stacklevel=2
        )
    elif b0 is not None and data is not None and axes != "neu":
        off = angle_off(b0, data, horizontal=axes == "along")
        if off > DIRECTION_TOLERANCE_DEG:
            warnings.warn(
                f"{spec}: a-priori direction {off:.2f}° off the data, axes from "
                "the data",
                stacklevel=2,
            )
            b = data
    if axes in GIVEN_MODES:
        if (direction or {}).get(axes) is None:
            raise ValueError(
                f"{spec}: no {axes} direction (directions: in station_areas.yaml, "
                "--direction/--slope or --direction-from)"
            )
    elif b is None:
        if axes != "neu":
            raise ValueError(f"{spec}: no direction for --axes {axes}")
        warnings.warn(f"{spec}: no a-priori coordinates, plotted A - B", stacklevel=2)
        return s
    out = orient(s, pair_frame(b, axes, direction))
    note = direction_note(b, axes, direction)
    return (
        dataclasses.replace(out, meta={**out.meta, "direction": note}) if note else out
    )


def sign_pair(spec: str, s: Series, sign: str = "apart") -> Series:
    """A baseline ``s`` on its re-signed N/E/U (:func:`pair_orient`, ``neu``)."""
    return pair_orient(spec, s, sign)


def figure_labels(series: Sequence[Series], lang: Any, unit: str) -> tuple[str, ...]:
    """The y-axis labels of a figure of ``series``.

    Their components (all alike, or ValueError); "+ apart" on a component
    whose + is lengthening (:data:`APART_KEYS`) when every series is an
    oriented pair, "(A − B)" on one every series kept raw (a component kept
    raw in only some is reported on stderr).
    """
    from gps_plot import labels

    keys = {tuple(x.components) for x in series}
    if len(keys) != 1:
        raise ValueError(f"one figure, different axes: {sorted(keys)}")
    comps = keys.pop()
    apart, raw = label_flags(series)
    marks = [apart and k in APART_KEYS for k in comps]
    return labels.axis_labels(comps, lang, unit, apart=marks, raw=raw)


def label_flags(
    series: Sequence[Series],
) -> tuple[bool, tuple[bool, bool, bool]]:
    """What the axis labels of a figure of ``series`` may say.

    (every series is an oriented baseline, per component: every one kept it
    as A − B). A component kept raw in only some of them is reported on
    stderr; the label then says "+ apart" for the rest.
    """
    signed = [s for s in series if s.meta.get("axes") is not None]
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


#: a baseline day set aside is drawn under the most serious reason it had
#: (as :data:`gps_plot.sources.gps.OVERLAY_PRIORITY`)
_OVERLAYS = ("netday", "iced", "buried", "outliers")


def _mixed_overlays(s: Series, ax: Axes, y: Any) -> dict[str, Any]:
    """The set-aside overlays of ``s`` on axes that mix components.

    A rotated component is set aside wherever a component it mixes was
    (``y``, the rotated values, is NaN there): it is drawn from the whole
    vector of that epoch (the kept values completed by the overlays) under
    the most serious reason among the components it mixes.
    """
    full = np.array(s.y, dtype=float)
    for name in _OVERLAYS:
        pair = getattr(s, name)
        if pair is not None:
            v = np.asarray(pair[0], dtype=float)
            full = np.where(np.isnan(full) & np.isfinite(v), v, full)
    r = rotate(full.T, ax, sigma=np.asarray(s.sigma, dtype=float).T)
    assert r.sigma is not None
    yf, sf = r.values.T, r.sigma.T
    lost = np.isnan(y) & np.isfinite(yf)
    uses = (ax.matrix != 0.0).astype(float)
    taken = np.zeros_like(lost)
    out: dict[str, Any] = {}
    for name in _OVERLAYS:
        pair = getattr(s, name)
        if pair is None:
            out[name] = None
            continue
        flagged = np.isfinite(np.asarray(pair[0], dtype=float)).astype(float)
        cell = lost & ((uses @ flagged) > 0) & ~taken
        taken |= cell
        out[name] = (np.where(cell, yf, np.nan), np.where(cell, sf, np.nan))
    return out


def _rotate_full(y: Any, sigma: Any, corr: Any, ax: Axes) -> tuple[Any, Any, Any]:
    """(values, σ, corr) along ``ax``, each (3, N), through the covariance."""
    from geo_dataread.rtk_read import RHO_INDEX, covariance

    r = rotate(np.asarray(y, dtype=float).T, ax, cov=covariance(sigma, corr))
    assert r.cov is not None and r.sigma is not None
    sd = r.sigma  # (N, 3)
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = np.stack([r.cov[:, i, j] / (sd[:, i] * sd[:, j]) for i, j in RHO_INDEX])
    return r.values.T, sd.T, rho


def orient(s: Series, ax: Axes) -> Series:
    """``s`` along ``ax``: values, σ (full covariance when ``s.corr`` is
    known, else uncorrelated), overlays, ends, level.

    A component is ``aborted`` when any component it mixes was. ``meta``
    gains ``axes`` (the component keys), ``raw`` (components left as rover −
    reference) and ``apart`` (+ is lengthening).
    """
    if tuple(s.components) != ("north", "east", "up") or "axes" in s.meta:
        # a re-signed series is still N/E/U: signing it again would undo it
        raise ValueError(f"{s.label}: orient wants raw N/E/U, got {s.components}")
    corr = None
    if s.corr is not None:
        y, sigma, corr = _rotate_full(s.y, s.sigma, s.corr, ax)
    else:
        r = rotate(
            np.asarray(s.y, dtype=float).T,
            ax,
            sigma=np.asarray(s.sigma, dtype=float).T,
        )
        assert r.sigma is not None
        y, sigma = r.values.T, r.sigma.T
    uses = ax.matrix != 0.0
    aborted = None
    if s.aborted is not None:
        bad = np.asarray(s.aborted, dtype=bool)
        aborted = [bool((uses[i] & bad).any()) for i in range(3)]
    if np.count_nonzero(ax.matrix - np.diag(np.diag(ax.matrix))):
        overlays = _mixed_overlays(s, ax, y)
    else:  # a re-signing: each overlay keeps its own components
        overlays = {k: _overlay(getattr(s, k), ax) for k in _OVERLAYS}
    meta = dict(s.meta)
    if "level" in meta:
        meta["level"] = _rows(np.asarray(meta["level"])[:, None], ax)[:, 0]
    meta.update(axes=ax.keys, raw=ax.raw, apart=ax.keys == ("north", "east", "up"))
    ends = None
    if s.ends is not None:
        ends = (orient(s.ends[0], ax), orient(s.ends[1], ax))
    return dataclasses.replace(
        s,
        y=y,
        sigma=sigma,
        corr=corr,
        components=ax.keys,
        provisional=_overlay(s.provisional, ax),
        **overlays,
        aborted=aborted,
        ends=ends,
        meta=meta,
    )


class Orienter:
    """Puts each series of a figure on its pair's axes (``--sign``/``--axes``).

    For a stream of series read piece by piece (RTK): called on every series
    as it is read, before any alignment, so shifts and notes are in the
    plotted axes. A pair's axes are fixed on its first series: from the
    a-priori coordinates, checked against the RTK data vector (the absolute
    first-hour level of an RTK series) when that series is RTK; more than
    :data:`DIRECTION_TOLERANCE_DEG` off, or no coordinates, and the data
    vector is used, with a message. ``--axes neu`` without either leaves the
    pair rover − reference; ``along``/``3d`` refuse it (ValueError): one
    figure never mixes axes. ``--sign raw`` with ``neu`` changes nothing.
    """

    def __init__(
        self,
        sign: str = "apart",
        axes: str = "neu",
        directions: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        if axes not in AXES_MODES:
            raise ValueError(f"--axes {axes!r}: choose from {', '.join(AXES_MODES)}")
        self.sign, self.mode = sign, axes
        self.directions = dict(directions or {})
        self.found: dict[str, Axes | None] = {}
        self.notes: dict[str, str] = {}
        self.messages: list[str] = []

    def _say(self, text: str) -> None:
        self.messages.append(text)
        print(text, file=sys.stderr)

    def _resolve(self, pair: str, s: Series) -> Axes | None:
        b0 = apriori(pair)
        data = None
        if s.meta.get("source") == "rtk" and s.meta.get("level") is not None:
            data = np.asarray(s.meta["level"], dtype=float) / 1000.0  # mm → m
        if self.mode in GIVEN_MODES:
            given = self.directions.get(pair)
            if (given or {}).get(self.mode) is None:
                self._say(f"{pair}: no {self.mode} direction")
                return None
            b = b0 if b0 is not None else data
            self.notes[pair] = direction_note(b, self.mode, given)
            return pair_frame(b, self.mode, given)
        if b0 is None and data is None:
            self._say(f"{pair}: no a-priori coordinates, plotted A - B")
            return None
        b = b0 if b0 is not None else data
        if b0 is None:
            self._say(f"{pair}: no a-priori coordinates, axes from the RTK vector")
        elif data is not None and self.mode != "neu":
            off = angle_off(b0, data, horizontal=self.mode == "along")
            if off > DIRECTION_TOLERANCE_DEG:
                self._say(
                    f"{pair}: a-priori direction {off:.2f}° off the RTK vector, "
                    "axes from the RTK vector"
                )
                b = data
        self.notes[pair] = direction_note(b, self.mode, None)
        return pair_frame(b, self.mode)

    def __call__(self, s: Series) -> Series:
        if self.mode == "neu" and self.sign == "raw":
            return s
        pair = s.meta.get("pair", s.label)
        if not _is_pair(pair):
            self.found.setdefault(pair, None)
            return s
        if pair not in self.found:
            self.found[pair] = self._resolve(pair, s)
        ax = self.found[pair]
        if ax is None:
            if self.mode != "neu":
                raise ValueError(f"{pair}: no direction for --axes {self.mode}")
            return s
        out = orient(s, ax)
        note = self.notes.get(pair)
        return (
            dataclasses.replace(out, meta={**out.meta, "direction": note})
            if note
            else out
        )

    def summary(self) -> tuple[bool, tuple[bool, bool, bool]]:
        """(every pair re-signed, per component: every pair kept it raw).

        A component kept raw (under 1 m) in only some pairs of a group is
        reported on stderr.
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

    @property
    def keys(self) -> tuple[str, str, str]:
        """The plotted components (N/E/U unless ``--axes`` rotated them)."""
        for ax in self.found.values():
            if ax is not None:
                return ax.keys
        return ("north", "east", "up")

    def labels(self, lang: Any, unit: str) -> tuple[str, ...]:
        """The figure's y-axis labels (:func:`gps_plot.labels.axis_labels`)."""
        from gps_plot import labels

        signed, raw = self.summary()
        keys = self.keys
        apart = [signed and k in APART_KEYS for k in keys]
        return labels.axis_labels(keys, lang, unit, apart=apart, raw=raw)
