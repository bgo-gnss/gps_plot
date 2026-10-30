"""The normalized series every gps_plot data source returns.

A :class:`Series` is what a plot draws: epochs, values and sigmas per
component, plus what the source knows about them (overlays, abort flags,
whether a detrend was applied). Sources (``gps_plot.sources``) build it from
geo_dataread (later gps_api); layers and renderers only read it. Design:
``docs/cli-subcommands-design.md`` §3.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any

#: the N/E/U row order of every GPS series (mm)
NEU: tuple[str, ...] = ("north", "east", "up")


@dataclasses.dataclass(frozen=True)
class Series:
    """One plottable series: a station, a baseline or one end of a baseline.

    Attributes:
        label: What it is, e.g. ``"HOFN"``, ``"VFLN-VFLS"``.
        t: Epochs, fractional years, shape (N,).
        y: Values, shape (C, N), mm, rows in ``components`` order.
        sigma: One-sigma uncertainties, shape (C, N), mm.
        components: Row names of ``y``/``sigma``.
        outliers: Flagged epochs removed from ``y`` (cleaned view), as
            ``(values, sigmas)`` NaN outside the flags, or None.
        provisional: Epochs kept in ``y`` but pending a verdict, same shape
            as ``outliers``, or None.
        aborted: Per component, whether outlier detection aborted (the
            component is then served raw), or None.
        iced: Epochs removed from ``y`` as icing suspected (GAMIT autcln
            index), same shape as ``outliers`` (which then excludes them),
            or None.
        buried: Up epochs removed from ``y`` as antenna-under-snow suspected
            (geo_dataread.snow_burial), same shape as ``outliers`` (Up row
            only; ``outliers`` then excludes them), or None.
        netday: Network days (whole-solution bad days,
            geo_dataread.network_days) removed from ``y``, every component,
            same shape as ``outliers`` (which then excludes them), or None.
        detrend_applied: Whether a stored trajectory was subtracted.
        ends: For a baseline, its two stations' own series, zeroed at the
            baseline's first common epoch; None otherwise.
        meta: Free-form provenance for titles and notes (e.g.
            ``sigma_note`` for a baseline's σ).
        corr: Correlations of the component pairs (0,1), (0,2), (1,2)
            (N/E/U: NE, NU, EU), shape (3, N), with ``sigma`` the full
            covariance; None = not known, taken as uncorrelated (RTK has
            them, daily GLOBK series not yet).

    The arrays are passed through as the source produced them (no copies or
    casts), so a plot drawn from a Series is the one drawn from the arrays.
    """

    label: str
    t: Any
    y: Any
    sigma: Any
    components: tuple[str, ...] = NEU
    outliers: tuple[Any, Any] | None = None
    provisional: tuple[Any, Any] | None = None
    aborted: Sequence[bool] | None = None
    detrend_applied: bool = False
    ends: tuple[Series, Series] | None = None
    meta: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    iced: tuple[Any, Any] | None = None
    buried: tuple[Any, Any] | None = None
    netday: tuple[Any, Any] | None = None
    corr: Any = None
