"""The ``profile`` panel of ``gpsplot polar``: the ground along each rover's
motion with its displacement since the onset on it.

One sub-panel per pair: the DEM profile along the pair's motion direction
through the rover (true heights, equal axes so the angles are true), the
rover at its height on the ground and its daily points from it, scaled up
equally in both directions (m per mm), with

* the ground slope at the rover along the motion (the 50 m plane's
  apparent dip, dashed);
* weekly medians (diamonds; a week needs :data:`MIN_PER_BIN` days);
* the plunge: a line through the rover fitted to the daily points with
  their σ, errors in both directions (:func:`plunge_fit`), and its 1σ wedge.

RTK points are reduced to daily medians first, so the weeks and the fit
count days, not 10-minute epochs (BGÓ 2026-11-01).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any

import numpy as np

#: the profile runs this far [m] either side of the rover
PROFILE_HALF_M = 250.0
#: weekly medians: days per bin from the onset, and the days a bin needs
BIN_DAYS = 7.0
MIN_PER_BIN = 3
#: the profile's ground samples
SAMPLES = 801


@dataclasses.dataclass
class Plunge:
    """The fitted plunge [°, + down] along the motion, its 1σ [°], the
    misfit (χ² per degree of freedom) and the number of days fitted."""

    deg: float
    sigma: float
    chi2: float
    n: int


def apparent_dip(motion: float, downslope: tuple[float, float]) -> float:
    """The dip [°, + down] of the ground plane (``downslope`` = (azimuth,
    dip) [°]) seen along ``motion`` [°]."""
    az, dip = downslope
    return float(np.degrees(np.arctan(np.tan(np.radians(dip))
                                      * np.cos(np.radians(motion - az)))))  # fmt: skip


def daily_medians(t: Any, *columns: Any) -> tuple[Any, ...]:
    """(t, *columns) as one median per day (``t`` in days since the onset);
    daily data passes through as it is."""
    t = np.asarray(t, dtype=float)
    day = np.floor(t)
    days = np.unique(day)
    if days.size == t.size:
        order = np.argsort(t)
        return (t[order], *(np.asarray(c, dtype=float)[order] for c in columns))
    out = [np.array([np.median(t[day == d]) for d in days])]
    for c in columns:
        c = np.asarray(c, dtype=float)
        out.append(np.array([np.median(c[day == d]) for d in days]))
    return tuple(out)


def weekly_medians(t: Any, along: Any, up: Any) -> tuple[Any, Any, Any]:
    """(t, along, up) medians per :data:`BIN_DAYS` bin from the onset, of the
    bins with at least :data:`MIN_PER_BIN` days."""
    t, along, up = (np.asarray(v, dtype=float) for v in (t, along, up))
    week = np.floor(t / BIN_DAYS)
    rows = [(np.median(t[m]), np.median(along[m]), np.median(up[m]))
            for m in (week == w for w in np.unique(week))
            if np.count_nonzero(m) >= MIN_PER_BIN]  # fmt: skip
    got = np.array(rows, dtype=float).reshape(-1, 3).T
    return got[0], got[1], got[2]


def plunge_fit(along: Any, up: Any, s_along: Any, s_up: Any) -> Plunge | None:
    """The line up = k · along through the rover fitted to the points [mm]
    with errors in both (effective variance s_up² + k² s_along², iterated);
    the 1σ from the points' σ, scaled up by the misfit when the scatter
    exceeds them (χ² per dof > 1). None with fewer than 2 points or no
    motion along."""
    along, up, s_along, s_up = (np.asarray(v, dtype=float)
                                for v in (along, up, s_along, s_up))  # fmt: skip
    ok = (np.isfinite(along) & np.isfinite(up) & np.isfinite(s_along)
          & np.isfinite(s_up) & (s_up > 0))  # fmt: skip
    along, up, s_along, s_up = along[ok], up[ok], s_along[ok], s_up[ok]
    if along.size < 2 or not np.any(along):
        return None
    k = float(np.sum(along * up) / np.sum(along**2))
    for _ in range(50):
        w = 1.0 / (s_up**2 + k**2 * s_along**2)
        new = float(np.sum(w * along * up) / np.sum(w * along**2))
        if abs(new - k) < 1e-9:
            break
        k = new
    w = 1.0 / (s_up**2 + k**2 * s_along**2)
    chi2 = float(np.sum(w * (up - k * along) ** 2) / (along.size - 1))
    sk = np.sqrt(max(chi2, 1.0) / np.sum(w * along**2))
    return Plunge(float(np.degrees(np.arctan(-k))), float(np.degrees(sk / (1 + k**2))),
                  chi2, int(along.size))  # fmt: skip


def sample(dem: Any, x: Any, y: Any) -> Any:
    """The DEM (:class:`gps_plot.terrain.DemWindow`) at (x, y) [m],
    bilinear; NaN outside it."""
    gx, gy, z = (np.asarray(v, dtype=float) for v in (dem.x, dem.y, dem.z))
    if gx[0] > gx[-1]:
        gx, z = gx[::-1], z[:, ::-1]
    if gy[0] > gy[-1]:
        gy, z = gy[::-1], z[::-1, :]
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    fx = np.interp(x, gx, np.arange(gx.size), left=np.nan, right=np.nan)
    fy = np.interp(y, gy, np.arange(gy.size), left=np.nan, right=np.nan)
    out = np.full(np.broadcast(fx, fy).shape, np.nan)
    ok = np.isfinite(fx) & np.isfinite(fy)
    i0 = np.minimum(np.floor(fx[ok]).astype(int), gx.size - 2)
    j0 = np.minimum(np.floor(fy[ok]).astype(int), gy.size - 2)
    u, v = fx[ok] - i0, fy[ok] - j0
    out[ok] = ((1 - u) * (1 - v) * z[j0, i0] + u * (1 - v) * z[j0, i0 + 1]
               + (1 - u) * v * z[j0 + 1, i0] + u * v * z[j0 + 1, i0 + 1])  # fmt: skip
    return out


@dataclasses.dataclass
class Profile:
    """One pair's sub-panel, ready to draw: the ground ``z`` at ``s`` [m]
    along the motion (NaN without a DEM), the rover's height ``z0``, the
    daily points (``t`` days, ``along``/``up`` mm, their σ), the plunge and
    the height range ``lo``..``hi`` [m] holding the ground and the points."""

    pp: Any
    s: Any
    z: Any
    z0: float
    t: Any
    along: Any
    up: Any
    s_along: Any
    s_up: Any
    plunge: Plunge | None
    lo: float
    hi: float

    @property
    def west(self) -> bool:
        """The motion heads west: drawn to the left (as the section)."""
        return bool(np.sin(np.radians(self.pp.motion)) < 0)


#: the RTK points inside these percentiles set the scale and the height
#: range; the hourly extremes beyond may leave the panel (one wild hour
#: halved the scale)
RTK_PCT = (1.0, 99.0)


def _points(pp: Any) -> list[Any]:
    """The (3, n) displacements [mm] that set the pair's sub-panel: the
    daily ones and, with ``--rtk``, the RTK tail and its path (per
    component inside :data:`RTK_PCT`)."""
    out = [pp.d] if pp.d.shape[1] else []
    rtk = getattr(pp, "rtk", None)
    if rtk is not None and np.size(getattr(pp, "rtk_since", [])):
        d = np.asarray(rtk.d, float)
        if np.isfinite(d).any(axis=1).all():
            lo, hi = np.nanpercentile(d, RTK_PCT, axis=1)
            d = np.where((d >= lo[:, None]) & (d <= hi[:, None]), d, np.nan)
        out += [d, np.asarray(rtk.path, float)]
    return out


def scale_for(pairs: Sequence[Any], half: float = PROFILE_HALF_M) -> float:
    """The profile's m per mm: the largest displacement drawn (the RTK
    points too: BGÓ 2026-11-13, they ran out of the panel) fills about half
    of ``half``."""
    from gps_plot.polar import _nice

    sizes = [np.hypot(np.hypot(d[0], d[1]), np.nan_to_num(d[2]))
             for pp in pairs for d in _points(pp)]  # fmt: skip
    big = max((float(np.nanmax(x)) for x in sizes if np.isfinite(x).any()),
              default=1.0)  # fmt: skip
    return _nice(0.5 * half / max(big, 1e-6))


def prepare(pp: Any, dem: Any, scale: float, half: float = PROFILE_HALF_M) -> Profile:
    """``pp``'s (:class:`gps_plot.polar.PairPlot`, with a motion) profile
    on ``dem`` (None: no ground, heights from the rover)."""
    a = np.radians(pp.motion)
    ux, uy = np.sin(a), np.cos(a)
    s = np.linspace(-half, half, SAMPLES)
    if dem is None:
        z, z0 = np.full_like(s, np.nan), 0.0
    else:
        x0, y0 = pp.pos
        z = sample(dem, x0 + s * ux, y0 + s * uy)
        z0 = float(sample(dem, np.array([x0]), np.array([y0]))[0])
        if not np.isfinite(z0):  # the rover off the DEM: no ground to stand on
            z, z0 = np.full_like(s, np.nan), 0.0
    sig = pp.sig if pp.sig is not None else np.full_like(pp.d, np.nan)
    along = pp.d[0] * uy + pp.d[1] * ux
    s_along = np.hypot(sig[0] * uy, sig[1] * ux)
    ok = np.isfinite(along) & np.isfinite(pp.d[2])
    t, along, up, s_along, s_up = daily_medians(pp.since[ok], along[ok], pp.d[2][ok],
                                                s_along[ok], sig[2][ok])  # fmt: skip
    rtk_up = [d[2][np.isfinite(d[2])] for d in _points(pp)[1:]]
    heights = np.r_[z[np.isfinite(z)], z0 + up * scale, z0,
                    *(z0 + u * scale for u in rtk_up)]  # fmt: skip
    span = float(np.ptp(heights)) if heights.size else 1.0
    return Profile(pp, s, z, z0, t, along, up, s_along, s_up,
                   plunge_fit(along, up, s_along, s_up),
                   float(np.min(heights)) - max(15.0, 0.05 * span),
                   float(np.max(heights)) + max(25.0, 0.1 * span))  # fmt: skip


def up_text(pp: Any) -> str:
    """ ", up ±σ mm/d" of the pair's 14-day Up rate, for the arrow's label."""
    vt = pp.track.vertical if pp.track is not None else None
    if vt is None or vt.recent is None:
        return ""
    return f", up {vt.recent[0] / 365.25:+.2f} ± {vt.recent[1] / 365.25:.2f} mm/d"


def draw(ax: Any, prof: Profile, style: dict[str, Any], scale: float, bar_mm: float,
         half: float = PROFILE_HALF_M, small: bool = False) -> None:  # fmt: skip
    """One pair's sub-panel on ``ax``."""
    from gps_plot.polar import _usetex, compass

    pp = prof.pp
    sign = -1.0 if prof.west else 1.0
    lo, hi, z0 = prof.lo, prof.hi, prof.z0
    if np.isfinite(prof.z).any():
        ax.fill_between(sign * prof.s, prof.z, lo, color="0.88", zorder=0)
        ax.plot(sign * prof.s, prof.z, color="0.2", lw=1.2, zorder=1)
    if pp.downslope is not None:
        dip = apparent_dip(pp.motion, pp.downslope)
        k = 0.3 * half
        ax.plot(sign * np.r_[-k, k], z0 - np.tan(np.radians(dip)) * np.r_[-k, k],
                ls="--", color=pp.slope_colour, lw=1.4, zorder=2,
                label=f"ground slope {dip:.0f}° down (50 m plane)")  # fmt: skip
    ax.scatter(sign * prof.along * scale, z0 + prof.up * scale, c=prof.t,
               s=30 if small else 55, marker=pp.marker, edgecolors="k",
               linewidths=0.6, zorder=4, **style)  # fmt: skip
    # no alpha: it would fill the open (older than the colours) points black
    rtk = getattr(pp, "rtk", None)
    if rtk is not None and np.size(pp.rtk_since):  # --rtk: the RTK tail on top
        from gps_plot.polar import _rtk_dots

        ux, uy = np.sin(np.radians(pp.motion)), np.cos(np.radians(pp.motion))
        r_along = rtk.d[0] * uy + rtk.d[1] * ux
        p_along = rtk.path[0] * uy + rtk.path[1] * ux
        _rtk_dots(ax, sign * r_along * scale, z0 + rtk.d[2] * scale, pp.rtk_since,
                  style, pp.marker, 4 if small else 8,
                  (sign * p_along * scale, z0 + rtk.path[2] * scale),
                  pp.rtk_colour)  # fmt: skip
    wt, wa, wu = weekly_medians(prof.t, prof.along, prof.up)
    if wt.size:  # the path without the day-to-day noise
        ax.plot(np.r_[0, sign * wa * scale], np.r_[z0, z0 + wu * scale], color="k",
                lw=0.8, zorder=5)  # fmt: skip
        ax.scatter(sign * wa * scale, z0 + wu * scale, c=wt, s=40 if small else 60,
                   marker="D", edgecolors="k", linewidths=1.0, zorder=6,
                   label=rf"weekly median ($\geq${MIN_PER_BIN} days)", **style)  # fmt: skip
    # the 14-day rate: ARROW_DAYS of it from the newest position, along the
    # motion and Up (BGÓ 2026-11-11; no ellipse)
    from gps_plot.polar import rate_step, rate_step_label

    step = rate_step(pp)
    if step is not None and prof.t.size:
        u_x, u_y = np.sin(np.radians(pp.motion)), np.cos(np.radians(pp.motion))
        newest = prof.t >= prof.t.max() - 2.0
        a0, h0 = float(np.median(prof.along[newest])), float(np.median(prof.up[newest]))
        da, dh = step[0] * u_y + step[1] * u_x, step[2]
        ax.annotate("", xy=(sign * (a0 + da) * scale, z0 + (h0 + dh) * scale),
                    xytext=(sign * a0 * scale, z0 + h0 * scale), zorder=7,
                    arrowprops={"arrowstyle": "-|>", "color": pp.colour, "lw": 1.8})  # fmt: skip
        ax.plot(
            [],
            [],
            color=pp.colour,
            lw=1.8,
            label=rate_step_label(pp) + (up_text(pp) if pp.rate_from != "rtk" else ""),
        )
    pl = prof.plunge
    if pl is not None:
        length = 1.15 * float(np.max(np.abs(prof.along))) * scale
        for deg, lw in (
            (pl.deg, 2.2),
            (pl.deg - pl.sigma, 0.8),
            (pl.deg + pl.sigma, 0.8),
        ):
            r = np.radians(deg)
            ax.plot([0, sign * length * np.cos(r)], [z0, z0 - length * np.sin(r)],
                    color=pp.colour, lw=lw, zorder=6, alpha=1.0 if lw > 1 else 0.6,
                    label=(f"motion {'plunges' if pl.deg >= 0 else 'rises'} "
                           f"{abs(pl.deg):.0f} ± {pl.sigma:.0f}° ({pl.n} days)")
                    if lw > 1 else None)  # fmt: skip
    _total_bar(ax, prof, sign, scale, half, small)
    ax.plot(0, z0, marker="^", ms=11 if small else 14, color="w", mec="k", mew=1.2,
            zorder=7)  # fmt: skip
    ax.annotate(pp.rover, (0, z0), xytext=(8, 10), textcoords="offset points",
                fontweight="bold", zorder=8, fontsize="small" if small else "medium",
                bbox={"boxstyle": "round", "fc": "w", "alpha": 0.85})  # fmt: skip
    xb, yb = 0.6 * half, lo + 0.08 * (hi - lo)
    ax.plot([xb, xb + bar_mm * scale], [yb, yb], "o-", color="k", lw=2.0, ms=3,
            zorder=6)  # fmt: skip
    ax.annotate(f"{bar_mm:g} mm (dots)", (xb, yb), xytext=(0, 6),
                textcoords="offset points", fontsize="x-small", zorder=6,
                bbox={"boxstyle": "round", "fc": "w", "alpha": 0.85})  # fmt: skip
    ahead, back = compass(pp.motion), compass(pp.motion + 180.0)
    left, right = (ahead, back) if prof.west else (back, ahead)
    what = f"along {pp.motion:.0f}° [m]"
    gap = r" \quad " if _usetex() else "   "
    ax.set_xlim(-half, half)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    size = "small" if small else "medium"
    ax.set_xlabel(rf"{left} $\leftarrow${gap}{what}{gap}$\rightarrow$ {right}",
                  fontsize=size)  # fmt: skip
    ax.set_ylabel("height [m]" if np.isfinite(prof.z).any()
                  else "height from the rover [m]", fontsize=size)  # fmt: skip
    if small:
        ax.tick_params(labelsize="x-small")
    ax.set_title(f"{pp.rover}: ground along its motion ({pp.motion:.0f}°)",
                 fontsize=size)  # fmt: skip
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="upper right", fontsize="xx-small" if small else "x-small",
                  framealpha=0.85, handlelength=1.5, borderpad=0.3,
                  labelspacing=0.25, scatterpoints=1)  # fmt: skip


def end_along_up(prof: Profile) -> tuple[float, float] | None:
    """The robust (along, Up) level [mm] of the newest days (as the map's T
    line: :data:`gps_plot.polar.END_DAYS`); None without one."""
    from gps_analysis.direction import end_level

    from gps_plot.polar import END_DAYS

    if not prof.t.size:
        return None
    newest = prof.t >= prof.t.max() - END_DAYS
    y = np.vstack([prof.along[newest], prof.up[newest]])
    sig = np.vstack([prof.s_along[newest], prof.s_up[newest]])
    sig = np.where(np.isfinite(sig) & (sig > 0), sig, 1.0)
    lvl, _, n = end_level(y, sig)
    if n == 0 or not np.isfinite(lvl).all():
        return None
    return float(lvl[0]), float(lvl[1])


def _day_text(days_1970: float) -> str:
    """``23 Sep`` of an epoch in days since 1970."""
    import datetime as dt

    return (dt.datetime(1970, 1, 1) + dt.timedelta(days=days_1970)).strftime("%d %b")


def total_movement(prof: Profile) -> tuple[float, float, float, str, str, bool] | None:
    """The movement the T bar shows: (start Up, end Up, total [mm], since
    when, colour, above its test), Up relative to the rover. The RTK
    figure (``--rate rtk`` with an episode onset): the RTK movement since that onset, the
    black arrow's line over its window, ending at the newest RTK path
    point, in black (BGÓ 2026-11-13); otherwise the newest daily level
    since the figure's onset, in the pair's colour."""
    pp = prof.pp
    rtk: Any = getattr(pp, "rtk", None)
    since = getattr(rtk, "since", None) if pp.rate_from == "rtk" else None
    if since is not None and rtk.since_t is not None and rtk.path is not None:
        path = np.asarray(rtk.path, dtype=float)
        ok = np.isfinite(path).all(axis=0)
        if ok.any():
            step = np.array([since.north, since.east, since.up]) * since.window
            end = float(path[2][ok][-1])
            return (end - float(step[2]), end, float(np.linalg.norm(step)),
                    _day_text(rtk.since_t), "k", bool(since.significant))  # fmt: skip
    end_lvl = end_along_up(prof)
    if end_lvl is None:
        return None
    along, up = end_lvl
    return 0.0, up, float(np.hypot(along, up)), _day_text(pp.t_on), pp.colour, True


def _total_bar(ax: Any, prof: Profile, sign: float, scale: float, half: float,
               small: bool) -> None:  # fmt: skip
    """The movement (:func:`total_movement`) as a vertical T bar beside the
    rover (BGÓ 2026-11-13): its Up at the dots' scale, capped at both ends;
    beside it since when, the total and the Up of it [mm] (and "below its
    test" when the RTK rate since then is)."""
    got = total_movement(prof)
    if got is None:
        return
    u0, u1, total, when, colour, shown = got
    z0 = prof.z0
    x = -sign * 0.2 * half  # behind the rover, clear of its points and name
    cap = 0.02 * half
    y0, y1 = z0 + u0 * scale, z0 + u1 * scale
    kw = {"color": colour, "lw": 2.2, "zorder": 6.5, "solid_capstyle": "butt"}
    ax.plot([x, x], [y0, y1], **kw)
    for y in (y0, y1):
        ax.plot([x - cap, x + cap], [y, y], **kw)
    text = f"since {when}:\n{total:.0f} mm total\nUp {u1 - u0:+.0f} mm"
    if not shown:  # as the black arrow's legend
        text += "\n(below its test)"
    ax.annotate(text,
                (x, 0.5 * (y0 + y1)),
                xytext=(-6 if sign > 0 else 6, 0), textcoords="offset points",
                ha="right" if sign > 0 else "left", va="center",
                fontsize="x-small" if small else "small", color=colour,
                zorder=8, bbox={"boxstyle": "round", "fc": "w", "alpha": 0.8,
                                "ec": "none"})  # fmt: skip


def panel(fig: Any, spec: Any, pairs: Sequence[Any], style: dict[str, Any], dem: Any,
          half: float = PROFILE_HALF_M, small: bool = True,
          hspace: float = 0.35) -> tuple[list[Any], str]:  # fmt: skip
    """The profile panel in ``spec`` (a subplot spec): one sub-panel per pair
    with a motion, stacked, their heights following their ranges so all share
    one m-per-m scale. Returns the axes and the note's part."""
    with_motion = [pp for pp in pairs if pp.motion is not None]
    if not with_motion:
        ax = fig.add_subplot(spec)
        ax.text(0.5, 0.5, "no motion direction", transform=ax.transAxes, ha="center")
        ax.set_axis_off()
        return [ax], ""
    scale = scale_for(with_motion, half)
    profiles = [prepare(pp, dem, scale, half) for pp in with_motion]
    from gps_plot.polar import _nice

    big = max(float(np.nanmax(np.hypot(p.along, p.up))) if p.t.size else 1.0
              for p in profiles)  # fmt: skip
    bar_mm = _nice(big / 4)
    sub = spec.subgridspec(len(profiles), 1, hspace=hspace,
                           height_ratios=[p.hi - p.lo for p in profiles])  # fmt: skip
    axes = []
    for k, prof in enumerate(profiles):
        axes.append(fig.add_subplot(sub[k]))
        draw(axes[-1], prof, style, scale, bar_mm, half, small)
    src = f"{dem.source} DEM" if dem is not None else "no DEM"
    return axes, (
        f"profile: ground along each motion ({src}), points {scale:g} m per mm; "
        "plunge: a line through the rover fitted to the daily points with their "
        "sigmas, errors in both directions (1-sigma wedge); diamonds: weekly medians"
    )


def spans(pairs: Sequence[Any], dem: Any, half: float = PROFILE_HALF_M) -> list[float]:
    """The height ranges [m] of the pairs' sub-panels (to size a figure)."""
    with_motion = [pp for pp in pairs if pp.motion is not None]
    if not with_motion:
        return [half]
    scale = scale_for(with_motion, half)
    return [(p := prepare(pp, dem, scale, half)).hi - p.lo for pp in with_motion]
