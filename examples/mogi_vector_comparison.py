"""Mogi model vs observed GNSS displacement-vector comparison (PyGMT).

Reproduces the Svartsengi *inflation08* validation figure: observed GNSS
horizontal displacement over an inflation window compared against two Mogi
point-source models — our GNSS-only fit and Vincent's operational model —
overlaid on a PyGMT map of the Reykjanes peninsula. This is the visual
companion to the ``gps_api`` real-data reconciliation (Pearson r = 0.993);
the near-identical red/blue fields despite different (depth, ΔV) illustrate
the classic depth-volume trade-off.

Data provenance
---------------
Reads the ``gps_api`` real-data validation fixture (gitignored; regenerate with
``gps-api-validate-deformation fetch`` — see ``gps_api/src/gps_api/validation``):
- ``reconciliation.json`` — our fitted Mogi source + window + station list
- ``manifest.json``        — Vincent's fixed source geometry + final ΔV
- ``neu/<STA>.NEU``        — observed displacement series (window end − start)
Station coordinates come from ``gps_parser``; model predictions from
``gps_analysis.mogi_forward``.

Requirements
------------
The ``maps`` extra (PyGMT) + a GMT ≥ 6 C library. Run with the GMT env set::

    export GMT_LIBRARY_PATH=$HOME/git/gmt/install/lib   # from-source build
    uv run --extra maps python examples/mogi_vector_comparison.py \\
        --fixture ~/work/projects/gpslibrary/gps_api/tests/fixtures/realdata \\
        --out examples/mogi_vector_comparison.png
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np

DEFAULT_FIXTURE = (
    Path.home() / "work/projects/gpslibrary/gps_api/tests/fixtures/realdata"
)


def _neu_window(path: Path, start: dt.date, end: dt.date) -> tuple[float, float] | None:
    """Observed (dE, dN) [mm] over [start, end] from a ``.NEU`` series."""
    if not path.exists():
        return None
    dates, dN, dE = [], [], []
    for line in path.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        p = line.split()
        try:
            dates.append(dt.datetime.strptime(p[0], "%Y/%m/%d").date())
        except ValueError:
            continue
        dN.append(float(p[2]))
        dE.append(float(p[4]))
    if not dates:
        return None
    i0 = int(np.argmin([abs((d - start).days) for d in dates]))
    i1 = int(np.argmin([abs((d - end).days) for d in dates]))
    return dE[i1] - dE[i0], dN[i1] - dN[i0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    ap.add_argument("--out", type=Path, default=Path("mogi_vector_comparison.png"))
    args = ap.parse_args()

    import pygmt  # noqa: PLC0415 — optional 'maps' extra; import lazily
    from gps_analysis import MogiSource, local_coordinates, mogi_forward  # noqa: PLC0415
    from gps_parser import ConfigParser  # noqa: PLC0415

    rec = json.loads((args.fixture / "reconciliation.json").read_text())
    man = json.loads((args.fixture / "manifest.json").read_text())
    start = dt.date.fromisoformat(rec["window_start"])
    end = dt.date.fromisoformat(rec["window_end"])
    ours = dict(
        lon=rec["source_lon_mean"],
        lat=rec["source_lat_mean"],
        depth=rec["depth_mean_km"] * 1e3,
        dv=rec["final_ours_m3"],
    )
    v = man["vincent_source"]
    vinc = dict(lon=v["lon"], lat=v["lat"], depth=v["depth_m"], dv=v["dv_m3_final"])

    cp = ConfigParser()
    lon, lat, names = [], [], []
    for s in rec["stations_used"]:
        try:
            info = cp.getStationInfo(s)["station"]
        except Exception:  # noqa: BLE001 — station absent from config
            continue
        lon.append(float(info["longitude"]))
        lat.append(float(info["latitude"]))
        names.append(s)
    lon, lat = np.array(lon), np.array(lat)

    def model(src: dict) -> tuple[np.ndarray, np.ndarray]:
        e, n = local_coordinates(lon, lat, src["lon"], src["lat"])
        enu = mogi_forward(
            e, n, MogiSource(x=0.0, y=0.0, depth=src["depth"], dv=src["dv"])
        )
        return enu[0] * 1e3, enu[1] * 1e3  # E, N in mm

    obsE = np.full(len(names), np.nan)
    obsN = np.full(len(names), np.nan)
    for i, s in enumerate(names):
        d = _neu_window(args.fixture / "neu" / f"{s}.NEU", start, end)
        if d:
            obsE[i], obsN[i] = d
    ourE, ourN = model(ours)
    vinE, vinN = model(vinc)

    region = [lon.min() - 0.06, lon.max() + 0.06, lat.min() - 0.03, lat.max() + 0.03]
    fig = pygmt.Figure()
    fig.basemap(
        region=region,
        projection="M18c",
        frame=[
            "af",
            f"+tSvartsengi {rec['cycle']} ({rec['window_start']} to "
            f"{rec['window_end']}): observed GNSS vs Mogi models",
        ],
    )
    fig.coast(
        land="240/240/235",
        water="200/220/240",
        shorelines="0.4p,gray50",
        resolution="f",
    )

    scale = 0.012  # cm(map) per mm(displacement)

    def velo(ve, vn, mask, color):
        z = np.zeros(int(mask.sum()))
        data = np.column_stack([lon[mask], lat[mask], ve[mask], vn[mask], z, z, z])
        fig.velo(
            data=data,
            spec=f"e{scale}/0.39/0",
            pen=f"1.4p,{color}",
            vector=f"0.4c+p1.4p,{color}+e+g{color}",
        )

    allm = np.ones(len(names), dtype=bool)
    velo(vinE, vinN, allm, "blue")
    velo(ourE, ourN, allm, "red")
    velo(obsE, obsN, np.isfinite(obsE), "black")

    fig.plot(x=lon, y=lat, style="c0.12c", fill="white", pen="0.6p,black")
    fig.plot(
        x=[vinc["lon"]], y=[vinc["lat"]], style="a0.5c", fill="blue", pen="0.6p,black"
    )
    fig.plot(
        x=[ours["lon"]], y=[ours["lat"]], style="a0.5c", fill="red", pen="0.6p,black"
    )

    rlon, rlat = region[0] + 0.02, region[2] + 0.02
    fig.velo(
        data=np.array([[rlon, rlat + 0.015, 50.0, 0.0, 0, 0, 0]]),
        spec=f"e{scale}/0.39/0",
        pen="1.4p,black",
        vector="0.4c+p1.4p,black+e+gblack",
    )
    fig.text(x=rlon, y=rlat + 0.028, text="50 mm", font="9p,Helvetica", justify="LM")
    fig.text(
        x=rlon,
        y=rlat + 0.004,
        text="black=observed  red=ours(3.7km)  blue=Vincent(4km)",
        font="9p,Helvetica-Bold",
        justify="LM",
    )

    fig.savefig(str(args.out), dpi=150)
    print(
        f"saved {args.out}  ({len(names)} stations, "
        f"peak |obs| = {np.nanmax(np.hypot(obsE, obsN)):.0f} mm)"
    )


if __name__ == "__main__":
    main()
