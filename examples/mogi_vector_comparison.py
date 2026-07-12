"""Mogi model vs observed GNSS displacement-vector comparison (PyGMT).

Reproduces the Svartsengi *inflation08* validation figure: observed GNSS
horizontal displacement over an inflation window compared against two Mogi
point-source models — our GNSS-only fit and Vincent's operational model —
overlaid on a PyGMT map of the Reykjanes peninsula. This is the visual
companion to the ``gps_api`` real-data reconciliation (Pearson r = 0.993);
the near-identical red/blue fields despite different (depth, ΔV) illustrate
the classic depth-volume trade-off.

The map itself is :func:`gps_plot.maps.deformation_vectors` — this script
only assembles the inputs: observed (dE, dN) from the fixture ``.NEU``
series, and the two model fields via :func:`gps_plot.maps.mogi_model_field`
(``gps_analysis.mogi_forward`` under the hood).

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

    from gps_parser import ConfigParser  # noqa: PLC0415 — dev-group sibling

    from gps_plot.maps import (  # noqa: PLC0415
        StationCoordinate,
        deformation_vectors,
        mogi_model_field,
    )

    rec = json.loads((args.fixture / "reconciliation.json").read_text())
    man = json.loads((args.fixture / "manifest.json").read_text())
    start = dt.date.fromisoformat(rec["window_start"])
    end = dt.date.fromisoformat(rec["window_end"])
    vinc = man["vincent_source"]

    cp = ConfigParser()
    stations: list[StationCoordinate] = []
    for s in rec["stations_used"]:
        try:
            info = cp.getStationInfo(s)["station"]
        except Exception:  # noqa: BLE001 — station absent from config
            continue
        stations.append(
            StationCoordinate(
                marker=s, lon=float(info["longitude"]), lat=float(info["latitude"])
            )
        )
    lon = np.array([c.lon for c in stations])
    lat = np.array([c.lat for c in stations])

    obsE = np.full(len(stations), np.nan)
    obsN = np.full(len(stations), np.nan)
    for i, c in enumerate(stations):
        d = _neu_window(args.fixture / "neu" / f"{c.marker}.NEU", start, end)
        if d:
            obsE[i], obsN[i] = d

    models = [
        mogi_model_field(
            lon,
            lat,
            source_lon=vinc["lon"],
            source_lat=vinc["lat"],
            depth=vinc["depth_m"],
            dv=vinc["dv_m3_final"],
            name=f"Vincent ({vinc['depth_m'] / 1e3:g} km)",
            color="blue",
        ),
        mogi_model_field(
            lon,
            lat,
            source_lon=rec["source_lon_mean"],
            source_lat=rec["source_lat_mean"],
            depth=rec["depth_mean_km"] * 1e3,
            dv=rec["final_ours_m3"],
            name=f"ours ({rec['depth_mean_km']:.1f} km)",
            color="red",
        ),
    ]

    region = (
        float(lon.min()) - 0.06,
        float(lon.max()) + 0.06,
        float(lat.min()) - 0.03,
        float(lat.max()) + 0.03,
    )
    deformation_vectors(
        stations,
        obsE,
        obsN,
        models,
        region=region,
        projection="M18c",
        resolution="f",
        land="240/240/235",
        water="200/220/240",
        shorelines="0.4p,gray50",
        title=(
            f"Svartsengi {rec['cycle']} ({rec['window_start']} to "
            f"{rec['window_end']}): observed GNSS vs Mogi models"
        ),
        scale=0.012,  # cm(map) per mm(displacement)
        scale_ref=50.0,
        outfile=args.out,
    )
    print(
        f"saved {args.out}  ({len(stations)} stations, "
        f"peak |obs| = {np.nanmax(np.hypot(obsE, obsN)):.0f} mm)"
    )


if __name__ == "__main__":
    main()
