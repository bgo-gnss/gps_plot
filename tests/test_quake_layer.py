"""Catalogue quakes + cumulative moment on a panel (gpsplot ts --quakes, Q-T6)."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from geo_dataread.seismic.selection import _empty

from gps_plot.layers import quakes as ql
from gps_plot.plot_gps_timeseries import build_parser

SEAM = pd.Timestamp("2026-02-01", tz="UTC")


def frame(rows: list[tuple[str, float, str, str, float]]) -> pd.DataFrame:
    """(time, magnitude, magnitude_type, evaluation_mode, moment_nm)."""
    return pd.DataFrame(
        {
            "magnitude": [r[1] for r in rows],
            "magnitude_type": [r[2] for r in rows],
            "evaluation_mode": [r[3] for r in rows],
            "moment_nm": [r[4] for r in rows],
        },
        index=pd.DatetimeIndex([pd.Timestamp(r[0], tz="UTC") for r in rows]),
    )


def report(*sources: tuple[str, str, str]) -> Any:
    parts = tuple(
        SimpleNamespace(
            source=s, start=pd.Timestamp(a, tz="UTC"), end=pd.Timestamp(b, tz="UTC")
        )
        for s, a, b in sources
    )
    return SimpleNamespace(parts=parts, summary=lambda: "parts")


ROWS = [
    ("2025-06-01", 1.0, "Mlw", "manual", 1e12),
    ("2025-09-01", 4.2, "Mlw", "manual", 1e15),
    ("2025-12-01", 0.5, "aut.mag", "automatic", np.nan),
    ("2026-03-01", 5.3, "Mpgv_w", "manual", 4e16),
    ("2026-04-01", 3.5, "Mpgv_w", "automatic", 2e14),
]
BOTH = report(
    ("sil-files", "2025-01-01", "2026-02-01"),
    ("quakes-seiscomp", "2026-02-01", "2026-06-01"),
)


def panel() -> Any:
    fig, ax = plt.subplots()
    ax.plot([dt.datetime(2025, 1, 1), dt.datetime(2026, 6, 1)], [0, 10], "o")
    return fig, ax


def test_panel_draws_classes_moment_and_seam() -> None:
    fig, ax = panel()
    xlim = ax.get_xlim()
    info = ql.add_quake_panel(ax, frame(ROWS), BOTH, "bb-caldera")
    assert ax.get_xlim() == xlim  # the data panel keeps its limits
    assert info.n == 5 and info.n_automatic == 2
    assert info.moment_nm == pytest.approx(1e12 + 1e15 + 4e16 + 2e14)
    assert info.moment_automatic_nm == pytest.approx(2e14)
    assert info.seam == dt.datetime(2026, 2, 1)
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert any("estimated from SeisComP" in s for s in labels)
    assert any("automatic" in s for s in labels)
    assert "bb-caldera: n = 5, 2 automatic" in ax.get_legend().get_title().get_text()
    # the seam line on the data panel
    seam_x = float(mdates.date2num(dt.datetime(2026, 2, 1)))
    vlines = [ln.get_xdata() for ln in ax.get_lines() if ln.get_linestyle() == "--"]
    assert any(np.allclose(np.asarray(v, dtype=float), seam_x) for v in vlines)
    # magnitude axis, moment axis (offset spine), step-line axis
    assert len(fig.axes) == 4
    assert fig.subplotpars.right == pytest.approx(ql.QUAKES_FIG_RIGHT)
    # value written from 5
    texts = [c.get_text() for a in fig.axes for c in a.texts]
    assert "5.3" in texts and "4.2" not in texts
    plt.close(fig)


def test_no_seam_and_no_moment() -> None:
    fig, ax = panel()
    sil = report(("sil-files", "2025-01-01", "2026-01-01"))
    info = ql.add_quake_panel(ax, frame(ROWS[:3]), sil, "x_y", moment=False)
    assert info.seam is None and info.moment_nm == 0.0
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert not any("moment" in s for s in labels)
    assert r"x\_y" in ax.get_legend().get_title().get_text()  # usetex-safe
    plt.close(fig)


def test_empty_catalogue() -> None:
    fig, ax = panel()
    info = ql.add_quake_panel(ax, frame([]), BOTH, "none")
    assert info.n == 0 and info.moment_nm == 0.0
    plt.close(fig)


def test_request_from_cli_args() -> None:
    p = build_parser()
    a = vars(p.parse_args(["VONC", "--quakes", "--quake-min-q", "40", "--no-moment"]))
    req = ql.request_from_args(a)
    assert req is not None and req.region is None and not req.moment
    assert req.options["quake_min_q"] == 40.0
    assert not any(k.startswith("quake_") or k == "no_moment" for k in a)
    a = vars(p.parse_args(["VONC", "--quakes", "bardarbunga-caldera"]))
    assert ql.request_from_args(a).region == "bardarbunga-caldera"  # type: ignore[union-attr]
    assert ql.request_from_args(vars(p.parse_args(["VONC"]))) is None
    # --quake (marks) and --quakes (catalogue) do not collide
    a = vars(p.parse_args(["VONC", "--quake", "2026-10-19,64.6,-17.5,5.2", "--quakes"]))
    assert a["quake"] == ["2026-10-19,64.6,-17.5,5.2"] and a["quake_catalogue"] == ""


REGIONS = """
regions:
  bb-caldera: {bbox: [-17.65, 64.59, -17.25, 64.72], area: bardarbunga, default: true}
  bb-map: {bbox: [-18.65, 64.18, -16.8, 65.16], area: bardarbunga}
  askja: {bbox: [-17.0, 64.9, -16.5, 65.2], area: askja, default: true}
"""


@pytest.fixture
def regions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "quake_regions.yaml"
    path.write_text(REGIONS)
    import geo_dataread.events as ev

    monkeypatch.setattr(
        ev,
        "read_station_areas",
        lambda path=None: {
            "bardarbunga": frozenset({"VONC", "KISA"}),
            "askja": frozenset({"OLAC", "KISA"}),
            "nowhere": frozenset({"XXXX"}),
        },
    )
    return path


def test_area_region(regions: Path) -> None:
    assert ql.area_region("VONC", regions) == "bb-caldera"
    with pytest.raises(ValueError, match="different quake regions"):
        ql.area_region("KISA", regions)
    with pytest.raises(ValueError, match="no default quake region"):
        ql.area_region("XXXX", regions)


def test_load_quakes_uses_the_area_region(regions: Path) -> None:
    calls: list[Any] = []

    def sil(a: Any, b: Any, **kw: Any) -> pd.DataFrame:
        calls.append(("sil", a, b, kw["min_quality"]))
        return _empty()

    def seiscomp(a: Any, b: Any, **kw: Any) -> pd.DataFrame:
        calls.append(("seiscomp", a, b, kw["bbox"]))
        return _empty()

    p = build_parser()
    a = vars(p.parse_args(
        ["VONC", "--quakes", "--quake-min-q", "40", "--quake-regions", str(regions)]
    ))  # fmt: skip
    req = ql.request_from_args(a)
    assert req is not None
    df, rep, name = ql.load_quakes(
        "VONC", req, dt.datetime(2025, 1, 1), dt.datetime(2026, 6, 1),
        readers={"sil": sil, "seiscomp": seiscomp},
    )  # fmt: skip
    assert name == "bb-caldera" and len(df) == 0
    assert [c[0] for c in calls] == ["sil", "seiscomp"]
    assert calls[0][3] == 40.0 and calls[1][3] == (-17.65, 64.59, -17.25, 64.72)
    assert ql._seam(rep) == dt.datetime(2026, 2, 1)
