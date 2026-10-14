"""``gpsplot stack``: pattern expansion, colours, offsets, labels, failures."""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib.figure import Figure

from gps_plot import compose
from gps_plot import stack as stk
from gps_plot.series import Series

STATIONS = ["SEY1", "SEY2", "SEY3", "SEYD", "HOFN", "HVOL", "HVEL"]


# ---------------------------------------------------------------------------
# series list
# ---------------------------------------------------------------------------


def test_split_spec_keeps_character_ranges():
    assert stk.split_spec("SEY?-SEYD") == ("SEY?", "SEYD")
    assert stk.split_spec("SEY[1-3]-SEYD") == ("SEY[1-3]", "SEYD")
    assert stk.split_spec("SEY[1-3]") is None
    assert stk.split_spec("HOFN") is None


def test_expand_baseline_pattern_drops_the_self_pair():
    assert stk.expand_series(["SEY?-SEYD"], STATIONS) == [
        "SEY1-SEYD",
        "SEY2-SEYD",
        "SEY3-SEYD",
    ]


def test_expand_mixes_stations_ranges_and_dedups():
    got = stk.expand_series(["SEY[1-2]-SEYD", "hv*", "SEY1-SEYD", "HOFN"], STATIONS)
    assert got == ["SEY1-SEYD", "SEY2-SEYD", "HVEL", "HVOL", "HOFN"]


def test_expand_refuses_an_empty_pattern():
    with pytest.raises(ValueError, match="matches no station"):
        stk.expand_series(["XX?"], STATIONS)
    with pytest.raises(ValueError, match="matches no station"):
        stk.expand_series(["SEYD-SEY[D]"], STATIONS)  # only the self-pair


def test_file_stem_has_no_hyphen_before_the_ref():
    assert stk.file_stem(["SEY1-SEYD", "SEY2-SEYD"]) == "stack_SEY1_SEYD_2"
    assert "-" not in stk.file_stem(["SEY1-SEYD"])


# ---------------------------------------------------------------------------
# composition
# ---------------------------------------------------------------------------


def test_colours_follow_the_label_not_the_position():
    a = compose.series_colors(["SEY1-SEYD", "HOFN"])
    b = compose.series_colors(["HOFN", "SEY1-SEYD", "RHOF"])
    assert a[0] == b[1] and a[1] == b[0]


def test_colours_are_distinct_within_a_figure():
    labels = [f"SEY{i}-SEYD" for i in range(1, 10)] + ["SEYD", "HOFN", "RHOF"]
    got = compose.series_colors(labels)
    assert len(set(got)) == len(compose.STACK_PALETTE) == len(labels)


def _series(label, n=30, level=0.0):
    t = np.linspace(2025.0, 2025.5, n)
    y = np.vstack([np.full(n, level), np.full(n, level + 1), np.full(n, level + 2)])
    return Series(label, t, y, np.ones((3, n)))


def test_stack_figure_offsets_series_k_by_k_times_offset():
    from gps_plot.timesmatplt import StationTitle

    series = [_series("A"), _series("B"), _series("C")]
    title = StationTitle("x", "y", "z", "black")
    fig = compose.stack_figure(series, title=title, offset=10.0, fig=Figure())
    for comp, ax in enumerate(fig.axes[:3]):
        drawn = [ln for ln in ax.lines if ln.get_marker() == "o"]
        assert len(drawn) == 3
        for k, ln in enumerate(drawn):
            np.testing.assert_allclose(ln.get_ydata(), comp + 10.0 * k)
    legend = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert legend == ["A", "B (+10 mm)", "C (+20 mm)"]


def test_stack_figure_without_offset_draws_the_series_as_fetched():
    from gps_plot.timesmatplt import StationTitle

    s = _series("A", level=3.5)
    fig = compose.stack_figure(
        [s], title=StationTitle("x", "y", "z", "black"), fig=Figure()
    )
    drawn = [ln for ln in fig.axes[2].lines if ln.get_marker() == "o"]
    np.testing.assert_array_equal(drawn[0].get_ydata(), s.y[2])


# ---------------------------------------------------------------------------
# run: failures and the not-detrended label
# ---------------------------------------------------------------------------


class _Config:
    def getPostProcessDir(self, option):
        return "/tmp"

    def getStationInfo(self):
        return STATIONS


def _run(argv, monkeypatch, fetch, tmp_path):
    from gps_plot.sources import gps as source

    monkeypatch.setattr(source, "fetch", fetch)
    monkeypatch.setattr(source, "rezero_detrended", lambda spec, req, s: s)
    parser = stk.build_parser(config=_Config())
    args = parser.parse_args(argv + ["--save", "png", "-d", str(tmp_path)])
    return stk.run(args, parser, _Config())


def test_run_lists_dropped_series_and_fails_when_none_plot(
    monkeypatch, tmp_path, capsys
):
    def fetch(spec, req, opts=None):
        raise ValueError("no data")

    assert _run(["SEY?-SEYD"], monkeypatch, fetch, tmp_path) == 1
    err = capsys.readouterr().err
    assert err.count("dropped") == 3 and "nothing plotted" in err
    assert not list(tmp_path.iterdir())


def test_run_labels_a_series_that_was_not_detrended(monkeypatch, tmp_path):
    import matplotlib.figure

    seen = {}

    def fetch(spec, req, opts=None):
        s = _series(spec)
        return Series(s.label, s.t, s.y, s.sigma, detrend_applied=spec != "HOFN")

    def capture(self, *a, **k):
        seen["legend"] = [t.get_text() for t in self.axes[0].get_legend().get_texts()]

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", capture)
    assert _run(["SEY1", "HOFN", "--ref", "detrend"], monkeypatch, fetch, tmp_path) == 0
    assert seen["legend"] == ["SEY1", "HOFN (not detrended)"]


def test_run_refuses_detrend_terms_without_detrend(monkeypatch, tmp_path, capsys):
    with pytest.raises(SystemExit):
        _run(["HOFN", "--detrend-terms", "periodic"], monkeypatch, None, tmp_path)
    assert "--detrend-terms needs --ref detrend" in capsys.readouterr().err


@pytest.mark.parametrize("bad", [["--ref", "all"], ["--session", "both"]])
def test_stack_has_no_fan_out_options(bad, capsys):
    parser = stk.build_parser(config=_Config())
    with pytest.raises(SystemExit):
        parser.parse_args(["HOFN", *bad])
