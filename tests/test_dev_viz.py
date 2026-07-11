"""Smoke tests for the gps_analysis dev-viz surface (gps_plot.dev_viz).

Fast end-to-end checks: synthetic data → gps_analysis chain → figure on
disk, using the Agg backend and a short MCMC chain. Numerical correctness
of the underlying math is tested in gps_analysis itself.
"""

import matplotlib

matplotlib.use("Agg")  # before any pyplot import — headless CI/dev

import numpy as np
import pytest

from gps_plot.dev_viz import (
    ComponentAnalysis,
    analyze_station,
    load_neu,
    main,
    render_dev_viz,
    synthetic_station,
)

# Short-chain MCMC settings: 16 temperatures x T_RUNS < N_RUNS must hold.
QUICK = {"n_runs": 900, "t_runs": 50, "seed": 1}


@pytest.fixture(scope="module")
def series():
    return synthetic_station(seed=1, n_days=240, wn_amp=2.0)


def test_synthetic_station_shape(series):
    assert series.y.shape == (3, 240)
    assert series.sigma is not None and series.sigma.shape == series.y.shape
    assert np.all(np.diff(series.t) > 0)
    assert series.true_breaks is not None and len(series.true_breaks) == 3


def test_analyze_without_detection_uses_truth(series):
    analysis = analyze_station(series, "up", detect=False)
    assert isinstance(analysis, ComponentAnalysis)
    assert analysis.break_result is None
    assert analysis.break_params == series.true_breaks[2]
    # Residuals are zero-mean-ish once the lineperiodic trend is removed.
    assert abs(float(np.mean(analysis.residuals))) < 5.0
    # Velocity carries the horizontal products (north+east labelled).
    assert analysis.velocity.magnitude is not None
    assert analysis.velocity.azimuth is not None
    # Default break input: zero-referenced raw series, starting near zero.
    assert analysis.break_input == "raw"
    assert analysis.break_series.shape == series.t.shape
    assert abs(float(np.mean(analysis.break_series[:10]))) < 5.0


def test_break_input_modes(series):
    for mode in ("raw", "seasonal_removed", "residuals"):
        analysis = analyze_station(series, "east", detect=False, break_input=mode)
        assert analysis.break_input == mode
        assert analysis.break_series.shape == series.t.shape
    with pytest.raises(ValueError, match="break_input"):
        analyze_station(series, "east", detect=False, break_input="bogus")


def test_render_dev_viz_end_to_end_with_mcmc(series, tmp_path):
    out = tmp_path / "devviz.png"
    fig, analysis = render_dev_viz(series, "east", out_path=out, **QUICK)
    try:
        assert out.exists() and out.stat().st_size > 0
        assert analysis.break_result is not None
        assert analysis.break_result.model == "BPD1"
        assert analysis.break_params is not None
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)


def test_cli_synthetic_no_detect(tmp_path, capsys):
    out = tmp_path / "cli.png"
    rc = main(["--synthetic", "--no-detect", "--component", "north", "--out", str(out)])
    assert rc == 0
    assert out.exists() and out.stat().st_size > 0
    stdout = capsys.readouterr().out
    assert "mm/yr" in stdout and "break" in stdout


def test_load_neu_roundtrip(tmp_path):
    neu = tmp_path / "TEST.NEU"
    neu.write_text(
        "printing header and file\n"
        "# date time dN DN dE DE dU DU\n"
        "2024/01/01 12:00:00 1.0 2.0 3.0 4.0 5.0 6.0\n"
        "2024/01/02 12:00:00 1.5 2.0 3.5 4.0 5.5 6.0\n"
    )
    series = load_neu(neu)
    assert series.station == "TEST"
    assert series.y.shape == (2 + 1, 2)  # 3 components x 2 epochs
    np.testing.assert_allclose(series.y[:, 0], [1.0, 3.0, 5.0])
    np.testing.assert_allclose(series.sigma[:, 1], [2.0, 4.0, 6.0])
    assert 2023.99 < series.t[0] < series.t[1] < 2024.01
