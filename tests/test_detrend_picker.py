"""The picker's contract: it writes the grammar the CLI parses.

The round trip is the whole point — anything the picker emits must parse back
into the plan it came from, or "one grammar, two producers" is a slogan.
"""

from __future__ import annotations

import shlex

import pytest

from geo_dataread.stage_plan import build_stage_plan
from gps_plot.detrend_picker import render_command


def _roundtrip(stages: list[str], holds: list[str]):
    plan = build_stage_plan(stages, holds)
    cmd = shlex.split(render_command("SELF", plan))
    again_stages = [cmd[i + 1] for i, a in enumerate(cmd) if a == "--stage"]
    again_holds = [cmd[i + 1] for i, a in enumerate(cmd) if a == "--hold"]
    return plan, build_stage_plan(again_stages, again_holds), cmd


@pytest.mark.parametrize(
    "stages, holds",
    [
        (
            ["clean:secular,periodic@2001.6:2019.5", "long:secular"],
            ["long:periodic=stage:clean"],
        ),
        (
            ["clean:secular,periodic@2001.6:2008.0", "long:secular,step"],
            ["long:periodic=stage:clean"],
        ),
        (["fit:periodic"], ["secular=donor:OLAC"]),
        (["a:secular@:2008.35;2008.7:"], []),
        (["a:secular@:"], []),
    ],
)
def test_emitted_command_parses_back_to_the_same_plan(stages, holds) -> None:
    plan, again, _ = _roundtrip(stages, holds)
    assert again == plan


def test_command_targets_the_workbench(self=None) -> None:
    plan = build_stage_plan(["clean:secular"], [])
    cmd = shlex.split(render_command("SELF", plan))
    assert cmd[0] == "gps-detrend-workbench"
    assert cmd[1] == "SELF"


def test_stage_prefix_only_when_ambiguous() -> None:
    # One stage: the borrow one-liner stays short.
    _, _, cmd = _roundtrip(["fit:periodic"], ["secular=donor:OLAC"])
    assert "secular=donor:OLAC" in cmd
    # Two stages: the prefix is emitted, because without it the CLI refuses.
    _, _, cmd2 = _roundtrip(
        ["clean:secular,periodic", "long:secular"], ["long:periodic=stage:clean"]
    )
    assert "long:periodic=stage:clean" in cmd2


def test_inherit_is_emitted_as_no_at_sign() -> None:
    # segments=None must not become '@:', which would silently change
    # "inherit the caller's domain" into "the full span".
    _, _, cmd = _roundtrip(["long:secular"], [])
    assert "long:secular" in cmd
    assert not any(a.startswith("long:secular@") for a in cmd)


def test_extra_flags_are_appended() -> None:
    plan = build_stage_plan(["a:secular"], [])
    cmd = render_command("SELF", plan, ["--max-gap-years", "1.5"])
    assert cmd.endswith("--max-gap-years 1.5")


def test_headless_backend_refuses_rather_than_hanging() -> None:
    import matplotlib

    from gps_plot.detrend_picker import pick_stage_plan

    old = matplotlib.get_backend()
    matplotlib.use("Agg", force=True)
    try:
        with pytest.raises(RuntimeError, match="cannot show a window"):
            pick_stage_plan("SELF", [2000.0], [[0.0]], None, {"model": "linear"})
    finally:
        matplotlib.use(old, force=True)


class TestMarimoNotebook:
    """The notebook is a real artifact, so it gets real checks.

    marimo enforces that each name is defined in exactly ONE cell — that
    strictness is what makes its dataflow analysable, and it caught a genuine
    duplicate-import bug while this was being written. ``marimo export`` runs
    that analysis, so exporting IS the structural test.
    """

    NOTEBOOK = "notebooks/detrend_picker.py"

    def test_notebook_exists_and_is_a_marimo_app(self) -> None:
        from pathlib import Path

        src = Path(self.NOTEBOOK)
        if not src.is_file():
            pytest.skip("notebook not present in this checkout")
        text = src.read_text()
        assert "marimo.App(" in text
        assert "app.run()" in text

    def test_dataflow_analyses_cleanly(self) -> None:
        # Catches duplicate definitions and cycles across cells.
        import shutil
        import subprocess
        from pathlib import Path

        if not Path(self.NOTEBOOK).is_file():
            pytest.skip("notebook not present in this checkout")
        if shutil.which("marimo") is None:
            pytest.skip("marimo not installed (dev group)")
        r = subprocess.run(
            ["marimo", "export", "script", self.NOTEBOOK, "-o", "/dev/null"],
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert r.returncode == 0, r.stderr

    def test_notebook_never_commits(self) -> None:
        # The contract: it emits the command, the workbench stores. A --commit
        # in here would be a second path to stored science.
        from pathlib import Path

        if not Path(self.NOTEBOOK).is_file():
            pytest.skip("notebook not present in this checkout")
        text = Path(self.NOTEBOOK).read_text()
        assert "commit_record" not in text
        assert "write_stage_plan" not in text


class TestQtPicker:
    """The Qt picker is a GUI, so test what can be tested without a screen."""

    def test_import_guard_explains_itself(self) -> None:
        from gps_plot.detrend_picker_qt import _require_qt

        pg, qtw = _require_qt()  # installed in the dev group
        assert hasattr(pg, "LinearRegionItem")
        assert hasattr(qtw, "QApplication")

    def test_picks_reach_the_fit_and_the_command(self) -> None:
        # Drives the real widgets offscreen: move a pick, assert the fit and
        # the emitted command both follow. This is the whole contract.
        import os

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        import numpy as np

        import geo_dataread.gps_read as gpsr
        from geo_dataread.detrend_estimate import FitDefaults, resolve_fit_settings
        from gps_plot.detrend_picker_qt import PickerWindow, _require_qt

        _pg, qtw = _require_qt()
        _app = qtw.QApplication.instance() or qtw.QApplication([])

        try:
            yearf, data, sigma, _ = gpsr.getData(
                "NYLA", ref="plate", Dir=None, tType="TOT", uncert=10
            )
        except Exception:  # pragma: no cover - station data not present
            pytest.skip("NYLA data not available in this environment")
        if yearf is None or len(yearf) == 0:  # pragma: no cover
            pytest.skip("NYLA data not available")

        settings = resolve_fit_settings("NYLA", None, FitDefaults(max_gap_years=3.0))
        w = PickerWindow(
            "NYLA",
            np.asarray(yearf, float),
            np.atleast_2d(np.asarray(data, float)),
            np.atleast_2d(np.asarray(sigma, float)),
            settings,
            max_gap_years=3.0,
            uncert=10.0,
        )

        # Full domain on NYLA aborts, so there is no record and the command
        # carries no --segment.
        assert "--segment" not in w.command.text()

        # Drag the domain: the fit must follow and the flag must appear.
        for r in w.domain_regions:
            r.blockSignals(True)
            r.setRegion((w.span[0], 2020.0))
            r.blockSignals(False)
        w.refit()
        assert w.record is not None
        assert max(w.record["rms"]) < 10.0  # was 65 mm across the unrest
        assert "--segment" in w.command.text()

        # Double-click equivalent: declaring the 2023-11-10 dike as a step.
        for r in w.domain_regions:
            r.blockSignals(True)
            r.setRegion(w.span)
            r.blockSignals(False)
        w._add_step(2023.862)
        assert "--step 2023.862" in w.command.text()
        assert w.record is not None
        assert w.record["step_epochs"] == [2023.862]

        # A transient reaches the model and the record's terms.
        w._clear_steps()
        w.cb_term.setChecked(True)
        for ln in w.onset_lines:
            ln.blockSignals(True)
            ln.setValue(2018.0)
            ln.blockSignals(False)
        w.refit()
        assert "--term log@2018.0,tau=2.0" in w.command.text()
        assert w.record is not None
        assert w.record["record_version"] == 2

    def test_emitted_command_parses_back(self) -> None:
        # Same contract as the other producers: whatever it prints must be
        # re-parseable by the grammar, or "one grammar" is a slogan.
        import shlex

        from geo_dataread.stage_plan import build_stage_plan
        from geo_dataread.term_spec import parse_term_spec

        cmd = (
            "gps-detrend-workbench NYLA --term log@2018.0,tau=2.0 "
            "--stage clean:secular,periodic@2006.5767:2018.0 "
            "--stage long:secular,transient --hold long:periodic=stage:clean "
            "--segment 2006.5767:2020.0 --max-gap-years 3.0"
        )
        c = shlex.split(cmd)
        parse_term_spec(c[c.index("--term") + 1])
        build_stage_plan(
            [c[i + 1] for i, a in enumerate(c) if a == "--stage"],
            [c[i + 1] for i, a in enumerate(c) if a == "--hold"],
        )
