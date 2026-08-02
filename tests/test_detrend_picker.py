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
