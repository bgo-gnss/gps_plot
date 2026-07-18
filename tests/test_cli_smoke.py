"""Console-script smoke tests.

These exist because `plot-gps-timeseries` was broken for its entire life without
anyone noticing: `main()` called `config.getPostprocessConfig(...)` (lowercase p,
and the wrong method besides) while *building* the argparse parser, so the script
died on import of its own arguments — `--help` included. The library
(`timesmatplt.plotTime`) worked, so the unit tests were green and the failure was
invisible until someone ran the CLI.

The lesson these tests encode: argparse `default=`/`const=` are evaluated at
parser-construction time, so anything they call runs before argument parsing —
including on `--help`. Config lookups there must not raise.

Run the real entry point in a subprocess. Importing `main` and calling it would
not catch a broken `[project.scripts]` wiring, and monkeypatching config would
hide exactly the failure mode being tested.
"""

from __future__ import annotations

import os
import shutil
import subprocess

import pytest

# The installed console script, not `python -m` — the cron lines and the
# container CMD invoke this name, so this is what must be exercised. Resolving
# it also proves the [project.scripts] wiring survived packaging.
_CLI = shutil.which("plot-gps-timeseries")

requires_installed_cli = pytest.mark.skipif(
    _CLI is None,
    reason="plot-gps-timeseries console script not on PATH (install the package first)",
)


def _run_cli(
    *args: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """Invoke the installed console script the way a user or cron line would."""
    assert _CLI is not None
    return subprocess.run(
        [_CLI, *args],
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, **(env or {})},
    )


@requires_installed_cli
def test_help_exits_zero() -> None:
    """`--help` must work. This is the regression that broke the console script."""
    proc = _run_cli("--help")
    assert proc.returncode == 0, (
        f"plot-gps-timeseries --help exited {proc.returncode}\n{proc.stderr}"
    )
    assert "Plot tool for GPS time series" in proc.stdout


@requires_installed_cli
def test_help_works_without_any_deployed_config(tmp_path) -> None:
    """`--help` must not require a deployed gpsconfig.

    The container image is built and smoke-tested before `gpsconfig` is
    deployed into it, and `totPath` is commented out even in the real deployed
    config — so a config lookup that raises at parser-construction time makes
    the CLI unrunnable in exactly the environments we are moving to.
    """
    proc = _run_cli("--help", env={"GPS_CONFIG_PATH": str(tmp_path)})
    assert proc.returncode == 0, (
        f"--help must survive a missing config; exited {proc.returncode}\n{proc.stderr}"
    )


@requires_installed_cli
@pytest.mark.parametrize(
    "flag",
    [
        # The flag surface the gpsplot cron scripts depend on. If any of these
        # disappears, the port of plot-gamit.sh / plot-reykjanes.sh breaks —
        # silently, since a dropped flag becomes an unrecognized-argument error
        # only at runtime, inside a cron job nobody reads.
        "--figDir",
        "--Dir",
        "--ref",
        "--special",
        "--ylim",
        "--uncert",
        "--start",
        "--end",
        "--events",
        "--eventf",
        "--fix",
        "--save",
        "--tType",
        # Modern-lane levers the new scheduler relies on.
        "--view",
        "--workers",
    ],
)
def test_legacy_flag_surface_present(flag: str) -> None:
    """Pin the CLI contract shared with the legacy 2018 scripts."""
    proc = _run_cli("--help")
    assert proc.returncode == 0
    assert flag in proc.stdout, (
        f"{flag} missing from the CLI — the gpsplot port depends on it"
    )
