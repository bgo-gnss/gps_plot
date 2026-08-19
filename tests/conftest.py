"""Shared pytest fixtures and collection hooks for gps_plot.

Two things live here: the rcParams isolation below, and the network-test
deselect further down.

``tests/test_detrend_workbench.py::test_tos_live_matches_the_fixture`` is
marked ``@pytest.mark.network`` because it hits the live TOS REST endpoint
and so needs (a) the ``tostools`` import and (b) the Veðurstofa VPN.  A
pytest marker only SELECTS IN with ``-m network``; on its own it does not
deselect anything, so a plain ``pytest`` run used to execute the live test
and fail on every clean checkout / CI box / laptop off-VPN.

The hook below skips network-marked tests unless the operator explicitly
asked for them via ``-m`` (any ``-m`` expression mentioning ``network``).
``addopts = "-m 'not network'"`` was rejected as the fix because pytest
ANDs multiple ``-m`` expressions, so ``addopts`` + a later ``-m network``
on the CLI collapses to ``not network and network`` and runs nothing.
"""

import pytest


@pytest.fixture(autouse=True)
def _isolated_rcparams():
    """Undo each test's global matplotlib state before the next one runs.

    ``timesmatplt.init_plot_style`` turns on ``text.usetex`` PROCESS-WIDE and
    latches ``_STYLE_INITIALIZED`` so it only ever runs once — correct for the
    production plot lane, which wants one style for the whole run, and fine
    for a CLI that renders one figure and exits.  In a test session it is a
    one-way door: any workbench or plot-driver test that renders leaves usetex
    on for everything collected after it, and ``test_dev_viz.py``'s ``Δv``
    annotation then dies in latex ("Package inputenc Error").  Each file
    passed alone; ``uv run pytest`` failed.  That is the worst shape a suite
    can have — it made the gate unusable for every other fix, and two commit
    messages claimed a green "161 passed" that no clean run produced.

    Restoring rcParams alone is not enough: the latch would stay set and the
    next module that legitimately wants the production style would silently
    get matplotlib's defaults.  So the flag is reset too, and the reusable
    Figure with it, since it snapshots rcParams at construction.

    The underlying coupling is NOT fixed by this — importing ``dev_viz`` and
    the plot lane into one process still has the plot lane win.  Nothing does
    that today outside the test session, which is where it actually bit.
    """
    import matplotlib as mpl

    from gps_plot import timesmatplt

    saved = mpl.rcParams.copy()
    yield
    mpl.rcParams.update(saved)
    timesmatplt._STYLE_INITIALIZED = False
    timesmatplt._FRAME_FIG = None


def pytest_collection_modifyitems(config, items):
    """Skip ``@pytest.mark.network`` tests unless ``-m`` opted into them."""
    markexpr = config.getoption("-m") or ""
    if "network" in markexpr:
        return  # the operator explicitly asked for the live tests
    skip_network = pytest.mark.skip(
        reason="live TOS — run with -m network (needs tostools + VPN)"
    )
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip_network)
