"""Shared pytest fixtures and collection hooks for gps_plot.

The single purpose of this file today is the network-test deselect below.
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
