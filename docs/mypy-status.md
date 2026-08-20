# gps_plot — the typing gate

Adopted 2026-08-18. `uv run mypy src` is **clean** under the scope declared in
`pyproject.toml`'s typing policy, and `.github/workflows/ci.yml` enforces it
alongside ruff.

Before this, `mypy` appeared nowhere in `pyproject.toml` — not in the dev
group, no `[tool.mypy]` section — while `CLAUDE.md` described the modern lane
as "typed, mypy-strict/ruff/black clean". That claim was never checkable here.

## What the gate covers, and what it does not

`strict = true`, scoped to the same modern lane as the ruff excludes:

- **Checked:** `detrend_workbench.py`, `detrend_picker.py`,
  `detrend_picker_qt.py`, `dev_viz.py`, `maps.py`, `__init__.py`.
- **`ignore_errors`:** `gasmatplt`, `gmtplot`, `timesmatplt`,
  `plot_gps_timeseries` — golden-pinned plotting code. Widening the scope
  means typing those two big modules, which is a project, not a drive-by.
- **`exclude` (not `ignore_errors`):** the two dated snapshots. They are
  Python-2 source, so mypy fails at the *parser* and aborts the whole run
  before any per-module setting applies. `ignore_errors` cannot reach a file
  that will not parse — that distinction cost one debugging round.
- **`tests/` is out of the gate on purpose.** The Qt picker tests poke widget
  attributes that are `Any` all the way down; strict typing there would
  measure pyqtgraph's missing stubs, not this package's correctness.
- **`ignore_missing_imports`** for `geo_dataread`, `gps_parser`, `gtimes`,
  `tostools`, `pygmt`, `pyqtgraph`, `scipy` — none ship a `py.typed` marker,
  so mypy can check nothing inside them either way. That is ~30 of the
  original 137 errors: declining to report one fact 30 times, not lost
  coverage.

## Getting from 137 to 0

The baseline was **137 errors in 7 files**. Scoping removed 125 of them; 12
were real and fixed:

- 3 stale `# type: ignore` comments, now unused under this config.
- 2 `no-any-return` in `maps.py::_resolve_slip_component` — `product` is
  `Mapping[str, Any]`, so the key type stayed `Any` through `totals` and both
  `max(...)` returns silently widened the declared `str`. One annotation on
  `available` fixed both.
- 1 `dict` → `dict[str, Any]`.
- **6 `no-untyped-call` at the modern↔legacy seam** — the interesting ones.
  `detrend_workbench` is strict-checked and cannot call an untyped function,
  so four legacy helpers gained signatures: `timesmatplt.addEvent` and
  `plot_gps_timeseries._build_outlier_params` / `outlier_param_help` /
  `_stage_overrides`. Those four live in `ignore_errors` modules, so **mypy
  does not verify their bodies against the signatures** — a wrong annotation
  on `addEvent` (3 call sites here, more in the plot driver) would typecheck
  and fail at runtime. The full pytest suite is the check that matters there,
  and it was run.

## What the gate is not for

It would not have caught any of the five invariant violations in the picker
lane (`detrend-lane.md`). "A second place assembling the same decision" is a
design shape, invisible to a type checker — the picked-vs-declared steps bug
typechecks perfectly. Over 137 errors it found exactly one latent defect:

```python
import matplotlib as mpl        # timesmatplt.py:51
mpl.dates.HourLocator(...)      # line 1223 and 43 more
```

`import matplotlib` does **not** import the `dates` submodule. Verified:
bare `import matplotlib` → `hasattr(mpl, "dates")` is `False`; after
`import matplotlib.pyplot` it is `True`. It works only because line 54
imports pyplot, which pulls `matplotlib.dates` in as a side effect — an
undeclared dependency on another module's import order. Dormant, not broken.
Left alone here (`timesmatplt` is `ignore_errors`); one-line fix when someone
is in that file.

## CI

`.github/workflows/ci.yml` runs ruff check, ruff format --check and
`mypy src`. Two things it does differently from `gps_analysis`'s workflow,
which was the template:

- **`uv sync --all-groups --no-sources`.** `[tool.uv.sources]` overrides the
  sibling deps with editable paths (`../geo_dataread`, `../gps_analysis`,
  `../gps_parser`) that exist only in a local `gpslibrary` checkout. Without
  `--no-sources` the resolve fails outright in a single-repo CI checkout.
- **No pytest job.** 6 of 8 test files read real GLOBK station data and the
  deployed `~/.config/gpsconfig` catalogs; only 11 tests guard with
  `pytest.skip`. A test job that cannot pass trains everyone to ignore red —
  the same failure as a doc claiming a gate that was never installed.

All five sibling repos are public, so CI needs no token.

## RESOLVED 2026-08-20: the `gps_analysis` blocker

`outlier-step-protection-flanks` (37 commits) merged to `gps_analysis` main as
`eec3b5b`, bringing `staged.py` and the `GROUP_ORDER` / `TrajectoryModel` /
`with_steps` exports with it; `geo_dataread` (`bc78527`) and `gps_parser`
(`cd2a6a2`) followed. Verified the way it matters — a CI-faithful install,
siblings resolved from `git+https` rather than the editable checkouts:

```bash
UV_PROJECT_ENVIRONMENT=/tmp/ci-venv uv sync --all-groups --no-sources
/tmp/ci-venv/bin/python -m ruff check src tests      # All checks passed
/tmp/ci-venv/bin/python -m ruff format --check src tests
/tmp/ci-venv/bin/python -m mypy src                  # Success, 10 files
```

All three CI steps pass. The `@main` pin on `gps_analysis` stays explicit: the
repo's GitHub **default branch is still `analysis-lane-base`**, so an unpinned
URL would go back to resolving a package without the detrend API. The history
below is kept because that trap is still live.

## How the blocker looked (history)

Testing the CI path surfaced a real production bug, unrelated to typing.

`gps_analysis`'s GitHub **default branch is `analysis-lane-base`**, which
predates the entire detrend API. An unpinned `git+https://` URL resolves to
that default, so a production or container install of `gps_plot` received a
`gps_analysis` with no `detrend`, no `staged`, no `trajectory_from_record` —
i.e. the detrend lane and both pickers could not import. Invisible locally,
because `[tool.uv.sources]` substitutes the editable checkout. The dependency
is now pinned `@main` explicitly, which fixes most of it (7 CI-faithful mypy
errors → 3).

**The remaining 3 are a cross-package gap, not a `gps_plot` defect.**
`gps_analysis`'s `main` is **37 commits behind** the working branch
`outlier-step-protection-flanks`, and `detrend_picker.py` imports three things
that exist only on that branch:

| needed by `detrend_picker.py` | on `main`? |
|---|---|
| `gps_analysis.staged` (module) | no |
| `GROUP_ORDER` | not exported |
| `TrajectoryModel` | not exported |

So `mypy src` is clean locally (editable siblings) and would be **red in CI**
until that branch reaches `main`. CI triggers only on push-to-`main` and pull
requests, so nothing is red yet — but this must be cleared before a PR:

1. merge `outlier-step-protection-flanks` → `main` in `gps_analysis`, and
2. set that repo's GitHub default branch to `main`, after which `gps_plot`'s
   `@main` pin could go back to being implicit (leaving it explicit is safer).

## Cross-References

- `../pyproject.toml` — `[tool.mypy]`, and the `[tool.ruff] extend-exclude`
  whose scope it mirrors
- `../.github/workflows/ci.yml` — the enforcement, and why there is no
  pytest job
- `../../geo_dataread/pyproject.toml` — the typing-policy precedent this copies
