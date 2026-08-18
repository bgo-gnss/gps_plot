# gps_plot — where mypy actually stands

Measured 2026-08-18. `gps_plot/CLAUDE.md` described the modern lane as
"typed, mypy-strict/ruff/black clean" — a claim nothing in this checkout could
check, because **`mypy` appears nowhere in `pyproject.toml`**: not in the dev
group, and there is no `[tool.mypy]` section. `uv run mypy` dies with
`Failed to spawn: mypy`.

Six siblings do run it (`gps_analysis`, `geo_dataread`, `gtimes`, `gps_api`,
`receivers`, `aflogun`), so `gps_plot` is the ecosystem outlier, not the norm.

## What it says today

Run against the ruff-scoped modern lane (legacy `gasmatplt*` and `gmtplot`
excluded, as they are for ruff):

```bash
uv run --with mypy mypy --strict --python-version 3.13 \
    --exclude 'gasmatplt.*' --exclude 'gmtplot' src/gps_plot/
```

**137 errors in 7 files.** The distribution is the whole point — it is not a
uniformly untyped package:

| file | errors | character |
|---|---|---|
| `timesmatplt.py` | 75 | never typed; 44 are one repeated pattern (below) |
| `plot_gps_timeseries.py` | 31 | never typed |
| `detrend_workbench.py` | 20 | mostly untyped sibling imports + calls into legacy |
| `detrend_picker_qt.py` | 8 | `pyqtgraph`, `scipy.signal`, `geo_dataread` stubs |
| `maps.py` | 4 | 2 missing stubs, 2 `no-any-return` |
| `detrend_picker.py` | 2 | missing stubs |
| `dev_viz.py` | 1 | missing `gtimes` stub |

By code: 57 `attr-defined`, 28 `import-untyped`, 23 `no-untyped-call`,
20 `no-untyped-def`, 3 `unused-ignore`, 3 `no-any-return`, 2
`import-not-found`, 1 `type-arg`.

**~30 of the 137 are not defects in this package.** `geo_dataread`,
`gps_parser`, `gtimes`, `tostools`, `pyqtgraph`, `pygmt` and `scipy.signal`
ship no `py.typed` marker or stubs. `gps_analysis` handles exactly this with

```toml
[[tool.mypy.overrides]]
module = ["scipy.*", "gtimes.*"]
ignore_missing_imports = true
```

**`dev_viz.py` and `maps.py` — the modules the "mypy-strict clean" line was
really about — are effectively clean** (1 and 4 errors, all stubs or trivial
`no-any-return`). The claim was roughly true of them and false of the sentence
it was written in.

## One finding worth acting on regardless

44 of `timesmatplt.py`'s 75 errors are the same line shape:

```python
import matplotlib as mpl        # line 51
...
mpl.dates.HourLocator(...)      # line 1223 and 43 more
```

`import matplotlib` does **not** import the `dates` submodule. Verified:

```
bare import matplotlib -> has .dates? False
after importing pyplot -> has .dates? True
```

It works today only because line 54 imports `matplotlib.pyplot`, which pulls
`matplotlib.dates` in as a side effect. That is an undeclared dependency on
another module's import order — dormant, not broken, and a one-line fix
(`import matplotlib.dates as mdates`). It is a fair example of what the gate
would buy: nothing here is failing, and nothing here is guaranteed either.

## Adopting it, if wanted

The path `geo_dataread` already took, which is the closest analogue (a typed
new lane beside golden-pinned legacy):

1. `mypy>=1.13` into `[dependency-groups] dev`.
2. `[tool.mypy] strict = true`.
3. `ignore_missing_imports` overrides for the untyped siblings and GUI libs —
   removes ~30 errors that are not this package's to fix.
4. Per-module `ignore_errors` for `timesmatplt` + `plot_gps_timeseries`, with
   the same scope rationale as the existing `[tool.ruff] extend-exclude` —
   removes ~106 more, and stops the gate from demanding a rewrite of
   golden-pinned plotting code.

That lands the modern lane at a real, enforceable near-zero without touching
legacy. It is a dependency-metadata change, so it needs a decision rather than
a drive-by commit.

## Cross-References

- `../CLAUDE.md` — package summary and the Status section this backs
- `../pyproject.toml` — `[tool.ruff] extend-exclude`, the existing scope precedent
- `../../geo_dataread/pyproject.toml` — the typing-policy comment worth copying
