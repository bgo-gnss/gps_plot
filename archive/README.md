# Archived code (not installed, not imported, not type-checked)

Moved out of `src/gps_plot/` on 2026-11-16 (BGÓ). These modules are Python-2-era
reference code: nothing in the package, the tests or the tools imports them, and
the live equivalents are `timesmatplt.py` (the time-series figure) and
`plot_gps_timeseries.py` (its CLI).

They were kept as reference, not as code: `mypy` was told to skip them and pyright
and the linters still reported them (missing `gpsUtils`, a bare
`import timesmatplt`, `== None`, mutable defaults, `gdata` undefined), so every
session's diagnostics bar carried ~25 findings that nobody was going to fix in
dead code. Archived, they leave `src/` clean and stay readable here.

| file | lines | what it was |
|---|---|---|
| `gps_plot/gasmatplt.py` | 608 | GAS strain series plots, `gpsUtils`-era |
| `gps_plot/gasmatplt_bgo_15May17.py` | 608 | dated snapshot of the above |
| `gps_plot/gasmatplt_workingon15May17.py` | 613 | dated working copy of the above |
| `gps_plot/gmtplot.py` | 89 | the old `gmtplot` entry |

Recover anything here with `git log --follow -- archive/legacy/gps_plot/<file>`;
the files are unchanged by the move, so history is intact. If one is ever needed
again, bring it back into `src/` and type it — do not import from `archive/`.
