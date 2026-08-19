# gps_plot — the dev-viz lane

Deep context for `dev_viz.py`, routed out of `gps_plot/CLAUDE.md` when that
file passed its 150-line subdir budget a second time. **Moved verbatim** — the
`--break-input` finding and the MCMC cost numbers were measured, and
compressing while moving is where that dies.

Read `../CLAUDE.md` first for the package summary and cross-references.

---

## Dev-viz (analysis lane, thread C / L5)

Three shared-axis panels for one station/component: observed + `lineperiodic`
fit with WLS velocity annotation (rate ± σ, horizontal |v|/azimuth ± σ);
detrended residuals; GBIS4TS BPD1/BPD2 optimal break model on the break-input
series with break epochs marked. All math is called from `gps_analysis` —
nothing is re-derived here. `--break-input` selects what detection runs on:
`raw` (zero-referenced observed, default — break epoch recovers, rates absorb
seasonal on short windows), `seasonal_removed`, or `residuals` (known-bad:
the lineperiodic fit absorbs the ramp — kept to keep the failure visible).

```bash
uv run gps-analysis-devviz --synthetic --component east --out /tmp/devviz.png
uv run gps-analysis-devviz --neu STA.NEU --component north --breaks 2 --out sta.png
uv run gps-analysis-devviz --synthetic --no-detect ...   # skip MCMC, overlay truth
```

Data source is a parameter: `load_neu()` reads published `.NEU` products
(`date time dN DN dE DE dU DU`, mm — aflogun-verified format);
`synthetic_station()` generates BPD+seasonal+noise series so the viz runs
without station data. MCMC dev default is 2 000 kept iterations on a 1-yr
series (`--runs`/`--t-runs`/`--days`; production GBIS4TS uses 1e6 — the dev
posterior is indicative only). Pre-H3 cost warning: each noise-parameter
step factorizes the dense N×N covariance (~10 ms at N=365, ~0.3 s at N=730),
so long series make detection slow until the Toeplitz/Cholesky speedup lands.

## Cross-References

- `../CLAUDE.md` — package summary, layout, console scripts
- `../../gps_analysis/CLAUDE.md` — the leaf math library this visualizes
- `../../PLAN-analysis-lane.md` — thread C / task L5 (dev-viz), task H6
  (the Toeplitz/Cholesky speed pass the cost warning above waits on)
- `detrend-lane.md`, `map-lane.md` — the other two deep lanes
