# gps_plot — the detrend lane

Deep context for the detrend/curation lane, routed out of `gps_plot/CLAUDE.md`
when that file passed 450 lines. **Moved verbatim** — every measured number
here (RHOF 90 → 45 mm, SELF `step_amp_1 [-150.8, 148.0, 55.6]`, the
non-monotonic 4.0 → 20 / 3.0 → 55 / 2.0 → 0, 0.54 mm/yr, 16.48 mm) was
expensive to obtain, and compressing while moving is where that dies.

Covers: the cleaned view (`plot-gps-timeseries --view cleaned`), the detrend
workbench, segments, and the Qt picker. Read `../CLAUDE.md` first for the
package summary and cross-references.

---

## Cleaned view (`--view cleaned`)

`timesmatplt._mask_outliers` mirrors `gps_views.read_gps_view`'s station-aware
chain (`station_step_epochs` + `resolve_protect_windows` +
`resolve_outlier_detection`), so a plotted cleaned view matches the canonical
read/write path for that station; each resolver degrades to "nothing declared"
with a `UserWarning`, never a failed plot. Declared steps are not optional — an
undeclared coseismic/equipment step over-flags or trips the excess-candidate
abort at ANY threshold. Flags MASK (NaN) + grey overlay.

Three marker states, and the distinction is what each COMMITS to:

| marker | meaning | in the series? |
|---|---|---|
| red | clean | yes |
| grey | flagged outlier | **no** (NaN) |
| **gold** | **provisional** — recent, indeterminate step evidence | **yes** |

Gold epochs are the ones the detector cannot rule on yet (no data follows
them, so a blunder and the onset of deformation look identical). They stay in
the data; the marker only says the verdict is pending, and it WILL change as
epochs arrive. The grey overlay is **hidden by default** (display only —
the epochs are masked either way; hiding the overlay lets the y-axis tighten
to the cleaned series, often dramatically: RHOF north 90 mm → 45 mm of
range). `--show-outliers` restores it. Neither decides the masking, and
neither hides gold: decluttering removes decided outliers, never undecided
ones.
`--provisional-days` bounds the recency window (0 disables; default 14 from
`geo_dataread`), and the bound matters — indeterminate clusters also sit at
old mid-series gaps and would otherwise dominate.

```bash
plot-gps-timeseries RHOF --view cleaned --outlier-param window_n_sigma=3.0
plot-gps-timeseries RHOF --view cleaned --outlier-overrides ./overrides.csv
plot-gps-timeseries RHOF --view cleaned --uncert 10 --show-outliers
```

`--uncert` is the first lever for "obvious outliers survive": detection
whitens by the formal σ, so a large excursion with a large error bar is not
anomalous (RHOF: +83 mm Up at σ=13.2 mm scores only 3.3σ). The default 15 mm
lets those through; `--uncert 10` drops 40 of 4713 epochs and removes most of
them. Without σ at all the flag count rises ~45 %.

`--outlier-param NAME=VALUE` (repeatable, also `plotTime(outlier_params=)`)
builds a `gps_analysis.OutlierParams` off the dataclass itself — no default is
restated here, and an unknown NAME lists the valid ones. `--help` lists them
too, on both CLIs: `outlier_param_help()` generates the block (grouped by
stage via `OUTLIER_PARAM_GROUPS`, unknown fields falling into "other" so
coverage cannot go stale) into the parser epilog, which is why both parsers
now use `RawDescriptionHelpFormatter` and wrap their own prose. Any override REPLACES
the station's `outlier_overrides.csv` row; unset defers to that catalog, else
spec defaults. `min_outlier=` is the scalar floor, NOT the catalog's
per-component `[N,E,U]` vector.

Loosening is **not monotonic** — RHOF 2023→ (3483 component-epochs):
`window_n_sigma` 4.0 (spec) → 20 flagged, 3.0 → 55, 2.0 → **0**, because at k=2
the candidate fraction passes `max_flag_fraction=0.05` and the excess-candidate
rule aborts, silently serving raw. A sudden drop to zero means abort, not a
clean station. PLAN Phase 3: consume `read_gps_view`'s `{comp}_cleaned` /
`{comp}_outlier` instead of re-deriving here.

## Detrend workbench (`gps-detrend-workbench`)

One station, one PDF: plate-frame series + fitted trajectory, and the detrended
series. Detrend choice is **curation, not computation** — which window is
pre-unrest, whether a station supports periodic terms, whose parameters to
borrow. All estimation is called from `geo_dataread.detrend_estimate`
(`station_record_from_arrays`, `resolve_fit_settings`, `build_document`), so
workbench and batch `gps-estimate-detrend` can never disagree about what a
record means. Detection runs the FULL pipeline, falling back loudly to S0-only
if the excess-candidate rule aborts — an S0 record leaves model-visible outliers
in the fit, so on a fallback declaring the missing step beats accepting it
(measured on RHOF vertical: 0.54 mm/yr of rate difference).

**First `gps_plot` module that writes config.** `--commit` merge-writes one
station into `detrend_params.json`, preserving the rest; without it nothing is
stored. Proven end-to-end: `plot-gps-timeseries <STA> --view detrended` then
renders from that record.

**Per-station curation is config, not just a record.** `--commit` writes three
things a batch re-run must find, all merge-written one station at a time into
the deployed files: the record → `detrend_params.json`; the stage plan →
`analysis.yaml` `detrend.estimation.stage_plans`; the model + `--term`
transients → `analysis.yaml` `detrend.estimation.models`. A fit-time decision
living only inside the record is invisible to `gps-estimate-detrend`, which
RECOMPUTES the record — that was true of stage plans until 2026-08-16 and of
model/terms until the same day.

```bash
# catalogs deployed to ~/.config/gpsconfig 2026-07-29 — no GPS_CONFIG_PATH needed
gps-detrend-workbench SELF --max-gap-years 2.0 --out SELF-iter1.pdf
gps-detrend-workbench SELF --max-gap-years 1.0 --hide-outliers
gps-detrend-workbench NYLA --max-gap-years 1.5 --show-outliers   # audit the verdicts
gps-detrend-workbench RHOF --model periodic --donor VMEY --commit
gps-detrend-workbench SELF --segment 2002.1:2008.35 --segment 2008.7:2019.5 \
    --max-gap-years 1.5          # excise the transient, estimate the offset
```

### The drawn trajectory is the model, not the data epochs

`trajectory_curve` (in `detrend_workbench`, used by BOTH the PDF's page 1 and
the Qt picker) evaluates the record on its own dense daily grid. Evaluating at
`yearf` and joining the dots drew a straight chord across every data gap —
and that chord is the one part of the blue line that is not the model: it
says the station moved linearly, while the record says the secular trend
continued and the seasonal kept oscillating. SELF's widest gap is **268 days
with 0 samples inside it** under the old sampling. `evaluate_record` is
documented valid at arbitrary epochs, so nothing was being avoided by not
doing this.

Two things it deliberately does NOT change. **Extent** stays exactly
`[nanmin(yearf), nanmax(yearf)]` — sampling density is cosmetic, a wider
domain would be a new claim, and the dashed window edges already say where
the fit was constrained. **Steps** stay vertical: each `step_epochs` entry is
bracketed by a sample pair at ±`STEP_BRACKET_YEARS` (1e-6 yr ≈ 32 s), so the
jump renders vertical at any zoom. A NaN break would arguably be more honest
about a discontinuity, but it changes how every stepped figure in this
package has always looked, and the ask was about gaps. Bracketing also fixes
a case the old sampling got wrong: a step epoch *inside* a gap — SELF's 2008
Ölfus under a segmented fit — would otherwise ramp across the whole outage.

The picker keeps a SECOND evaluation at the data epochs, because the residual
periodogram needs data minus model at the same epochs. Two samplings, two
purposes; only the drawn curve moved. No fitted quantity changes — verified
on SELF (`step_amp_1 [-150.8, 148.0, 55.6]` segmented, unchanged) and by a
test that re-serializes the record around the call.

## Segments (`--segment START:END`, repeatable)

The fit domain is a **union of intervals**, not one window. That single
generalisation answers four asks at once: two (or N) detrend periods; cutting
transients out; offsets estimable *outside* the trend window; and include/skip
per offset (which epochs you declare). `--window-start`/`--window-end` is the
one-segment spelling and is refused alongside `--segment` — two ways to say
which epochs are fitted, and letting one win silently would change stored
science. An empty side is an open bound (`:2008.35`).

Why it works: `estimate_detrend`'s step filter uses the **hull** of the kept
epochs, so a step epoch inside an excised gap still enters the model, and its
amplitude is estimable precisely because the flanking segments constrain the
level on both sides. SELF 2008 Ölfus M6.3, transient excised:
`step_amp_1 [-150.8, 148.0, 55.6] mm` — a line that did not exist under any
single window, and it matches `steps.yaml`'s own annotation.

The gates changed meaning, each identically equal to the old one at J = 1:
**`max_gap_years` is now per segment** (on the hull the deliberate excision was
the largest diff, so every union was rejected — the blocker; and raising the
threshold past it would have disabled the gate *inside* every segment too);
**`min_span_years` is summed coverage** (Σ ≤ hull always, so uniformly stricter
— a hull gate passes two 18-day nubs 17 yr apart while four seasonal terms fit
36 days); `min_epochs` stays a total. A segment with **zero** kept epochs is a
hard error naming its index — the real failure is a bound typed into a data gap.

Two steps with no fitted epoch between them are now **refused** (identical
Heaviside columns ⇒ rank-deficient design, amplitudes meaningless, covariance
infinite). This bites in practice: `--step 2008.4071` on SELF merges with
`steps.yaml`'s declared 2008.4085 into two steps for one event. It used to
degrade silently.

Persisted per station via a `segments` column in `fit_windows.csv`
(`a:b;c:d`, `;` already means list there, `:` avoids minus-sign ambiguity) —
one cell, not one row per segment, because that reader is strict by design and
a mistyped extra row would parse into different-but-valid science. The header
check became an allowlist so the deployed 8-column files still read.

The stored record gains `segments` + `segment_gaps` **additively at
`record_version` 1**: `window` keeps its 2-tuple hull meaning (two readers
index it positionally), `trajectory_from_record` never reads either, so
segmented and the 37 deployed single-window records coexist in one document.
`segment_gaps` reports the *realized* excision rather than gating it — a
boundary placed inside a genuine outage hides it from the per-segment gate, and
naming the distance is the honest answer without adding a knob.

Rejected epochs get the cleaned view's grey vocabulary (`timesmatplt`'s own
`OUTLIER_*` constants, both pages); `--hide-outliers` is display-only there and
here. The mask is the FIT's inlier verdict, lifted across the non-finite drop
and the window subset by `geo_dataread.station_estimate_from_arrays` (new seam;
`station_record_from_arrays` now wraps it), so per-component counts equal the
printed `n_rejected`. Re-running `detect_view_outliers` *inside* the window
would disagree by construction — it sees neither the fit window nor a CLI
`--step`.

`--show-outliers` inverts that emphasis — flagged red, everything else grey —
display-only in the same sense (same masks, counts and record), exclusive with
`--hide-outliers`. Solid vs hollow still separates the two lanes, gold stays
gold (provisional has no inverse), and the y-axis widens. The swap is an
OVERPAINT, not a colour argument: `stdTimesPlot` draws red and has no colour
knob, and adding one would reach into `plot-gps-timeseries`' path for a
workbench display option — so the kept series is redrawn grey over itself, the
green "last datapoint" rim survives (it marks the epoch, not a verdict), and
default figures stay bit-identical.

What it makes legible: **an all-grey component is one where NOTHING was
flagged**, which on an aborted component is the only on-figure trace of the
abort — `screen_outside_window` discards `view_flags`' abort list, so
`ABORT_BADGE_*` never reaches this lane (it is wired on `--view cleaned`).
NYLA is the worked case (numbers in the memory note): unmodeled deformation in
the last two years alone trips the abort, and a +48 mm one-day blunder from
2022 therefore renders as ordinary data.

**Outside** the window there is no such conflict — the fit passed no verdict at
all, and with a pre-unrest window that is most of the series (RHOF: 3337 of
4789 epochs). Drawn plain they claim "clean" and one blunder owns the y-axis, so
`screen_outside_window` fills the silence with the view detector, restricted to
`~estimate.in_window` (`--no-screen-outside-window` to opt out). It is a
**second lane, never a merge**: hollow grey against the fit's solid grey, its
own printed count, and the record — `n_rejected`, `--commit`, everything stored
— is untouched. RHOF flags [31, 16, 10] out there and north's out-of-window
range falls 73 → 27 mm. Masked because a *view* verdict masks (the cleaned
view's rule), not because "not in the fit" — no out-of-window epoch is.

Two seams make the lanes agree rather than argue: `timesmatplt.view_flags` is
now the single detector call site (`_mask_outliers` is its plotting half),
taking `restrict=` — detection still runs on the FULL series, only the verdict
narrows — and `step_epochs=`, fed by `_declared_step_epochs` (steps.yaml ∪
fit-catalog ∪ `--step`). `record["step_epochs"]` is the wrong source and
instructively so: it keeps only epochs *inside* the window. Expect these flags
to differ from `--view cleaned` — `uncert` screens σ at read time and the
workbench defaults to 10 against the plot driver's 15.

Gold DOES appear out there, bounded by `--provisional-days` (same meaning as in
the plot driver; the bound matters more here, since a decade of screened epochs
holds old mid-series indeterminate clusters that would otherwise dominate the
lane). "A fit has no provisional category" is a statement about *fit* verdicts;
the view detector has one, and with a pre-unrest window the newest epochs are
exactly the out-of-window ones — rendering a genuinely undecided recent epoch
red would be the one claim nobody can make. Gold survives `--hide-outliers`;
both greys do not. The dashed royalblue window edges stay essential — they are
what says which grey is which, which is also why `--donor` skips the screen
outright and says so: the edges would then be the donor's window while the
unjudged epochs are the station's own.

`--out` shares the scratch figdir with `tools/local-plot/figview.sh`: a bare
filename lands in `$FIGDIR`, else the checkout's gitignored `tmp-figdir/`, else
CWD. A path with a separator is honoured verbatim, so an exported `FIGDIR` can
never relocate an explicit `--out`. `--max-gap-years` is effectively required —
the 0.5 default rejects every station in the working set.

Every event line is **clipped to the plotted span** by `render` itself
(`clip_events_to_span`), and `main` prints what that dropped. Not cosmetic:
`axvline` clips to the axes but `Text` does not, so an out-of-span event loses
its line and keeps its caption, stranded in the margin beside an axis it does
not mark. BJTV is the case — installed 2021-08-09, solution starts 2024.09.
The clip lives in `render` so a direct call (test, REPL) cannot skip it; an
empty or all-NaN `yearf` keeps everything, since with no span nothing can be
outside it.

Events are **declared, never detected** (tier A), in three colours: `darkgreen`
new antenna / receiver installs (live `tostools`), `darkred` seismic
(`steps.yaml` rows whose `kind` is
earthquake/coseismic/seismic, plus `--event YYYYMMDD[,LABEL]`), `royalblue` the
fit. Seismic lines read `steps.yaml` via `gps_parser.outlier_catalogs.read_steps`,
**not** `gps_views.station_step_epochs` — that one drops `kind`, so it cannot
separate an earthquake from an antenna swap. No skjálftalísa client exists
anywhere in the ecosystem (planned only, in `analysis.yaml`).

A green line claims the phase centre may have MOVED, so three filters stand
between a TOS row and one. Subtype: `EQUIPMENT_SUBTYPES` = `antenna` +
`gnss_receiver` only, verified against the canonical 153 — `children_connections`
is one row per device join of any kind, and RHOF's 2023-08-16 line was a GSM
modem plus a SIM card drawn across all three components. Resolving the device
takes one entity call each (`resolve_devices` → `DeviceInfo`, ~30 ms,
process-cached): the join row carries only `id_entity_child`, which is why
labels read "2 devices", then "(antenna, receiver)", and now name the
instrument — `2010-06-03 (rx TRIMBLE NETRS)`. Installs only — `time_to` is read
by nothing, a removal is not a line. **Actually new**: a join continuing the
same model AND serial it replaces is a re-registration, caught by
`is_same_unit`. SELF 2010-06-03 is the case — antenna TRM29659.00/263955
re-joins the day it closes, so the day claimed an antenna change when only the
receiver moved (5700 → NETRS); invisible until labels named the unit. Plus the
`1000-01-01` sentinel. **`MIN_DEPLOYMENT_DAYS = 30`**: a campaign measurement is
registered exactly like a permanent install, so only duration separates them —
SELF's 2001-07-01/-07-16/-09-14 are 3–4 day deployments against installs of
3040 days and open, four lines in a two-year span where two belong. An OPEN join
always counts however young, else the newest equipment on every station is
invisible. Survivors coalesce per day, always one line (RHOF's 13:20 and 15:30
are one visit). Net: RHOF 4 lines → 3, **SELF 6 → 2**. A whitelist fails
SILENT, so a lookup that resolves
nothing raises rather than returning an empty list — bare and "never swapped"
render identically. `monument` is excluded per the operator rule and is the one
excluded subtype that physically could matter.

Offsets are declare-and-fit: epochs FIXED, amplitudes estimated and shown at
once. Epoch detection is absent deliberately — it is *circular* today: a jump
detector needs clean data, the outlier detector needs declared jumps to make it
(SELF: 9.1 % candidates and abort until one step was declared).

Round-trip fidelity — what you judged is what gets stored — is enforced, not
assumed. `--terms` still does NOT round-trip (`model=` is stored at fit time,
`terms=` is per-call and unstored — two different decisions, 16.48 mm max
divergence), so `--commit` under a non-default `--terms` is **refused** before
any data is read (exit 5), naming `--model` as the lever that is stored.
Looking under `--terms` stays free. `uncert` screens sigma at READ time, so it
changes which epochs were fitted while leaving no trace in any fitted
quantity: both sides now carry it in `refs` and both expose the flag
(`gps-estimate-detrend --uncert`, default 15 = `getData`'s own; workbench
default 10), and a commit that used a non-default prints the batch invocation
that reproduces it.

Remaining gaps: `--donor` copies rather than points, so it will not follow a
re-estimated donor; the `max_gap_years=0.5` gate fails every station in the
working set, so pass `--max-gap-years` (0.5 is the *shared* default — it also
lives in `gps_analysis.estimate_detrend` and `gps_api`'s precompute config, so
changing it is a fleet decision, not a CLI default).

Fractional-year epochs are at **noon** — 2008-05-29 is `149.5/366 = 2008.40847`,
not `149/366`. That trap and the TOT join live in `tools/local-plot/README.md`.

## Qt picker (`gps-detrend-picker-qt`)

Layered ON TOP of the workbench CLI, never replacing it, and its whole
promise is one invariant: **the emitted command reproduces the figure.**
**The store extends the invariant:** a `store` button runs that command
in-process (declarations → ``steps.yaml``, then the commit →
``detrend_params.json``), with the command shown first in a dialog —
one write path, the workbench's own. The marimo picker
(``gps-detrend-picker``) was retired 2026-08-24; the Qt picker is now
the one GUI.

**Two restore-time guards** (2026-08-24), each fixing a state the window
used to restore silently and then refuse or mis-fit. `normalize_segments`
runs on session load: overlapping clean intervals are dropped (keep the
earliest non-overlapping chain) with the dropped ones NAMED, because a
hairline overlap — an edge dragged a few days past its neighbour — used to
leave the window stuck on "NO RECORD" every restart with no way to tell the
file from a fresh open. And in the events phase, when holding ``s(t)`` from
``self``, a step whose epoch lies outside the saved background's span is
announced ("⚠ offset at … is before the background's earliest data") rather
than silently measured against an extrapolation — the SELF 2008 coseismic
comes out ~0 mm against a post-event-only background, and naming the step is
what turns that into something fixable (re-save ``s(t)`` with a clean
interval on EACH side of it).

Every divergence found so far has been a second place that assembled the
same decision — so settings are built by the workbench's own
`_override_settings` (one assembly site), and every run parameter that
changes the data is emitted.

Eight ways it broke that invariant — the sixth is the `--max-gap-years`
ordering, the seventh the missing abort fallback and the eighth a refused stage
plan emitted as an unstaged command, all three described above —
all fixed 2026-08-09/16/17/19/20 and worth
knowing because the shape recurs: a picked step REPLACED the declared ones
while `--step` MERGES (on SELF the difference between an aborted fit and a
clean one); the domain region opened on the DATA SPAN and emitted `--segment`
only when moved off it, so an untouched region on a station with a
`fit_windows.csv` window fitted everything and emitted a command reproducing
the window; `--tot-dir` was never emitted (a different series, not a
different fit); `--uncert` was float here and int there, so any non-default
screen emitted a command that will not parse; and the stage lane's final-stage
free-group list was built from the PICKED steps.

That fifth one is the same lever as the first. `steps.yaml` is a floor that
`_override_settings` merges in, so on a station with a declared step an
untouched stage plan freed only `secular` while the fit still carried
`step_amp_1` — refused, every time, with *"never estimated and not held in
the final stage"*. Both sides refused identically, so the emitted command
still reproduced the figure; what broke was the **feature**: the stage lane
was unusable on exactly the two stations in `steps.yaml` (SELF, HOFN), and
nothing said that re-declaring the already-declared step was the way out.
`free2` now asks `_declared_step_epochs(sta, settings.steps)` — the merged
set, the same function `_override_settings` uses. Conditional, not blanket:
RHOF has no declared step and its plan is byte-identical to before
(`--stage long:secular`).

**One assembly site for the run flags, because there were two pickers when
this was written** (the marimo ``gps-detrend-picker``, retired 2026-08-24).
`detrend_workbench.run_flags()` builds the `--tot-dir` / `--max-gap-years` /
`--uncert` / `--provisional-days` tail. The marimo picker had TWO of the
four violations above still live after the Qt picker was fixed — `--tot-dir`
read and never emitted, and `--uncert` as `type=float` emitting `12.0` at a
`type=int` parser — while printing *"the workbench re-parses this line, so
every refusal still applies."* Two pickers assembling the same list is two
places to forget the same flag — which is why there is now one.
`WORKBENCH_UNCERT_DEFAULT` now lives in `detrend_workbench` beside
`BATCH_UNCERT_DEFAULT`; omission is only correct because it is the
workbench's own default, not a number restated. (The marimo picker
formerly accepted `--uncert 12.5` and hard-errored — a breaking change
that was deliberate, because what it accepted it could not emit.
`run_flags` RAISES on a non-integral `uncert` rather than rounding.)

Measured 2026-08-17 by driving `PickerWindow` offscreen and diffing the
picker's record against the one the emitted command produces, elementwise:
**11 cases, fitted quantities identical in all of them** (untouched, moved
domain, picked step, `--term`, stage, stage+step, stage+term, RHOF stage
baseline, `--uncert 12`, `--provisional-days`+`--tot-dir`, catalog union).
Two provenance-only divergences existed when the picker had no `--commit`
(the store now runs the workbench command in-process, resolving them):
`refs.uncert` is absent
picker-side (deliberate — `estimate_record`'s docstring states the picker
passes a subset), and `refs.window_source` reads `defaults` picker-side
against `workbench-cli(+defaults)` CLI-side, because the picker folds
`--max-gap-years` into `FitDefaults` where the workbench routes it through
`_override_settings`.

The union case is verified against a SYNTHETIC catalog injected via
`--fit-catalog`, not deployed config: the deployed `fit_windows.csv` has
exactly one row (DYNG, no window, no segments), so no deployed row declares a
union. Note also that the picker has no `--fit-catalog` flag — it always
reads the deployed catalog, and so does the command it emits.

**Group states persist; `step` joins them; `transient` and MLE σ do not**
(2026-08-20, closing the composition work's open points).

Group states are saved with the session, restored AFTER `cb_stage` — because
`_toggle_stage` rewrites them, so restoring first is simply undone — and a
`hold` is *not* restored when the payload says staging was off, since that
would show a value the fit cannot use. An unknown group or state is DROPPED
rather than fatal: this key is newer than the files already in the field, and
a session written by a build that knows one more term group must not make the
rest of somebody's curation unusable.

`step` gets free/held but **no ABSENT**: there is no CLI spelling for
un-declaring a step. `steps.yaml` is a floor that merges in, and a picked step
is removed by removing the pick — so the state would promise something no
emitted command could carry out.

`transient` deliberately stays a checkbox. Its third state would be "hold from
the clean window", i.e. a transient estimated inside the window chosen for
being quiet — a state that is nearly always wrong, so converting a hardened,
session-coupled control to gain it buys uniformity and nothing else.

**MLE σ was attempted and NOT shipped.** `estimate_noise_mle` would give
honest (white + power-law) uncertainties where the record carries optimistic
WLS formal σ, and reporting it in the refine panel was the plan. Wired up, it
failed its known-truth check: on a synthetic with a true 3.00 mm/yr trend the
record recovers 2.98/2.93/3.05 while the MLE call returned 1.96/1.80/1.95 with
κ pinned at the −2.5 bound — a misspecified design or a `t_ref` convention
mismatch. A rate 35 % low under the label "honest σ" is worse than the absence
of the feature, so it was removed rather than shipped. Redo it by establishing
the estimator's own centering convention first.

**The model form on screen, the numbers in the terminal** (2026-08-21). The
panel leads with the general equation — `x(t) = a₀ + a₁·(t−t₀) + c₁·cos(2πt) +
…` — read off the record's own `param_names`, so it always describes the model
that was actually fitted, however many steps and transients it carries, rather
than a formula written down once and left to rot. The full per-component
parameter vector is printed to the **terminal** instead: with steps and
transients it is wider than the control column, and it is something an operator
wants to keep, scroll and paste. Components are ANSI-coloured red/green/blue to
match the residual periodogram's three curves, so a component is the same
colour in both places.

**`secular` is called `linear` on screen** (2026-08-21, BGÓ). The stage
grammar's `secular` names the LINEAR term alone, but "secular" properly names
the long-term background as a whole — linear *and* periodic, which is what
`lineperiodic` composes. Showing the grammar's word invites reading one row as
the whole background. The row therefore reads `linear`, every emitted flag
still says `secular`, and the tooltip names both so the command stays traceable
to the control. `GROUP_LABELS` is display-only: renaming the group itself would
reach into `gps_analysis`'s `GROUP_ORDER` and the 37 deployed records.

**Detrended view** (`detrended (data − f(t))`, 2026-08-21). Plots the residuals
instead of the series, relabels each axis (`North residual [mm]`) and stops
drawing the trajectory — subtracted, the model IS the zero line. DISPLAY ONLY,
the same convention as `--hide-outliers`: the masks, the record and the emitted
command are untouched, because it subtracts a model that was already fitted
rather than fitting a different one. This is where a signal departing from the
background becomes readable — RHOF's 2020–22 unrest is unmistakable in Up once
the background is gone.

**The header says how many epochs the domain kept** (2026-08-21) — `fitting
1718 of 4812 epochs`. The blue region IS the control that decides which epochs
are fitted, but shaded background does not read as a control, and the plots
show the whole series either way — so a fit restricted to a window looked
identical to a fit over everything, and the same confusion recurred across a
whole session.

**To fit on a window and see it extended, move the DOMAIN, not the stage.**
A recurring confusion, and the tool invited it: the blue domain region already
restricts which epochs are fitted while `trajectory_curve` still draws the
model across `[nanmin(yearf), nanmax(yearf)]`. Measured on RHOF: dragging the
domain to 2001.7327–2016.2057 fits 1718 of 4812 epochs (rate
`[-0.672, -0.759, -0.119]` against the full `[-0.707, -1.185, 0.105]`) and
emits `--segment 2001.7327:2016.2057`, with the curve still drawn to 2026.63.
Staging is for something else — carrying part of the model across a span while
estimating the rest against it — and holding the WHOLE background needs
something left free, which a station with no step and no transient does not
have.

**A refused fit must LOOK refused** (2026-08-21). The trajectory goes grey and
dashed when the fit is refused, because the previous configuration's curve
stayed on screen solid and blue and was read as a result. Measured: RHOF staged
with the whole background held has nothing free in the long stage, so the plan
is refused — the header said `NO RECORD`, the curve said otherwise, and the
curve won. An operator spent a session believing they were looking at a fit
"within the orange window" that had never been computed.

**Parameters go in the PANEL, not only stdout** (2026-08-21). The picker is
normally launched from a sway keybinding, which `exec`s it with no terminal
attached — so printing the parameter table to stdout put it nowhere anyone
could see. It is in the panel under the record now, and still on stdout for the
times it is run from a shell.

**Compare unstaged, then adopt** (2026-08-21). Unchecking `stage the fit` to
look at the plain fit is DESTRUCTIVE — the toggle rewrites the group states on
the way out and again on the way back, so peeking costs you the setup.
`compare unstaged` instead fits the same configuration with no stage plan and
draws it as a **magenta dashed overlay** beside the trajectory, printing both
parameter sets to the terminal. The figure, the emitted command and the group
states are all untouched: the blue line remains the only thing the command
describes, which is what keeps the overlay from becoming violation nine.
`switch to the unstaged fit` then makes it real, by turning staging off and refitting —
so the record, the figure and the command all come from one refit, as they do
for every other control. The overlay clears on any refit, because a comparison
is a snapshot of a DIFFERENT configuration and leaving it up after the blue
line moves invites reading the two as one fit.

Worth knowing what it shows: on RHOF with `periodic` held from 2003–2015, Up's
`cos_annual` is **1.818** staged against **1.086** unstaged — the held seasonal
comes from the clean window and genuinely differs from the all-data one.

**Refine τ from the visual seed** (2026-08-20, slice 3). The visual fit fixes
everything except the one genuinely nonlinear parameter, which the operator has
been setting by eye — and `gps_analysis.profile_transient_tau` exists for
exactly that, "the opt-in nonlinear refinement of an operator-fixed τ". The
button solves it by VARPRO, seeded by the fit on screen. Verified on a
synthetic series with a known τ = 1.50: seeded deliberately wrong at 4.0, it
recovers 1.505 ± 0.020 with a closed interval.

Per COMPONENT, because the profiler takes one series while `--term …,tau=X`
applies one τ to all three. All three are reported and the best-constrained
(tightest relative interval) is applied — a stated rule, overridable by typing.
Applying it means writing the **spinbox**, which is the single source the fit
and the command both read, so a refinement cannot move one without the other.

**An unclosed interval is a BOUND and is never applied.** The profiler's own
guidance is to publish a bound when its identification conditions fail
(T_post ≳ 5τ̂, amplitude SNR ≥ 5), and applying one would silently turn "τ is at
least this" into "τ is this". SELF is the real case: a transient placed on its
declared 2008 coseismic is collinear with the step, so τ runs to the bound and
the spinbox is left alone with the reason printed.

Two things the first version got wrong, both caught by running it rather than
reading it. The profiler was given bounds (0.02, 40) wider than the spinbox's
(0.05, 50), so a solved τ = 0.020 came back and the control clamped to 0.05 —
the summary and the command then disagreed about the number the figure was
drawn with; bounds now come FROM the spinbox. And τ gained a third decimal,
because refinement resolves it to better than a hundredth of a year and
rounding back to 2 dp would discard precision the fit had just earned.

**Per-group hold — the background model** (2026-08-20, slice 2). The stage plan
is now COMPOSED from the group states rather than hardcoded: whatever is *held*
is estimated on the clean window and carried across the full span, and what
stays free is estimated against that background. Turning `stage the fit` on
defaults `secular` and `periodic` to held, which is the background model —
trend and seasonal from the quiet window, extended, leaving residuals in which
short-term deviations can be read.

That default is a change of scientific claim. The old plan hardcoded *hold
`periodic` only*, so the trend was silently re-estimated over the whole span
including the unrest it was meant to be a background FOR. The old behaviour is
still reachable — set `secular` back to `estimate here` and the command returns
to `--stage long:secular,step --hold long:periodic=stage:clean`.

The clean stage still frees `secular` as a NUISANCE even when the trend is
estimated later, because a seasonal fitted on a window that ignores the trend
inside that window absorbs part of it. Turning staging off puts any held group
back to `estimate here` rather than leaving an unreachable state selected.

Holding *everything* is refused, not corrected: with nothing free the long
stage estimates nothing, and the grammar's own answer is that "a stage that
estimates nothing is not a stage". On a station with no declared step and no
transient — RHOF — the default therefore refuses, and that is the honest
outcome rather than a silently different plan.

Which is how **violation eight** appeared, introduced while building this
slice and caught by an existing test. When the plan would not build,
`_command` fell through to the UNSTAGED spelling, so the window showed a
refusal while the command described a different, perfectly fittable fit
(`gps-detrend-workbench RHOF --max-gap-years 1.5`). Copying it produced the
figure the picker had just refused to show. A refused plan is now emitted as
asked, and the workbench refuses it with the same message.

Sweep at this slice: 12 unstaged + 7 staged combinations, 0 divergences.

**Three-state term controls** (2026-08-19, slice 1 of the composition work —
alignment in `.interrogate-picker-terms.md`). `secular` and `periodic` each get
**estimate here / hold from window / not in the model**, because those are three
CLAIMS and not two: a seasonal *held* from the quiet window asserts it continues
across the span, a seasonal *absent* asserts there is none. Three states are
expressible at all only because `--model` and the stage plan are **orthogonal**
in the estimator — one decides which terms are in the design matrix, the other
where each is estimated.

The two states compose the stored `--model` (`lineperiodic` / `linear` /
`periodic`) in `_current()`, the same place the flags are built, so the design
matrix the picker fits and the model the copied command asks for cannot come
apart. Both absent has no spelling — every model in the vocabulary carries at
least one — so it is refused there rather than sent to the estimator. `hold` is
greyed until staging exists to hold *from*, not hidden: removing the item would
renumber the rest and silently move a stored pick.

Making `--model periodic` reachable exposed the **seventh** violation. The leaf
RETURNS None on an outlier abort (recoverable — retry S0-only) and RAISES on a
failed gate (not). `build_record` handled that; the picker called
`estimate_record` directly and did not, so the same command rendered a figure
from the CLI and "no record" in the window. A periodic-only model leaves the
trend in the residuals, the candidate fraction trips `max_flag_fraction`, and
detection aborts — invisible while the model was always `lineperiodic`. Both
now go through `estimate_with_abort_fallback` and report with
`abort_fallback_note`, so an S0 record is announced as one in either tool.

Verified by the sweep that is this work's definition of done: 12 unstaged
combinations (secular × periodic × term × step), picker record vs the emitted
command's record, diffed elementwise — 0 divergences.

**Run parameters are live controls** (2026-08-19) — `max-gap [yr]`,
`provisional [d]` and `uncert [mm]` in a `run` group, previously
command-line-only. Each writes the SAME attribute `run_flags` emits, so a
control cannot move the figure without moving the command with it; that is the
only way to add a knob here without adding another way to break the invariant.

`uncert` is the odd one and is treated differently: it screens sigma at READ
time, so it **re-reads the series from disk** rather than refitting. A refit
alone would have moved the command while the figure kept the old series — the
same divergence one lever over. The picks are deliberately left where they are
(screening sigma is not a statement about which window to fit), a failed read
keeps the current data and says so, and the epoch delta is printed above the
record because it is otherwise invisible: SELF 10 → 8 mm drops 377 of 4902
epochs and looks identical on a plot of 4525.

Adding the gap control surfaced the **sixth** violation, pre-existing.
`main` baked `--max-gap-years` into `FitDefaults` BEFORE `resolve_fit_settings`,
which puts it *below* the catalog row, while the workbench applies it as an
override AFTER resolving. So on a station whose `fit_windows.csv` sets its own
gate the flag was silently discarded: `gps-detrend-picker-qt DYNG
--max-gap-years 2.0` fitted at the catalog's 1.0 and emitted a command that
fits at 2.0. Same shape as the other five — a second place assembling the same
decision. The picker now uses plain `FitDefaults` and passes the gate through
`_override_settings`, exactly as `build_record` does.

**Layout: plots left, controls right, command full width** (2026-08-19).
Everything used to stack vertically, which spent HEIGHT — the scarce dimension,
with three component panels plus a periodogram — on a one-line control strip
that had itself run out of width. The divider is a `QSplitter`, so the share is
draggable; the plots take the space on resize and the control column stays put.

Two details that are deliberate. The **command stays full width along the
bottom** rather than joining the right column: it is the window's output, runs
past 200 characters on a staged fit with a term and a segment, and exists to be
read and copied — which a third of the window cannot do. And the controls are
grouped by **what a control decides**, not by widget type: *model* changes what
is fitted and therefore the record, *picks* only moves what is already on the
plot. That is the distinction an operator needs to tell curation from
navigation, the same one the workbench draws between a stored decision and a
look-only one.

`rms` is now rounded to 2 dp in the summary, matching the PDF's `summarise`.
Raw it printed as `[1.8299068022476694, …]` and wrapped over three lines in the
narrower column, burying the number actually being compared between iterations.

An UNTOUCHED domain region passes `segments=None` rather than its own hull —
a catalog row may declare a UNION and one region cannot draw one, so the
header says `shown as hull of N catalog segments` instead of silently
re-including an excision. `load_session` runs at LAUNCH and parses the whole
payload before touching a widget: a corrupt session degrades to the declared
defaults and names the file (it is somebody's curation), where it used to
take the application down before the window appeared.

## The compositional model — f(t) = Σ mᵢ(t)

Added 2026-08-22. The operator builds a trajectory in parts: the background
on a quiet window, then steps against that background, then transients
against both. **The CLI grammar already expressed all of it** —
`build_stage_plan` has always taken N stages with per-stage windows and
holds. What could not express it was the GUI, which offered two stages and
one window and *derived* the second stage's groups rather than letting them
be chosen. So this was UI exposure and storage, not an engine change.

### Membership is not assignment

The three-state combo (`estimate here` / `hold from window` / `not in the
model`) fused two decisions the estimator has always treated as orthogonal:
`--model` says which terms are in the design matrix, the stage plan says
where each is estimated. Fusing them made one thing inexpressible — a group
estimated in TWO stages, which the nuisance rule requires, because the clean
stage frees `secular` even when the kept value comes from later.

They are now separate controls, and the old states map onto the new pair
exactly, which is the evidence the split factored what was already there:

| old state | membership | assignment |
|---|---|---|
| `not in the model` | out | — |
| `hold from window` | in | the FIRST stage (so later ones hold it) |
| `estimate here` | in | the LAST stage |

`migrate_group_states` is that table, and it is pure and takes the stage
names as arguments. It has to: reading them off a not-yet-built card list
collapsed `first` and `last` onto one stage, putting a held group and a
freely estimated one in the same place.

Holds are **derived from card order** (`compose_drafts`), never set: a group
a stage does not estimate is held at whatever the last earlier stage fitted.
A settable hold would be a second place deciding one thing.

### One windowed stage is a real fit

The two-stage default refused a plan that held everything, on the grounds
that staging exists to carry something across the span. Once the preset
opened out into N stages, RHOF — no step, no transient — laid out ONE
windowed stage, and that refusal **broke the invariant**: `build_stage_plan`
accepts the single-stage spelling, so the window said refused while the
emitted command fitted perfectly well. It is also the right fit: the model
is estimated inside the window and evaluated across the span, which is what
holding the background from a quiet window means when there is nothing else
to estimate. The refusal now fires only for ≥ 2 stages with no holds, where
the earlier stages genuinely did no work.

### The peel follows the active stage

Selecting stage k plots data minus stages 1…k−1 — exactly what stage k is
fitted to. The operator's own description of the workflow turned out to be a
*specification* of the display rule.

`group_contribution` is the arithmetic. The model is linear in every
parameter it solves for (τ and the step epochs are fixed inputs), so zeroing
the other groups' coefficients and evaluating gives those terms exactly —
verified as a decomposition, not an approximation: the groups sum back to
the whole model with **max error 0.0** on a deployed record. Classification
is delegated (`TrajectoryModel.group_mask` for v2, `group_parameter_mask`
for v1) and is the STAGED vocabulary; the apply-time one folds step
amplitudes into `secular` and would remove the very step the next stage
exists to estimate. Only the step tail is decided locally, and by
construction: `to_record` APPENDS `step_amp_k`, so no classifier sees them.

The card's group checkbox is the one control — it decides both the hold and
what the plot subtracts. `cb_detrend` remains view-only and still subtracts
everything.

### Fitting the screened epochs needed no new flag

`--stages S1,S2` already means "flag nothing": S1/S2 are structural and
always run, so naming only those turns despike, global, window and
protection all off. It also settles the abort question — with no candidates
there is no fraction to exceed, and an explicit stage set is an operator
override the S0 fallback never second-guesses. `n_rejected` goes to zero,
which is the honest answer.

Drawing the grey points is a separate, view-only control. Same masks, same
counts, same record.

### `--final joint`, and why the commit mode is forced

Staging identifies a model; it does not automatically report one. Every
stage after the first conditions on earlier values treated as KNOWN, so its
uncertainties are conditional and the covariance between a held group and a
free one is absent. `--final joint` re-fits the identified structure with
every group free over the domain, and reports the staged→joint movement
scaled by the joint σ.

No seeding is involved and none is needed: everything solved is linear, so
the joint solve has one minimum. Staging chose the structure, the windows
and τ — that is the whole of its contribution to the numbers.

It changes the **run**, not just the commit. Plotting the staged fit and
committing the joint one would put a figure and a record side by side that
are not the same thing.

**The commit mode is forced by the batch, not chosen.**
`gps-estimate-detrend` RECOMPUTES the record and reads the plan from
`analysis.yaml`, so committing the joint solve while leaving a stage plan
behind means the next batch run rebuilds a STAGED record over the top of it
— no error, no warning, other science. Joint mode therefore stores no plan
and CLEARS a stale one.

Measured on SELF, and the reason the delta report exists: a plan holding
lin+per from a window that *starts after* the 2008 Ölfus coseismic, then
extrapolating back across it, put `step_amp_1` at −0.04 mm where the joint
solve puts it at **−150.74 mm — 1130 σ**, rms 63.9 → 1.97. The staged
partition was claiming a separation the data cannot support, and nothing in
the staged output said so.

### Per-group provenance in the record

A `groups` block records, per term group, its slice of `param_names`, the
window it was estimated on, and whether it came from this station or a
donor. Additive at the current `record_version`, following the `segments`
precedent: `from_record` ignores unknown keys, so the 37 deployed records
stay valid and gain a block when next re-committed. An UNSTAGED record gets
none — for a single fit the answer is derivable from keys already present,
and a written copy is a copy that can drift.

**Donors stay pointers.** `DonorRef` resolves against the donor's current
record at estimation time, deliberately: re-estimating a donor is *meant* to
reach everyone borrowing from it. What was missing was visibility, so the
block records the donor's vintage and a digest of the borrowed coefficients
at commit, and a batch re-run warns when they move — and still uses the new
values. Verified end to end: RHOF holding `periodic` from ALHV, ALHV
re-estimated, the re-run warned naming both digests and RHOF's periodic came
out equal to ALHV's *current* values.

### What is verified

- **Invariant sweep**, 12 configurations × the whole matrix: emitted command
  parsed back, re-estimated, records diffed elementwise. Allowed to differ:
  `refs`, `fitted_at`. Never diffed: which card is expanded, the peel,
  whether grey points are drawn. Negative control — suppressing the
  `--final joint` emission fails 3 of 12 with all 7 parameters mismatched.
- **Batch round-trip**, both commit modes on SELF: parameters reproduce to
  9 decimals; staged stores the plan, joint clears it.
- **Donor round-trip** as above.

### Still open

`stage_plan_to_config`'s docstring claims a stored record's `stage_plan`
always parses back via `stage_plan_from_config`. That is **false for donor
holds**: the record spells them `explicit:donor:STA@<fitted_at>`, which
`parse_hold_spec` refuses. The existing round-trip test only covers `stage:`
holds. Nothing here depends on it, but the claim needs correcting or the
spelling reconciling.

## Cross-station borrowing: re-anchoring + the apply-only plan (2026-08-26)

A `--hold GROUP=store:STA` naming ANOTHER station used to pin this station's
LEVEL to the donor's: the stored `offset` is the intercept at t = 0 in
absolute fractional years, so it is wildly station-specific. Measured on SENG
holding SKSH's s(t) over [2015.5, 2019.9]: mean residual (−2.9, −30.1,
+41.5) mm N/E/U — pure datum error that lands in whatever is free.
`geo_dataread.resolve_stage_plan` now RE-ANCHORS such a borrow at resolution
time: only the offset is replaced (found by NAME, never position) with the
1/σ²-weighted mean of the borrower's own residual against the datum-free
donor model; rate and every seasonal coefficient stay the donor's verbatim.

- **`--anchor-window START,END`** (workbench only) picks the averaging
  window; absent, the full fit span is used. The window actually used shows
  in the summary's per-group provenance line and is stored in the record —
  `store:STA@<fitted_at> anchored [START,END]`, distinguishable both from a
  verbatim `store:` hold and from `donor:`. A window selecting zero epochs is
  refused, naming the station. `store:self` is untouched, byte-identical.
- **Apply-only stage** `--stage apply:` (empty free list) + holds for every
  group: the fully-borrowed station (ELDC, THOB — no pre-unrest data).
  Nothing is fitted, `held_covariance="applied"`, covariance all-zero, and
  the record carries `borrowed={from, terms: "all", donor_fitted_at}`.
- The pre-flight existence check is `check_stage_plan_sources` — it verifies
  pointers before any data is read; the series-needing refusal stays on
  `resolve_stage_plan`, whose values are actually used.
- `gps-estimate-detrend` cannot run store: holds at all today (its `main`
  wires no `lookup_secular`), so a committed store-borrow plan re-runs only
  through the workbench until that gap is closed.

## Background borrow — the station with no clean interval (2026-08-26)

The background phase asks "which intervals are clean?". For a station
installed after the deformation started that question has no answer, and
until now the phase had nothing to offer it. Measured on the Svartsengi
cluster (band ELDC→SUDV, lat 63.80–63.95): **18 stations, and only SENG and
SKSH have any data from before the 2020 unrest** — ELDC and THOB start 2021
(bar two stray 2015 epochs), the other 14 were installed in 2024.

`or borrow s(t) from:` in the background box takes a donor code and emits
`borrow_command`:

```
gps-detrend-workbench THOB --stage apply: \
  --hold apply:secular=store:SVAR_NOAM --hold apply:periodic=store:SVAR_NOAM
```

THOB fitting its own background:  `rate [138.62, -494.49, -33.47] mm/yr`
THOB borrowing `SVAR_NOAM`:       `rate [  0.11,    7.41, -16.75] mm/yr`

The first is the unrest, not a background. The donor's datum is NOT carried:
the offset is re-anchored to THOB's own level (see the re-anchoring section),
so only rate and seasonal cross.

**One donor for all three components, by contract.** A secular velocity is a
single 3-vector; north-from-A / east-from-B is not a velocity. The terms
tickboxes select which term GROUPS are borrowed (both = apply-only, nothing
estimated here; unticking one frees it to be fitted locally) — groups, never
components.

**The plate-frame refusal.** `getData(ref="plate")` removes a PER-STATION
plate model, and the Svartsengi assignment is mixed: SENG/SKSH/ELDC/THOB and
9 others are NOAM, while GRIV/AUSV/VMOS/SUDV and 3 more are EURA. Measured on
SENG's own series through both models, **EURA − NOAM = N +2.26, E −15.97
mm/yr** — the size of the deformation these stations are watched for. A
crossed frame would not look like an error, it would look like an intrusion,
so a `derived` store entry carries a `frame` and the lookup refuses to cross
it, in the panel rather than at fit time.

The deployed store carries two cluster backgrounds (`SVAR_NOAM`,
`SVAR_EURA`), the inverse-variance mean of SENG's and SKSH's pre-2020
backgrounds with SKSH's seasonal — SENG's horizontal annual is 2–3× every
other station in the region and is plausibly the plant's production cycle.
Half-separation between the two donors is N 0.14 / E 2.89 / U 1.24 mm/yr:
the honest error on any station borrowing it, and small against a constant
rate offset's effect on change detection (a velocity CHANGE is a change in
slope, which a constant slope error does not hide).

### The anchor is the whole point (2026-08-26)

The donor supplies rate and seasonal; the **constant is this station's own**
and has to come from somewhere in this station's series. In borrow mode the
picked intervals therefore mean ANCHOR, not fit domain — nothing is fitted,
so an interval can only mark where the borrowed curve sits level with the
data. The domain is deliberately NOT narrowed: the whole series stays drawn
against the borrowed curve, because that departure is what is being read.

**This applies to a FULL borrow only.** In a partial borrow (seasonal from
the donor, line fitted here) the datum was never borrowed, so there is
nothing to re-anchor and the intervals keep their usual meaning — the fit
domain. That case is the Askja manoeuvre and it *needs* the interval
control: `katlafitlong` fits the seasonal on 2001.6-2019.5 and the line on a
different, longer span. A first cut made every interval an anchor as soon as
a donor was named, which left no way to say where the line is fitted.
Verified on SKSH: `--segment 2013.9:2019.9` gives rate 0.177/9.321/-18.069,
`--segment 2015.5:2019.9` gives 0.222/9.200/-17.210, and the emitted command
reproduces each.

Default is the full fit span, and on a station deforming throughout its
record that mean is not a datum. Measured on THOB borrowing SENG, model minus
data at 2021.25:

| | full span | anchored 2021.0–2021.5 |
|---|---|---|
| north | **+295.9 mm** | −0.6 mm |
| east | **−966.3 mm** | −0.2 mm |
| up | −80.8 mm | +1.3 mm |

Unanchored, the curve floats at the mean of a series that moved metres. The
panel says so when no interval is picked rather than leaving it to be noticed
on the plot.

### `store:` vs `donor:` — two stores, one datum problem (2026-08-26)

The two hold kinds read different objects and that distinction is real:

| | reads | holds | stations |
|---|---|---|---|
| `store:STA` | `analysis.yaml` `detrend.secular` | s(t) as a component | 18 |
| `donor:STA` | `detrend_params.json` | the finished f(t); the hold takes only the named group out of it | 53 |

So `donor:` is the one most stations actually have, and a `--hold
secular=store:X` failure on a station that has a record is usually asking for
`donor:X` instead. The lookup's error message now says so.

**Both are re-anchored.** The datum problem belongs to the hold KIND, not the
store: the offset in either object is the level of the station it was fitted
on. Measured, THOB holding SENG through `donor:`, before the fix: +75.3 /
-34.3 / -124.6 mm N/E/U off THOB's own data, with `--anchor-window` silently
ignored on that path. After: -0.5 / -0.2 / +1.3, identical to the `store:`
route. Provenance keeps the kinds distinct (`donor:SENG@… anchored […]`).

The split the fix makes explicit: **the donor's RATE crosses, its DATUM does
not.** Pointer semantics are unchanged — re-estimate a donor and every
borrower's rate follows — but the level is now the borrower's own.

### The legacy manoeuvre, and when to prefer it (detrend-OLAC)

`gps_data_analyses/detrend-OLAC/detrend_test.py::katlafitlong` (lines
422-439) applies parameters between stations without ever transferring a
polynomial:

```python
pb, _  = fittimes(lineperiodic, syearf, sdata, sDdata, p0=p0)   # clean window
ddata  = detrend(yearf, data, Ddata, fitfunc=periodic, p=pb)    # seasonal ONLY
pl1, _ = fittimes(line, ryearf, rdata, rDdata)                  # line, full span
data   = detrend(yearf, data, Ddata, fitfunc=line, p=pl1)       # remove it
```

The load-bearing detail is that the legacy `periodic(x, p0..p5)` **silently
ignores p0 and p1**, so a whole `lineperiodic` vector can be handed to it and
only the seasonal is evaluated. The level and trend are then re-estimated on
the target's own series. That is why `detrend_itrf2008.csv` has no offset
column and why the legacy never had the datum problem: **the seasonal
transfers, the polynomial is always local.**

Our grammar spells it directly:

```
--stage fit:secular --hold fit:periodic=store:<donor>
```

Measured on SKSH borrowing SENG's seasonal versus fitting its own:
rms 2.17/1.84/4.24 vs 2.03/1.73/4.06 mm, rate 0.18/9.32/-18.07 vs
0.20/9.34/-18.11 mm/yr. No anchoring is involved — `periodic` has no DC term,
so the re-anchor pass skips it by construction.

**Prefer this wherever the station's own trend is usable.** It cannot go
wrong: no datum and no rate cross. Holding `secular` as well is for the case
the legacy never had — a station whose local trend IS the deformation
(ELDC free-fits at -494 mm/yr east), and that is what the re-anchoring above
exists for. In the Qt picker the terms tickboxes select between them, and **naming a
donor defaults to seasonal-only** (2026-08-26): `linear` is unticked on the
TRANSITION into borrowing, once. After that the boxes are the operator's — a
deliberate decision to borrow the rate too survives re-editing the donor
field, which it would not if the default reapplied on every edit. Clearing
the donor restores `linear`, because fitting needs it (a background with the
linear term off is a model no `--model` value can express).

### The picker resolves the donor code against both stores (2026-08-29)

The CLI grammar refuses to infer a hold kind — `stage:`, `donor:` and
`store:` produce different provenance, so a bare value is an error. The
picker sits above that grammar and can resolve: it tries the secular store
first (the purpose-built object), falls back to the finished record, and
**spells the kind it found into the emitted command**. Nothing is inferred
downstream; the command still reproduces the figure.

This matters because the two stores are very unevenly populated — on
2026-08-29, 72 stations have a finished record and 37 a saved background.
Typing `HS02` used to dead-end on "has no saved background" even though the
station had a perfectly good record.

| donor | resolves to | emitted |
|---|---|---|
| SENG (both) | secular store | `--hold fit:periodic=store:SENG` |
| HS02 (record only) | finished record, with a note | `--hold fit:periodic=donor:HS02` |
| NOPE (neither) | refused, naming both stores | — |

## ⚠️ The secular store is operator state, not deployable config

`deploy.py` overwrote `~/.config/gpsconfig/analysis.yaml` on **2026-08-27
12:19**, replacing 1310 lines with the repo's 109-line skeleton and
destroying **38 curated secular backgrounds**. The file was in
`SHARED_CONFIGS` and in neither `PROTECTED_CONFIGS` nor `NEVER_WRITE`.

37 were reconstructed (16 from a backup as-saved, 19 rebuilt from
`detrend_params.json` via `secular_from_record`, 2 cluster entries from a
working copy); THOB had no record and was lost. Fidelity of the rebuild,
measured on the 16 present in both sources: 15 bit-identical, DYNC differing
(rate 0.079 mm/yr, seasonal 0.287 mm) because its saved background was fitted
on different segments than its committed record.

`analysis.yaml` is now in `deploy.py`'s `NEVER_WRITE` and out of
`SHARED_CONFIGS`. The lesson generalises: **anything `--save-secular` or
`--commit` merge-writes one station at a time is operator state**, and the
deploy lists have to know it — `detrend_params.json` already did, and the
comment above `NEVER_WRITE` had warned that "safe by omission is precisely
how such a thing breaks two years later".
