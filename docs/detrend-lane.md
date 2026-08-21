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
epochs arrive. `--hide-outliers` drops the grey overlay (display only — the
epochs are already masked; the y-axis then tightens to the cleaned series,
often dramatically: RHOF north 90 mm → 45 mm of range) but deliberately does
NOT hide gold: decluttering removes decided outliers, never undecided ones.
`--provisional-days` bounds the recency window (0 disables; default 14 from
`geo_dataread`), and the bound matters — indeterminate clusters also sit at
old mid-series gaps and would otherwise dominate.

```bash
plot-gps-timeseries RHOF --view cleaned --outlier-param window_n_sigma=3.0
plot-gps-timeseries RHOF --view cleaned --outlier-overrides ./overrides.csv
plot-gps-timeseries RHOF --view cleaned --uncert 10 --hide-outliers
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
single window, and it matches `steps.csv`'s own annotation.

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
`steps.csv`'s declared 2008.4085 into two steps for one event. It used to
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
narrows — and `step_epochs=`, fed by `_declared_step_epochs` (steps.csv ∪
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
(`steps.csv` rows whose `kind` is
earthquake/coseismic/seismic, plus `--event YYYYMMDD[,LABEL]`), `royalblue` the
fit. Seismic lines read `steps.csv` via `gps_parser.outlier_catalogs.read_steps`,
**not** `gps_views.station_step_epochs` — that one drops `kind`, so it cannot
separate an earthquake from an antenna swap. No skjálftalísa client exists
anywhere in the ecosystem (planned only, in `analysis.yaml` + the CSV header).

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

That fifth one is the same lever as the first. `steps.csv` is a floor that
`_override_settings` merges in, so on a station with a declared step an
untouched stage plan freed only `secular` while the fit still carried
`step_amp_1` — refused, every time, with *"never estimated and not held in
the final stage"*. Both sides refused identically, so the emitted command
still reproduced the figure; what broke was the **feature**: the stage lane
was unusable on exactly the two stations in `steps.csv` (SELF, HOFN), and
nothing said that re-declaring the already-declared step was the way out.
`free2` now asks `_declared_step_epochs(sta, settings.steps)` — the merged
set, the same function `_override_settings` uses. Conditional, not blanket:
RHOF has no declared step and its plan is byte-identical to before
(`--stage long:secular`).

**One assembly site for the run flags, because there are two pickers.**
`detrend_workbench.run_flags()` builds the `--tot-dir` / `--max-gap-years` /
`--uncert` / `--provisional-days` tail for both. The marimo picker
(`gps-detrend-picker`) had TWO of the four violations above still live after
the Qt picker was fixed — `--tot-dir` read and never emitted, and `--uncert`
as `type=float` emitting `12.0` at a `type=int` parser — while printing
*"the workbench re-parses this line, so every refusal still applies."*
Two pickers assembling the same list is two places to forget the same flag.
`WORKBENCH_UNCERT_DEFAULT` now lives in `detrend_workbench` beside
`BATCH_UNCERT_DEFAULT` and is imported by both; omission is only correct
because it is the workbench's own default, not a number restated.
**Breaking, deliberately:** `gps-detrend-picker --uncert 12.5` used to be
accepted and now hard-errors, because what it accepted it could not emit.
`run_flags` RAISES on a non-integral `uncert` rather than rounding — rounding
is the same bug in disguise, trading a loud argparse refusal for a command
that parses and then reads a different set of epochs than the figure was
fitted on.

Measured 2026-08-17 by driving `PickerWindow` offscreen and diffing the
picker's record against the one the emitted command produces, elementwise:
**11 cases, fitted quantities identical in all of them** (untouched, moved
domain, picked step, `--term`, stage, stage+step, stage+term, RHOF stage
baseline, `--uncert 12`, `--provisional-days`+`--tot-dir`, catalog union).
Two divergences remain and are provenance-only, neither able to reach a
stored record because the picker has no `--commit`: `refs.uncert` is absent
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
un-declaring a step. `steps.csv` is a floor that merges in, and a picked step
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

**Compare unstaged, then adopt** (2026-08-21). Unchecking `stage the fit` to
look at the plain fit is DESTRUCTIVE — the toggle rewrites the group states on
the way out and again on the way back, so peeking costs you the setup.
`compare unstaged` instead fits the same configuration with no stage plan and
draws it as a **magenta dashed overlay** beside the trajectory, printing both
parameter sets to the terminal. The figure, the emitted command and the group
states are all untouched: the blue line remains the only thing the command
describes, which is what keeps the overlay from becoming violation nine.
`adopt comparison` then makes it real, by turning staging off and refitting —
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
