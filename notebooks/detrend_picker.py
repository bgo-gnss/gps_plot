"""Curate a staged detrend plan reactively, then print the command.

Run it:

    cd gps_plot && uv run marimo edit notebooks/detrend_picker.py

Why a notebook and not (only) a GUI: detrend choice is **curation, not
computation** — where the pre-unrest period ends, whether a station supports
periodic terms, whose parameters to borrow. That is *exploratory*, and the
expensive part is not typing ``2001.6``; it is seeing what the fit does when
you move it. Editing a flag, re-running and opening a PDF makes that loop
slow enough that you take fewer looks than you should. Here, moving a bound
re-fits and redraws immediately.

What it does NOT do: fit-and-store. It emits the ``gps-detrend-workbench``
command, and that command is the only path to committed science — so every
refusal the CLI makes still applies, and this notebook cannot store anything
the CLI would have rejected. One grammar, several producers.

The figure is the PRODUCTION one (``detrend_workbench.build_pages``), grey
and gold outlier lanes included, so what you judge here is what the PDF
would show.
"""

import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    import matplotlib

    matplotlib.use("Agg")  # figures are displayed as images, not windows

    # marimo requires each name to be defined in exactly ONE cell -- that
    # strictness is what makes the dataflow analysable, so every import lives
    # here rather than beside its use.
    from geo_dataread.stage_plan import build_stage_plan
    from geo_dataread.term_spec import parse_term_spec
    from gps_plot.detrend_picker import populated_groups, render_command
    from gps_plot.detrend_workbench import (
        build_pages,
        build_record,
        declared_event_epochs,
        summarise,
    )

    return (
        build_pages,
        build_record,
        build_stage_plan,
        declared_event_epochs,
        mo,
        parse_term_spec,
        populated_groups,
        render_command,
        summarise,
    )


@app.cell
def _(mo):
    mo.md("""
    # Detrend stage picker

    Pick a **staged** plan and watch the fit follow. Nothing is stored —
    the command at the bottom is what commits.
    """)
    return


@app.cell
def _(mo):
    station = mo.ui.text(value="SELF", label="station")
    max_gap = mo.ui.number(value=1.5, start=0.1, stop=10.0, step=0.1,
                           label="--max-gap-years")
    uncert = mo.ui.number(value=10.0, start=1.0, stop=50.0, step=1.0,
                          label="--uncert")
    mo.hstack([station, max_gap, uncert])
    return max_gap, station, uncert


@app.cell
def _(build_record, declared_event_epochs, max_gap, station, uncert):
    # Loading is its own cell so it re-runs only when the station or the
    # read-time levers change -- not on every slider nudge.
    sta = station.value.strip().upper()
    record0, yearf, data, sigma, est0 = build_record(
        sta, uncert=uncert.value, max_gap_years=max_gap.value
    )
    seismic, _other = declared_event_epochs(sta)
    span = (float(yearf.min()), float(yearf.max()))
    return data, est0, record0, seismic, sigma, span, sta, yearf


@app.cell
def _(mo, record0, span, sta, populated_groups, term_specs):
    # The groups on offer must describe the model AS CONFIGURED, not the
    # baseline record. Adding a transient puts log_amp/exp_amp into the
    # model, and a stage plan that never frees them is refused outright:
    # "parameters ['log_amp_1'] are never estimated and not held in the
    # final stage". Offering the group is what lets the emitted command run.
    groups = populated_groups(record0) + (("transient",) if term_specs else ())
    mo.md(
        f"**{sta}** — {span[0]:.2f} to {span[1]:.2f} · "
        f"populated groups: `{', '.join(groups)}`"
    )
    return (groups,)


@app.cell
def _(groups, mo, span):
    # THE reactive control. Drag it and the fit below re-runs. A range slider
    # rather than a brush on purpose: the win here is watching the fit move,
    # and a slider delivers that without a second chart in a second visual
    # vocabulary competing with the production figure.
    stage1 = mo.ui.range_slider(
        start=span[0], stop=span[1], step=0.05,
        value=[span[0], min(span[0] + 8.0, span[1])],
        label="stage 1 window", full_width=True, show_value=True,
    )
    free1 = mo.ui.multiselect(
        options=list(groups), value=["secular", "periodic"],
        label="stage 1 estimates",
    )
    free2 = mo.ui.multiselect(
        options=list(groups),
        value=[g for g in ("secular", "step", "transient") if g in groups],
        label="stage 2 estimates (whole domain)",
    )
    hold2 = mo.ui.dropdown(
        options=["(none)"] + [g for g in groups],
        value="periodic" if "periodic" in groups else "(none)",
        label="stage 2 holds, from stage 1",
    )
    mo.vstack([stage1, mo.hstack([free1, free2, hold2])])
    return free1, free2, hold2, stage1


@app.cell
def _(mo, span):
    # Adding a TRANSIENT is the other half of curation, and on a deforming
    # station it is the half that matters: an excess-candidate abort is a
    # MODEL-ADEQUACY problem, so no window and no stage plan fixes it -- only
    # a term that can follow the signal. Watch n_rejected in the summary:
    # [0, 1, 0] means the detector aborted and judged nothing.
    use_term = mo.ui.checkbox(value=False, label="add a transient")
    kind = mo.ui.dropdown(options=["log", "exp"], value="log", label="kind")
    t_tau = mo.ui.slider(
        start=0.1, stop=10.0, step=0.1, value=2.0,
        label="tau [yr]", show_value=True,
    )
    t_epoch = mo.ui.slider(
        start=span[0], stop=span[1], step=0.05,
        value=round((span[0] + span[1]) / 2.0, 2),
        label="onset epoch", show_value=True, full_width=True,
    )
    mo.vstack([mo.hstack([use_term, kind, t_tau]), t_epoch])
    return kind, t_epoch, t_tau, use_term


@app.cell
def _(kind, parse_term_spec, t_epoch, t_tau, use_term):
    term_specs, term_error = [], None
    if use_term.value:
        spec = f"{kind.value}@{round(t_epoch.value, 4)},tau={round(t_tau.value, 3)}"
        try:
            parse_term_spec(spec)   # the CLI grammar is the authority
            term_specs = [spec]
        except ValueError as exc:
            term_error = str(exc)
    return term_error, term_specs


@app.cell
def _(build_stage_plan, free1, free2, hold2, stage1):
    lo, hi = (round(v, 4) for v in stage1.value)
    specs = [f"clean:{','.join(free1.value)}@{lo}:{hi}"]
    holds = []
    if free2.value:
        specs.append(f"long:{','.join(free2.value)}")
        if hold2.value != "(none)":
            holds.append(f"long:{hold2.value}=stage:clean")

    plan, plan_error = None, None
    try:
        plan = build_stage_plan(specs, holds)
    except ValueError as exc:
        # The grammar's refusals are the authority; surface them verbatim
        # rather than second-guessing them here.
        plan_error = str(exc)
    return plan, plan_error


@app.cell
def _(mo, plan_error):
    mo.md(f"⚠️ **{plan_error}**") if plan_error else mo.md("")
    return


@app.cell
def _(build_record, est0, max_gap, plan, plan_error, record0, sta, uncert, term_specs):
    # The fit itself, re-run whenever the plan changes.
    fit_error, record, estimate = None, record0, est0
    if plan is not None and plan_error is None:
        try:
            record, yearf_, data_, sigma_, estimate = build_record(
                sta, uncert=uncert.value, max_gap_years=max_gap.value,
                stage_plan=plan, lookup_donor=None,
                terms_spec=term_specs or None,
            )
        except (RuntimeError, ValueError) as exc:
            # A refused plan is a RESULT, not a crash: rank-deficient stages
            # and empty groups are the estimator telling you the plan cannot
            # be fitted. Show it and keep the previous record on screen.
            fit_error = str(exc)
    return estimate, fit_error, record


@app.cell
def _(fit_error, mo, term_error):
    msg = term_error or fit_error
    mo.md(f"⚠️ **{msg}**") if msg else mo.md("")
    return


@app.cell
def _(
    build_pages,
    data,
    estimate,
    mo,
    record,
    seismic,
    sigma,
    sta,
    summarise,
    yearf,
):
    figs = build_pages(
        sta, record, yearf, data, sigma,
        outliers=getattr(estimate, "outliers", None),
        seismic_events=seismic,
    )
    mo.vstack([mo.md(f"```\n{summarise(record, sta)}\n```"), figs[0], figs[1]])
    return


@app.cell
def _(max_gap, mo, plan, render_command, sta, term_specs, uncert):
    extra = ["--max-gap-years", str(max_gap.value)]
    if uncert.value != 10.0:
        extra += ["--uncert", str(uncert.value)]
    cmd = render_command(sta, plan, extra, terms=term_specs)
    mo.vstack([
        mo.md("## The command"),
        mo.md(f"```bash\n{cmd}\n```"),
        mo.md(
            "_Add `--commit` when the figure convinces you. The workbench "
            "re-parses this line, so every refusal still applies._"
        ),
    ])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
