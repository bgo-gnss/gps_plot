"""Pick a staged-estimation plan off the figure, and print the command.

The graphical half of the staged-estimation surface.  Its contract, from the
program plan, is one sentence: **it writes the same grammar the CLI parses —
one grammar, two producers.**  So this module builds a
:class:`geo_dataread.stage_plan.StagePlan` and renders it back into
``--stage`` / ``--hold`` flags; it never fits, never stores, and never
reimplements a refusal.  Everything the workbench refuses, it still refuses,
because the workbench is the only path to committed science.

Why a picker at all: a fit window is a *visual* judgement — where the
pre-unrest period ends, where a transient starts, which gap is real.  Typing
``2001.6:2008.0`` is transcription, not judgement, and transcription is where
the mistakes are.

Division of labour, deliberately hybrid:

* **Spans are picked graphically.** Dragging on the series is the part you
  must SEE.
* **Term groups and holds are chosen textually.** Which groups a stage
  estimates, and whose value to hold, is the part you must THINK about — a
  prompt that lists the station's actually-populated groups beats hunting for
  a checkbox, and it degrades gracefully over ssh.

The figure is drawn with the production plotting primitive
(``timesmatplt.stdTimesPlot``) plus the workbench's own event lines, so the
series, its scaling and the declared-event vocabulary are identical to the
PDF you would judge.  The grey/gold outlier lanes are NOT drawn here: they
live in closures inside ``detrend_workbench.render`` and hoisting them is a
refactor of the production path, which is not worth the regression risk to
pick a window.  See "Known gaps" below.

Known gaps (v1):
    * Outlier lanes are absent (see above) — pick against the PDF when the
      distinction matters.
    * ``--term`` (adding a transient) is not offered, because the CLI has no
      such flag yet; only groups the station's model already has are
      selectable.
"""

from __future__ import annotations

import shlex
import sys
from collections.abc import Sequence
from typing import Any

__all__ = ["pick_stage_plan", "populated_groups", "render_command", "main"]

#: Shading for accepted stage spans, cycled in declaration order.
_SPAN_COLORS = ("#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3")
_SPAN_ALPHA = 0.16


def populated_groups(record: dict[str, Any]) -> tuple[str, ...]:
    """Term groups this station's fitted model actually has parameters for.

    Public because the marimo notebook consumes it -- and marimo treats
    leading-underscore names as cell-private, so a shared helper cannot have
    one.

    Offering a group the model lacks would produce a plan the estimator
    refuses (an empty mask is a silent no-op, which is why it refuses), so
    the picker only ever offers what is addressable.
    """
    import numpy as np
    from gps_analysis import GROUP_ORDER, TrajectoryModel, with_steps
    from gps_analysis.detrend import _resolve_model
    from gps_analysis.staged import group_parameter_mask

    spec = record.get("terms")
    if spec is not None:
        # A record-version-2 record carries its own terms, so read them
        # rather than reconstructing from a model code -- a code plus step
        # epochs cannot express a transient, and offering "transient" only
        # when it is really there is what keeps the picker honest.
        model = TrajectoryModel.from_spec(spec).as_modelfunc()
    else:
        base, _ = _resolve_model(str(record["model"]))
        steps = np.asarray(record.get("step_epochs") or [], dtype=float).ravel()
        model = with_steps(base, steps) if steps.size else base
    return tuple(g for g in GROUP_ORDER if group_parameter_mask(model, g).any())


def _ask(prompt: str, valid: tuple[str, ...], *, allow_empty: bool = False) -> str:
    """Prompt until the answer is one of ``valid`` (or empty, if allowed)."""
    while True:
        raw = input(prompt).strip()
        if not raw and allow_empty:
            return ""
        if raw in valid:
            return raw
        print(f"  expected one of {', '.join(valid)}", file=sys.stderr)


def _ask_groups(prompt: str, populated: tuple[str, ...]) -> tuple[str, ...]:
    """Prompt for a comma-separated group list, validated against the model."""
    while True:
        raw = input(prompt).strip()
        if not raw:
            print("  a stage that estimates nothing is not a stage", file=sys.stderr)
            continue
        got = tuple(g.strip() for g in raw.split(",") if g.strip())
        bad = [g for g in got if g not in populated]
        if bad:
            print(
                f"  {bad} not available for this station; "
                f"populated groups: {', '.join(populated)}",
                file=sys.stderr,
            )
            continue
        return got


def pick_stage_plan(
    sta: str,
    yearf: Any,
    data: Any,
    sigma: Any,
    record: dict[str, Any],
    *,
    seismic_events: Any = None,
    tos_events: Any = None,
) -> Any:
    """Show the series, collect spans by dragging, then build a StagePlan.

    Returns the plan, or None if the operator quit without picking.  Raises
    RuntimeError when the matplotlib backend cannot show a window, because a
    picker that silently draws to a file is useless and easy to mistake for a
    hung process.
    """
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.widgets import SpanSelector

    from geo_dataread.stage_plan import build_stage_plan

    from gps_plot import timesmatplt as tplt
    from gps_plot.detrend_workbench import (
        SEISMIC_COLOR,
        TOS_COLOR,
        add_event_lines,
    )

    backend = matplotlib.get_backend()
    if backend.lower() in {"agg", "pdf", "svg", "ps", "template"}:
        raise RuntimeError(
            f"matplotlib backend {backend!r} cannot show a window, so there is "
            f"nothing to pick on. Run locally with an interactive backend "
            f"(TkAgg/QtAgg) — e.g. MPLBACKEND=TkAgg — or use the CLI's "
            f"--stage/--hold directly."
        )

    x = np.asarray(yearf, dtype=float)
    fig = tplt.stdTimesPlot(
        x,
        np.asarray(data, dtype=float),
        sigma,
        Title=tplt.make_title(sta, x[-1], ref="Plate (picker)"),
    )
    if tos_events:
        add_event_lines(fig, tos_events, TOS_COLOR)
    if seismic_events:
        add_event_lines(fig, seismic_events, SEISMIC_COLOR)

    spans: list[tuple[float, float]] = []
    patches: list[list[Any]] = []

    def _shade(lo: float, hi: float) -> None:
        colour = _SPAN_COLORS[len(spans) % len(_SPAN_COLORS)]
        drawn = [
            ax.axvspan(lo, hi, color=colour, alpha=_SPAN_ALPHA, zorder=0)
            for ax in fig.axes
        ]
        fig.axes[0].text(
            (lo + hi) / 2.0,
            fig.axes[0].get_ylim()[1],
            f"stage {len(spans)}",
            ha="center",
            va="top",
            fontsize=8,
            color=colour,
        )
        patches.append(drawn)
        fig.canvas.draw_idle()

    def _on_select(lo: float, hi: float) -> None:
        if hi - lo <= 0:
            return
        spans.append((round(lo, 4), round(hi, 4)))
        _shade(lo, hi)
        print(f"  stage {len(spans)}: {spans[-1][0]} : {spans[-1][1]}")

    def _on_key(event: Any) -> None:
        if event.key in ("u", "backspace") and spans:
            spans.pop()
            for art in patches.pop():
                art.remove()
            fig.canvas.draw_idle()
            print(f"  undone; {len(spans)} span(s) left")
        elif event.key in ("enter", "d"):
            plt.close(fig)
        elif event.key == "q":
            spans.clear()
            plt.close(fig)

    selector = SpanSelector(
        fig.axes[0],
        _on_select,
        "horizontal",
        useblit=True,
        props={"alpha": 0.3, "facecolor": "tab:blue"},
        interactive=True,
        drag_from_anywhere=True,
    )
    # Both must be kept alive: matplotlib holds only a weak reference to a
    # widget, and the key handler is useless unconnected -- which is exactly
    # what it was until a review caught it, silently making u/enter/q no-ops.
    fig._picker_selector = selector  # type: ignore[attr-defined]
    fig.canvas.mpl_connect("key_press_event", _on_key)

    print(
        f"\n{sta}: drag on the TOP panel to add a stage window.\n"
        f"  u / backspace  undo the last span\n"
        f"  enter / d      done\n"
        f"  q              quit without picking\n"
        f"A stage with NO span inherits the caller's fit domain — add it by\n"
        f"answering the prompts after you close the window.\n"
    )
    plt.show()

    if not spans:
        return None

    populated = populated_groups(record)
    print(f"\npopulated term groups for {sta}: {', '.join(populated)}\n")

    stage_specs: list[str] = []
    hold_specs: list[str] = []
    names: list[str] = []
    for i, (lo, hi) in enumerate(spans, start=1):
        default = "clean" if i == 1 else f"stage{i}"
        name = input(f"stage {i} [{lo}:{hi}] name [{default}]: ").strip() or default
        names.append(name)
        free = _ask_groups(f"  {name} estimates (comma list): ", populated)
        stage_specs.append(f"{name}:{','.join(free)}@{lo}:{hi}")

    if (
        _ask(
            "\nadd a final stage over the whole fit domain (the long-span "
            "re-fit)? [y/N]: ",
            ("y", "n", "Y", "N"),
            allow_empty=True,
        ).lower()
        == "y"
    ):
        name = input("  name [long]: ").strip() or "long"
        free = _ask_groups(f"  {name} estimates (comma list): ", populated)
        stage_specs.append(f"{name}:{','.join(free)}")
        names.append(name)

    # holds: only groups an EARLIER stage actually freed can be held from it,
    # which is the estimator's own rule -- surfaced here as the offer, so an
    # impossible hold is never typed.
    print()
    for j, name in enumerate(names):
        earlier = names[:j]
        if not earlier:
            continue
        raw = input(
            f"hold anything in {name!r}? "
            f"GROUP=stage:NAME or GROUP=donor:STA (blank for none): "
        ).strip()
        if raw:
            hold_specs.append(f"{name}:{raw}")

    try:
        return build_stage_plan(stage_specs, hold_specs)
    except ValueError as exc:
        # The grammar's refusals are the authority; the picker just relays.
        print(f"\nthat plan is not valid: {exc}", file=sys.stderr)
        return None


def render_command(
    sta: str,
    plan: Any,
    extra: list[str] | None = None,
    terms: Sequence[str] | None = None,
) -> str:
    """Render a plan back into the exact ``gps-detrend-workbench`` invocation.

    The round trip that makes "one grammar, two producers" true rather than
    aspirational: this output is parsed by the very same
    :func:`geo_dataread.stage_plan.build_stage_plan` that produced the plan.
    """
    from geo_dataread.stage_plan import StageRef

    parts = ["gps-detrend-workbench", sta]
    for term in terms or ():
        parts += ["--term", term]
    for st in plan.stages if plan is not None else ():
        spec = f"{st.name}:{','.join(st.free)}"
        if st.segments is not None:
            spec += "@" + ";".join(
                f"{'' if lo is None else lo}:{'' if hi is None else hi}"
                for lo, hi in st.segments
            )
        parts += ["--stage", spec]
    multi = plan is not None and len(plan.stages) > 1
    for st in plan.stages if plan is not None else ():
        for group, ref in st.held.items():
            value = (
                f"stage:{ref.stage}"
                if isinstance(ref, StageRef)
                else f"donor:{ref.station}"
            )
            lhs = f"{st.name}:{group}" if multi else group
            parts += ["--hold", f"{lhs}={value}"]
    parts += extra or []
    return shlex.join(parts)


def main(argv: list[str] | None = None) -> int:
    import argparse

    from gps_plot.detrend_workbench import build_record, declared_event_epochs

    p = argparse.ArgumentParser(
        prog="gps-detrend-picker",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("station")
    p.add_argument("--tot-dir", default=None)
    p.add_argument("--uncert", type=float, default=10.0)
    p.add_argument("--max-gap-years", type=float, default=None)
    args = p.parse_args(argv)
    sta = args.station.upper()

    try:
        record, yearf, data, sigma, _est = build_record(
            sta,
            tot_dir=args.tot_dir,
            uncert=args.uncert,
            max_gap_years=args.max_gap_years,
        )
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    seismic, _other = declared_event_epochs(sta)
    try:
        plan = pick_stage_plan(sta, yearf, data, sigma, record, seismic_events=seismic)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    if plan is None:
        print("nothing picked.")
        return 1

    extra: list[str] = []
    if args.max_gap_years is not None:
        extra += ["--max-gap-years", str(args.max_gap_years)]
    if args.uncert != 10.0:
        extra += ["--uncert", str(args.uncert)]

    print("\n" + "=" * 70)
    print(render_command(sta, plan, extra))
    print("=" * 70)
    print(
        "\nReview it, then run it. Add --commit when the figure convinces you.\n"
        "The workbench re-parses this line, so every refusal still applies —\n"
        "the picker cannot store something the CLI would have rejected."
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
