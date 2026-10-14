#!/usr/bin/env python3
"""Pixel gate over the plot variants, through every command name.

Renders a fixed table of variants (frames, refs, views, detrend terms,
sessions, a baseline, the production ``--ref all --special all`` fan-out)
with a reference source tree through the old ``plot-gps-timeseries`` main,
and with a candidate tree through ``gpsplot ts``, bare ``gpsplot`` and the
old name, then compares every PNG by sha256. Uses the real deployed config
and data, with a fixed ``-e`` so the window does not move; the "Plot
created on" text is frozen by the shims of ``check_pixel_equivalence.py``.

    git archive main | tar -x -C /tmp/main-src
    python bin/check_cli_variants.py --src-a /tmp/main-src --src-b . --out /tmp/gate

Exit 0 when all runs match the reference (first used for the gpsplot
split, 2026-10-14: 43 PNGs identical).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

VARIANTS: dict[str, list[str]] = {
    "rhof90": ["RHOF", "--special", "90d"],
    "rhofplate": ["RHOF", "--ref", "plate", "--special", "year"],
    "sengclean": [
        "SENG", "--ref", "detrend", "--special", "year",
        "--view", "cleaned", "--show-outliers", "5",
    ],
    "sengper": [
        "SENG", "--ref", "detrend", "--detrend-terms", "periodic", "--special", "year",
    ],
    "sengboth": ["SENG", "--session", "both", "-s", "20250101"],
    "vfln": ["VFLN-VFLS", "--show-ends", "--special", "year"],
    "rhof08": ["RHOF", "--ref", "ref_frame", "--frame", "itrf2008", "--special", "full"],
    "prod": ["RHOF", "SENG", "HOFN", "--ref", "all", "--special", "all", "-j", "6"],
    # stepped detrend records (SELF itrf14 + itrf08, HOFN itrf08): the step
    # offsets must stay in --ref detrend and go with --remove-steps
    "selfdet": ["SELF", "--ref", "detrend", "--special", "full"],
    "selfeq": ["SELF", "--ref", "detrend", "--remove-steps", "earthquake", "--special", "full"],
    "selflin": ["SELF", "--ref", "detrend", "--detrend-terms", "linear", "--special", "full"],
    "self08": ["SELF", "--ref", "detrend", "--frame", "itrf2008", "--special", "full"],
    "hofn08": ["HOFN", "--ref", "detrend", "--frame", "itrf2008", "--special", "full"],
    "hofn08eq": [
        "HOFN", "--ref", "detrend", "--frame", "itrf2008", "--remove-steps", "equipment",
        "--special", "full",
    ],
    # baselines through gps_plot.sources.gps: empirical σ, 8 h overlay, detrended
    "vflnemp": ["VFLN-VFLS", "--baseline-sigma", "empirical", "--special", "year"],
    "sengboth-bl": ["SENG-SKSH", "--session", "both", "-s", "20250101"],
    "sengdet-bl": ["SENG-SKSH", "--ref", "detrend", "--special", "year"],
    # OutlierParams objects must reach station_series intact
    "sengop": [
        "SENG", "--view", "cleaned", "--outlier-param", "window_n_sigma=3.0",
        "--show-outliers", "--special", "year",
    ],
    "sengstages": [
        "SENG", "--view", "cleaned", "--stages", "S0,S4", "--outlier-param",
        "window_n_sigma=4.0", "--show-outliers", "--special", "year",
    ],
    # --model (step 3): the stored-model line over plate / detrended series
    "hofnmodel": ["HOFN", "--model", "both", "--special", "full"],
    "sengmodel": ["SENG", "--model", "curve", "--special", "year"],
    "selfmodel": [
        "SELF", "--model", "both", "--remove-steps", "earthquake", "--special", "full",
    ],
    "selfresid": ["SELF", "--model", "residual", "--view", "cleaned", "--special", "year"],
}  # fmt: skip

#: candidate-only variants (no reference on main): each must render exactly
#: the given number of PNGs, so a caught per-station error cannot pass
NEW_VARIANTS: dict[str, tuple[list[str], int]] = {
    "selfterms": (
        ["SELF", "--model", "residual", "--detrend-terms", "periodic",
         "--model-terms", "linear,periodic,steps", "--special", "full"], 1,
    ),
}  # fmt: skip

OLD = (
    "import sys\nfrom gps_plot.plot_gps_timeseries import main\n"
    "sys.argv = ['plot-gps-timeseries'] + sys.argv[1:]\nmain()\n"
)
GPSPLOT = "import sys\nfrom gps_plot.cli import main\nsys.exit(main({}sys.argv[1:]))\n"
CANDIDATES = {
    "gpsplot-ts": GPSPLOT.format("['ts'] + "),
    "gpsplot": GPSPLOT.format(""),
    "old-name": OLD,
}


def _shims() -> tuple[str, str]:
    spec = importlib.util.spec_from_file_location(
        "cpe", HERE / "check_pixel_equivalence.py"
    )
    assert spec and spec.loader
    cpe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cpe)
    return cpe.SITECUSTOMIZE, cpe.FROZEN_CREATED


def _src(tree: Path) -> Path:
    return tree / "src" if (tree / "src" / "gps_plot").is_dir() else tree


def _render(
    src: Path,
    boot: str,
    out: Path,
    shim: Path,
    frozen: str,
    end: str,
    variants: dict[str, list[str]] = VARIANTS,
) -> None:
    for name, argv in variants.items():
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        env = dict(
            os.environ,
            PYTHONPATH=f"{shim}{os.pathsep}{_src(src)}",
            GPS_PLOT_FREEZE_CREATED=frozen,
            MPLBACKEND="Agg",
        )
        argv = argv + ["-e", end, "--save", "png", "-d", str(d)]
        p = subprocess.run(
            [sys.executable, "-c", boot, *argv], env=env, capture_output=True, text=True
        )
        if p.returncode:
            print(f"{out.name}/{name}: exit {p.returncode}\n{p.stderr[-600:]}")


def _hashes(d: Path) -> dict[str, str]:
    return {
        p.relative_to(d).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(d.rglob("*.png"))
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--src-a", type=Path, required=True, help="reference tree")
    ap.add_argument("--src-b", type=Path, required=True, help="candidate tree")
    ap.add_argument("--out", type=Path, required=True, help="work dir (emptied)")
    ap.add_argument("--end", default="20261010", help="-e for every variant")
    args = ap.parse_args()

    sitecustomize, frozen = _shims()
    shim = args.out / "shim"
    shim.mkdir(parents=True, exist_ok=True)
    (shim / "sitecustomize.py").write_text(sitecustomize)

    _render(args.src_a, OLD, args.out / "reference", shim, frozen, args.end)
    ref = _hashes(args.out / "reference")
    # a driver error is caught per station (exit 0, no file), so an empty
    # variant would otherwise pass as "nothing differs"
    empty = [v for v in VARIANTS if not any((args.out / "reference" / v).glob("*.png"))]
    if not ref or empty:
        print(f"FAIL: the reference rendered nothing for {empty or 'any variant'}")
        return 2
    (args.out / "reference_hashes.txt").write_text(
        "".join(f"{h}  {n}\n" for n, h in ref.items())
    )
    status = 0
    for name, boot in CANDIDATES.items():
        _render(args.src_b, boot, args.out / name, shim, frozen, args.end)
        got = _hashes(args.out / name)
        bad = sorted(
            (set(ref) ^ set(got)) | {n for n in set(ref) & set(got) if ref[n] != got[n]}
        )
        print(f"{name}: " + (f"OK, {len(got)} PNGs" if not bad else f"FAIL {bad}"))
        status |= bool(bad)
    new = args.out / "new"
    _render(
        args.src_b, CANDIDATES["gpsplot-ts"], new, shim, frozen, args.end,
        {k: v[0] for k, v in NEW_VARIANTS.items()},
    )  # fmt: skip
    for name, (_argv, want) in NEW_VARIANTS.items():
        n = len(list((new / name).glob("*.png")))
        print(f"new/{name}: " + ("OK" if n == want else "FAIL") + f", {n}/{want} PNGs")
        status |= n != want
    return status


if __name__ == "__main__":
    raise SystemExit(main())
