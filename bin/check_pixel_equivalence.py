#!/usr/bin/env python3
"""Pixel-equivalence gate for the static production plots.

Renders the SAME station batch with two gps_plot source trees (e.g. a git
``main`` export vs the working tree) and asserts the resulting PNGs are
byte-identical (sha256).  Used to prove a performance pass changed
nothing in the rendered output (perf-plot-parallel, 2026-07).

Determinism shims (written into a temp ``sitecustomize.py`` that both
runs -- and every pool worker -- pick up via PYTHONPATH):

- the wall-clock "(Plot created on ... %H:%M ...)" title fragment is the
  ONLY wall-clock text in the figure; it is frozen to a constant so runs
  minutes apart still compare byte-for-byte;
- ``ConfigParser.getPostprocessConfig`` (called by the driver, absent
  from current gps_parser) is aliased to ``getPostProcessDir`` --
  identically for both runs (pre-existing breakage, not part of the
  comparison).

The interpreter running this script must have the production deps
importable (geo_dataread, gps_parser, geofunc, gtimes, matplotlib + TeX);
gps_plot itself is taken from the two source trees.  Data/config come
from e.g. the geo_dataread golden-master fixtures::

    python bin/check_pixel_equivalence.py \\
        --src-a /tmp/main-export --src-b . \\
        --data-dir ../geo_dataread/tests/goldenmaster/fixtures/TOT \\
        --config-dir /tmp/hermetic-gpsconfig \\
        --stations DYNG ELDC SENG --end 20241115 --workers-b 6

(the config dir needs [PATHS] totDir/totPath/figDir + [FILES] platefile
pointing at the fixture set -- see the goldenmaster README).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path

FROZEN_CREATED = "(Plot created on Jul 11 2026 12:00 GMT)"

SITECUSTOMIZE = '''\
"""Determinism shims for the pixel-equivalence gate (see the check script)."""
import os

if os.environ.get("GPS_PLOT_FREEZE_CREATED"):
    _frozen = os.environ["GPS_PLOT_FREEZE_CREATED"]
    import gtimes.timefunc as _tf

    _tf.currTime = lambda String="": _frozen

if os.environ.get("GPS_PLOT_PARSER_SHIM"):
    import gps_parser as _cp

    if not hasattr(_cp.ConfigParser, "getPostprocessConfig"):
        _cp.ConfigParser.getPostprocessConfig = _cp.ConfigParser.getPostProcessDir
'''

BOOTSTRAP = """\
import sys
from gps_plot.plot_gps_timeseries import main

sys.argv = ["plot-gps-timeseries"] + sys.argv[1:]
main()
"""


def _src_dir(tree: Path) -> Path:
    """Accept either a repo root or its ``src`` dir."""
    return tree / "src" if (tree / "src" / "gps_plot").is_dir() else tree


def _run(
    src: Path,
    outdir: Path,
    shim_dir: Path,
    args: argparse.Namespace,
    extra: list[str],
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    env = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join([str(shim_dir), str(_src_dir(src))]),
        GPS_PLOT_FREEZE_CREATED=FROZEN_CREATED,
        GPS_PLOT_PARSER_SHIM="1",
        GPS_CONFIG_PATH=str(args.config_dir),
        MPLBACKEND="Agg",
    )
    argv = (
        args.stations
        + ["--ref", args.ref, "--special", args.special]
        + ["-e", args.end, "--save", "png"]
        + ["-d", str(outdir), "-i", str(args.data_dir) + os.sep]
        + extra
    )
    subprocess.run(
        [sys.executable, "-c", BOOTSTRAP] + argv,
        env=env,
        check=True,
        stdout=subprocess.DEVNULL,
    )


def _hashes(outdir: Path) -> dict[str, str]:
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(outdir.glob("*.png"))
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--src-a", type=Path, required=True, help="reference source tree")
    ap.add_argument("--src-b", type=Path, required=True, help="candidate source tree")
    ap.add_argument("--data-dir", type=Path, required=True, help="mb_*.dat{1,2,3} dir")
    ap.add_argument("--config-dir", type=Path, required=True, help="gpsconfig dir")
    ap.add_argument("--stations", nargs="+", required=True)
    ap.add_argument("--ref", default="plate")
    ap.add_argument("--special", default="all")
    ap.add_argument("--end", default="20241115", help="-e passed to the driver")
    ap.add_argument("--workers-b", type=int, default=None, help="-j for the B run")
    ap.add_argument("--keep", type=Path, default=None, help="keep outputs under here")
    args = ap.parse_args()

    workdir = args.keep or Path(tempfile.mkdtemp(prefix="gps-plot-pixel-"))
    shim_dir = workdir / "shim"
    shim_dir.mkdir(parents=True, exist_ok=True)
    (shim_dir / "sitecustomize.py").write_text(SITECUSTOMIZE)

    print(f"work dir: {workdir}")
    _run(args.src_a, workdir / "out-a", shim_dir, args, extra=[])
    extra_b = ["-j", str(args.workers_b)] if args.workers_b else []
    _run(args.src_b, workdir / "out-b", shim_dir, args, extra=extra_b)

    a, b = _hashes(workdir / "out-a"), _hashes(workdir / "out-b")
    bad = sorted(
        (set(a) ^ set(b)) | {name for name in set(a) & set(b) if a[name] != b[name]}
    )
    if not a or not b:
        print("FAIL: one of the runs produced no PNGs")
        return 2
    if bad:
        print(f"FAIL: {len(bad)}/{len(a | b)} files differ or are unmatched:")
        for name in bad:
            print(
                f"  {name}: A={a.get(name, '<missing>')[:12]} "
                f"B={b.get(name, '<missing>')[:12]}"
            )
        return 1
    print(f"OK: {len(a)} PNGs byte-identical (sha256)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
