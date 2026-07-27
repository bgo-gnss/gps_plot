#!/usr/bin/env bash
# plot-views.sh — local 3-view time-series workflow (raw | cleaned | detrended).
#
# The laptop proof lane against the local GLOBK join output:
#   gps-globk-tot (join -> TOT)  ->  gps-estimate-detrend (-> detrend_params.json)
#   ->  plot-gps-timeseries --view raw|cleaned|detrended
# Publishes nothing. Reads the layered test config from setup-testcfg.sh so the
# deployed ~/.config/gpsconfig is never touched.
#
# Requires the `gpslibrary` mamba env (all packages editable-installed there):
#   mamba activate gpslibrary && ./plot-views.sh ...
#
# Usage:
#   ./plot-views.sh STA [STA2 ...]              # plot given stations, all 3 views
#   ESTIMATE=1 ./plot-views.sh STA ...          # (re)estimate detrend first, then plot
#   SAVE=pdf,png VIEWS="raw detrended" ./plot-views.sh STA
#
# All generated output lives OUTSIDE this repo. Override via env:
#   TESTCFG  test config dir     (default ~/gps-data/testcfg — run setup-testcfg.sh first)
#   TOT_DIR  local join output   (default ~/gps-data/TOT)
#   FIGDIR   figure output dir   (default ~/gps-data/figs)
#   SAVE     save format(s)      (default png; e.g. pdf | eps,pdf,png)
#   VIEWS    views to render     (default "raw cleaned detrended")
#   OUT      detrend params path (default $TESTCFG/detrend_params.json; set it to
#            a scratch file to experiment without touching the network-wide doc)
set -euo pipefail

TESTCFG="${TESTCFG:-$HOME/gps-data/testcfg}"
TOT_DIR="${TOT_DIR:-$HOME/gps-data/TOT}"
FIGDIR="${FIGDIR:-$HOME/gps-data/figs}"
SAVE="${SAVE:-png}"
VIEWS="${VIEWS:-raw cleaned detrended}"
mkdir -p "$FIGDIR"

[[ -d "$TESTCFG" ]] || { echo "!! no test config at $TESTCFG — run ./setup-testcfg.sh first"; exit 1; }
for cmd in gps-estimate-detrend plot-gps-timeseries; do
  command -v "$cmd" >/dev/null || { echo "!! $cmd not on PATH — run 'mamba activate gpslibrary' first"; exit 1; }
done

export GPS_CONFIG_PATH="$TESTCFG"

if [[ "${ESTIMATE:-0}" == "1" ]]; then
  # gps-estimate-detrend REPLACES the whole document — it has no merge mode, so
  # estimating one station drops every other station's stored parameters.
  # Warn loudly rather than silently shrinking the file; ESTIMATE=force skips.
  PARAMS="${OUT:-$TESTCFG/detrend_params.json}"
  if [[ -f "$PARAMS" && -z "${OUT:-}" ]]; then
    have=$(python -c "import json,sys;print(len(json.load(open(sys.argv[1]))['stations']))" "$PARAMS" 2>/dev/null || echo 0)
    if (( have > $# )); then
      echo "!! $PARAMS holds $have stations; estimating $# would drop the rest."
      echo "   Re-estimate them all, or use ESTIMATE=1 OUT=<other.json>. Aborting."
      exit 1
    fi
  fi
  echo ">> estimate detrend for $* -> $PARAMS"
  gps-estimate-detrend "$@" --tot-dir "$TOT_DIR" --out "$PARAMS"
fi

for sta in "$@"; do
  for view in $VIEWS; do
    echo ">> $sta  --view $view  --save $SAVE"
    plot-gps-timeseries "$sta" -i "$TOT_DIR" -d "$FIGDIR" \
        --ref plate --view "$view" --save "$SAVE" 2>&1 | grep -E 'plotted|error' || true
  done
done
echo ">> figures in $FIGDIR:"
ls -1t "$FIGDIR"/*.{png,pdf,eps} 2>/dev/null | sed 's#.*/##' | head -20
