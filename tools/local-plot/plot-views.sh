#!/usr/bin/env bash
# plot-views.sh — local 3-view time-series workflow (raw | cleaned | detrended).
#
# The laptop proof lane against the local GLOBK join output:
#   gps-globk-tot (join -> TOT)  ->  gps-estimate-detrend (-> detrend_params.json)
#   ->  plot-gps-timeseries --view raw|cleaned|detrended
# Publishes nothing. Reads the layered test config from setup-testcfg.sh so the
# deployed ~/.config/gpsconfig is never touched.
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
#   GDR      geo_dataread dir    (default <repo>/../geo_dataread — needs Stage B: gps-estimate-detrend)
#   GPLOT    gps_plot dir        (default <repo> root)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPLOT="${GPLOT:-$(cd "$HERE/../.." && pwd)}"
GDR="${GDR:-$(cd "$GPLOT/../geo_dataread" && pwd)}"
TESTCFG="${TESTCFG:-$HOME/gps-data/testcfg}"
TOT_DIR="${TOT_DIR:-$HOME/gps-data/TOT}"
FIGDIR="${FIGDIR:-$HOME/gps-data/figs}"
SAVE="${SAVE:-png}"
VIEWS="${VIEWS:-raw cleaned detrended}"
mkdir -p "$FIGDIR"

[[ -d "$TESTCFG" ]] || { echo "!! no test config at $TESTCFG — run ./setup-testcfg.sh first"; exit 1; }

if [[ "${ESTIMATE:-0}" == "1" ]]; then
  echo ">> estimate detrend for $* -> $TESTCFG/detrend_params.json"
  ( cd "$GDR" && uv run gps-estimate-detrend "$@" \
      --tot-dir "$TOT_DIR" --out "$TESTCFG/detrend_params.json" )
fi

for sta in "$@"; do
  for view in $VIEWS; do
    echo ">> $sta  --view $view  --save $SAVE"
    ( cd "$GPLOT" && GPS_CONFIG_PATH="$TESTCFG" uv run plot-gps-timeseries "$sta" \
        -i "$TOT_DIR" -d "$FIGDIR" --ref plate --view "$view" --save "$SAVE" ) \
        2>&1 | grep -E 'plotted|error' || true
  done
done
echo ">> figures in $FIGDIR:"
ls -1t "$FIGDIR"/*.{png,pdf,eps} 2>/dev/null | sed 's#.*/##' | head -20
