#!/usr/bin/env bash
# view-plots.sh — look at the `gpsplot run` plot tree in a local aflogun.
#
# Serves the plot tree (laid out like the CDN) with python's http.server and
# starts aflogun pointed at it through the AFLOGUN_GRAPHS_*_URL environment
# overrides (aflogun branch feat/local-plots) — the deployed postprocess.cfg is
# never edited. Everything else (time series, maps) still comes from the
# deployed config. Ctrl-C stops both.
#
# Usage:
#   gpsplot run                         # fill the tree first (plots.yaml)
#   ./view-plots.sh                     # then open http://localhost:8050
#
# Env overrides:
#   PLOTS_DIR  plot tree         (default ~/gps-data/plots)
#   PORT       plot server port  (default 8070)
#   AFLOGUN    aflogun checkout  (default ../../../.worktrees/aflogun-local-plots)
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PLOTS_DIR="${PLOTS_DIR:-$HOME/gps-data/plots}"
PORT="${PORT:-8070}"
AFLOGUN="${AFLOGUN:-$HERE/../../../.worktrees/aflogun-local-plots}"

[ -d "$PLOTS_DIR" ] || { echo "no plot tree at $PLOTS_DIR — run gpsplot run first" >&2; exit 1; }
[ -d "$AFLOGUN" ] || { echo "no aflogun checkout at $AFLOGUN (branch feat/local-plots)" >&2; exit 1; }

python3 -m http.server -d "$PLOTS_DIR" "$PORT" >/dev/null 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null' EXIT
echo "plots: http://localhost:$PORT/  ($PLOTS_DIR)"

export AFLOGUN_GRAPHS_STANDARD_URL="http://localhost:$PORT/timeseries/graphs/standard_24h"
export AFLOGUN_GRAPHS_VOLCANOS_URL="http://localhost:$PORT/locations/volcanos"
echo "aflogun: http://localhost:8050/"
cd "$AFLOGUN" && uv run aflogun
