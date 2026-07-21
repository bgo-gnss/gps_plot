#!/usr/bin/env bash
# setup-testcfg.sh — build a LAYERED test gpsconfig for local plotting.
#
# Adds the analysis-lane catalogs (+ a slot for detrend_params.json) on top of
# the deployed ~/.config/gpsconfig WITHOUT touching production: the base config
# files are symlinked (read-only originals, NO credential files), the catalogs
# are real copies from the gps-config-data checkout. Non-destructive and
# rebuildable — safe to run repeatedly.
#
# The test config lives OUTSIDE this repo (generated artifact). Override any
# path via env var:
#   GPSCONFIG_SRC  deployed config to layer on   (default ~/.config/gpsconfig)
#   GCD            gps-config-data checkout       (default ~/git/gps-config-data)
#   TESTCFG        output test-config dir         (default ~/gps-data/testcfg)
#
# Prereq: the analysis-lane catalogs (steps/protect_windows/outlier_overrides/
# segment_exclusions/fit_windows) must exist in $GCD/analysis-lane — i.e. the
# Stage-B gps-config-data branch is checked out (or merged).
set -euo pipefail

GPSCONFIG_SRC="${GPSCONFIG_SRC:-$HOME/.config/gpsconfig}"
GCD="${GCD:-$HOME/git/gps-config-data}"
TESTCFG="${TESTCFG:-$HOME/gps-data/testcfg}"
AL="$GCD/analysis-lane"

# SAFE base files to symlink (never credentials: database.cfg, *cookies*).
BASE_FILES=(stations.cfg postprocess.cfg station-plate station_coord.xyz
            station_areas.yaml agencies.yaml detrend_itrf2008.csv)
# Analysis-lane catalogs copied in as real files (resolver looks in <gpsconfig>).
CATALOGS=(steps.csv protect_windows.csv outlier_overrides.csv
          segment_exclusions.csv fit_windows.csv)

echo ">> building test config: $TESTCFG"
echo "   base (symlink) <- $GPSCONFIG_SRC ; catalogs (copy) <- $AL"
rm -rf "$TESTCFG"; mkdir -p "$TESTCFG"

for f in "${BASE_FILES[@]}"; do
  if [[ -e "$GPSCONFIG_SRC/$f" ]]; then ln -s "$GPSCONFIG_SRC/$f" "$TESTCFG/$f"
  else echo "   WARN: base file missing: $GPSCONFIG_SRC/$f"; fi
done
for f in "${CATALOGS[@]}"; do
  if [[ -e "$AL/$f" ]]; then cp "$AL/$f" "$TESTCFG/$f"
  else echo "   WARN: catalog missing: $AL/$f"; fi
done

echo ">> done. detrend_params.json is NOT created here — generate it with:"
echo "     ESTIMATE=1 ./plot-views.sh <STA> ..."
echo "   (or: gps-estimate-detrend <STA> --tot-dir \$TOT --out $TESTCFG/detrend_params.json)"
