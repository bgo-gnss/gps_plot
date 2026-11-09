#!/bin/bash
# seed-from-laptop.sh: the data yang cannot get from git, copied once. Run on
# the laptop, as bgo (in group starfsmenn on yang, which owns the tree).
#
#   seed-from-laptop.sh [-n]        -n: rsync dry run
#
# Only reads the laptop. Credentials-only files (database.cfg, receivers.cfg,
# icinga.cfg, grafana_cookies.yaml) are not in the list. stations.cfg is needed
# for the active-station list: it arrives readable by owner and group only.
set -euo pipefail

DEST=yang-d01.vedur.is:/mnt/data/gpsops
ROOT=/mnt/data/gpsops
CONFIG=$HOME/.config/gpsconfig
RSYNC=(rsync -rlt --mkpath --partial --no-owner --no-group --info=stats1)
[ "${1:-}" = -n ] && RSYNC+=(-n)

send() {  # send MODE SRC... DEST: file mode 664 (rewritten on yang) or 640 (config)
    local mode=$1; shift
    "${RSYNC[@]}" --chmod="D2775,F$mode" "$@"
}

send 664 "$HOME/gps-data/gis/" "$DEST/gis/"
send 664 "$HOME/gps-data/pos/" "$DEST/pos/"
send 664 "$HOME/gps-data/velocities/" "$DEST/velocities/"
send 664 "$HOME"/gg/tables/igb14_comb.{apr,eq,stab_site} "$DEST/gg/tables/"
send 664 "$HOME/.cache/gps_analysis/noise/baseline_noise.json" "$DEST/cache/gps_analysis/noise/"

files=(analysis.yaml detrend_itrf2008.csv detrend_params.itrf14.json detrend_status.csv
    events.yaml excluded_epochs.csv fit_windows.csv junction_offsets.csv map_layers.yaml
    outlier_overrides.csv plots.yaml protect_windows.csv segment_exclusions.csv
    station-plate station_areas.yaml station_coord.xyz stations.cfg steps.yaml)
send 640 "${files[@]/#/$CONFIG/}" "$DEST/config/"

# postprocess.cfg with the laptop's paths replaced by yang's
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
sed -e "s|~/gps-data/|$ROOT/|g" -e "s|~/.config/gpsconfig/|$ROOT/config/|g" \
    -e "/^\[PATHS\]/a velocity_dir = $ROOT/velocities" \
    "$CONFIG/postprocess.cfg" > "$tmp/postprocess.cfg"
send 640 "$tmp/postprocess.cfg" "$DEST/config/"
