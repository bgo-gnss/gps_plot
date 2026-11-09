#!/bin/bash
# yang-gis-sync.sh: send changed GIS data to yang. Run on the laptop, where the
# layers are built and checked, BEFORE merging a map_layers.yaml that uses a new
# file (the config reaches yang within 10 min, and the sync refuses a missing
# layer). Only changed files travel; nothing on yang is deleted.
#
#   yang-gis-sync.sh [-n]        -n: show what would be sent
set -euo pipefail

HOST=yang-d01.vedur.is
GIS=/mnt/data/gpsops/gis
RSYNC=(rsync -rltO --partial --no-owner --no-group --itemize-changes --chmod=D2775,F664)
[ "${1:-}" = -n ] && RSYNC+=(-n)

"${RSYNC[@]}" "$HOME/gps-data/gis/" "$HOST:$GIS/"
# without -p rsync applies the remote umask
[ "${1:-}" = -n ] || ssh "$HOST" "cd $GIS && find . -user \$(id -un) ! -perm -g+w -exec chmod g+w {} +"
