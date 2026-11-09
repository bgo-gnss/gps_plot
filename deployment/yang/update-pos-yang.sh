#!/bin/bash
# update-pos-yang.sh: bring yang's ITRF2014 .pos series up to date from okada's
# daily SINEX (read-only NFS). The laptop's update_pos.sh also runs GAMIT's
# tssum for the ITRF2008 set; the plots use only imo.24hr_itrf14, so yang runs
# just gps-realign and the icing harvest and needs no GAMIT.
#
#   update-pos-yang.sh [--days N] [--dry-run]
#
# --days: re-make the last N days (default 40: okada re-makes D-3..D-36), and
# anything after the product's last day if that is older. Seed $OUT from the
# laptop first (seed-from-laptop.sh): a missing product is refused here, as a
# rebuild from 1995 is not what a timer should start.
set -euo pipefail
: "${YANG_ROOT:=/mnt/data/gpsops}"
SRC=/mnt_data/okada/D/SNX OUT=$YANG_ROOT/pos DAYS=40 DRY=0
PROD=imo.24hr_itrf14
while [ $# -gt 0 ]; do case $1 in
    --days) DAYS=$2; shift ;;
    --dry-run) DRY=1 ;;
    -h|--help) sed -n 2,12p "$0"; exit 0 ;;
    *) echo "bad argument: $1" >&2; exit 64 ;; esac; shift; done

timeout 10 test -d "$SRC" || { echo "update-pos: source $SRC not reachable" >&2; exit 1; }
csv=$OUT/days.$PROD.csv
[ -s "$csv" ] || { echo "update-pos: no $csv (run seed-from-laptop.sh first)" >&2; exit 1; }

today=$(date -u +%F)
start=$(date -u -d "-$DAYS day" +%F)
last=$(tail -n 1 "$csv" | cut -d, -f1)
[[ "$last" < "$start" ]] && start=$(date -u -d "$last +1 day" +%F)
echo "realign igb14: update $start … $today"
[ $DRY -eq 0 ] || exit 0

log=$OUT/realign.$PROD.log
if ! gps-realign --frame igb14 --update --start "$start" --end "$today" \
        --archive "$SRC" --out "$OUT" > "$log" 2>&1; then
    tail -n 5 "$log" >&2; echo "realign: gps-realign failed" >&2; exit 1
fi
grep -v '^[0-9]*/[0-9]* days$' "$log" || true

# the icing/snow indices; the series above are fine without them
gps-autcln-harvest --out "$OUT" || echo "autcln: harvest failed (icing index not updated)" >&2
