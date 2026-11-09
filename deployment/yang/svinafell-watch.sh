#!/bin/bash
# svinafell-watch.sh: Svínafellsheiði first (BGÓ 2026-11-09). Every 10 min in
# the morning and afternoon: as soon as okada's solution for yesterday holds
# SVIE and SVIN, bring the .pos series up and redraw only Svínafellsheiði.
#
# Takes gpsplot-run.lock and never waits for it: the .pos files are rewritten
# in place, so two writers must not overlap. A busy lock means a whole run is
# going, which draws Svínafellsheiði first anyway; the next tick tries again.
set -uo pipefail
: "${YANG_ROOT:=/mnt/data/gpsops}"

AREA=svinafellsheidi
HERE=$(dirname "$(readlink -f "$0")")
PLOTS=$YANG_ROOT/plots
STATUS=$PLOTS/locations/skridurogflod/svinafellsheidi/maps/svinafellsheidi-hreyfing-yfirlit.status.json
SNX_ROOT=/mnt_data/okada/D/SNX
LOCK=${XDG_RUNTIME_DIR:-/tmp}/gpsplot-run.lock

yesterday=$(date -d yesterday +%F)
shown=$(sed -n 's/.*"newest_day": "\([0-9-]*\)".*/\1/p' "$STATUS" 2>/dev/null | head -1)
if [ -n "$shown" ] && ! [[ "$shown" < "$yesterday" ]]; then
    echo "svinafell-watch: $yesterday already shown"; exit 0
fi
snx="$SNX_ROOT/$(date -d yesterday +%Y)/IMO0OPSFIN_$(date -d yesterday +%Y%j)0000_01D_01D_SOL.SNX.gz"
# read the whole file (grep -q would stop zcat early: SIGPIPE, and pipefail
# would then call it "not found")
n=$( { zcat "$snx" 2>/dev/null || true; } | grep -E '^ (SVIE|SVIN) ' | cut -c2-5 | sort -u | wc -l)
if ! [ -r "$snx" ] || [ "$n" -lt 2 ]; then
    echo "svinafell-watch: okada has no SVIE+SVIN for $yesterday yet"; exit 0
fi

exec 9>"$LOCK"
if ! flock -n 9; then
    echo "svinafell-watch: a whole run holds the lock (it draws $AREA first)"; exit 0
fi
echo "svinafell-watch: $yesterday in from okada, redrawing $AREA"
"$HERE/update-pos-yang.sh" || echo "svinafell-watch: .pos update failed, plotting the series as they are" >&2
gpsplot run --area "$AREA" --plots-dir "$PLOTS"
