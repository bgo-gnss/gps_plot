#!/bin/bash
# yang-plots.sh: yang's plot run, under gpsplot-run.lock (the units take it).
#
#   yang-plots.sh            everything selected, redrawn (nightly)
#   yang-plots.sh --changed  only what new days touched; nothing when the
#                            series did not change since this morning's run
#
# yang draws a selection of the shared plots.yaml: AREAS, then the standard
# plots. Each has its own --changed baseline (--state). Areas go first because
# the watched ones are among them.
set -uo pipefail
: "${YANG_ROOT:=/mnt/data/gpsops}"

AREAS=svinafellsheidi,reykjanes,seydisfjordur
HERE=$(dirname "$(readlink -f "$0")")
PLOTS=$YANG_ROOT/plots
POS=$YANG_ROOT/pos
SEEN=$PLOTS/_runs/.yang-plots-state     # "<date> <fingerprint>"

changed=()
[ "${1:-}" = --changed ] && changed=(--changed)

"$HERE/update-pos-yang.sh" || echo "yang-plots: .pos update failed, plotting the series as they are" >&2

# The data lines, not the file times or the "Release Date" header. The first
# run of a day always draws: the plot windows count back from today.
today=$(date +%F)
print=$(find "$POS" -maxdepth 1 -name '*.imo.24hr_itrf14.pos' -print0 | sort -z |
        xargs -0 cat | grep -v '^Release Date' | sha1sum | cut -c1-16)
if [ ${#changed[@]} -gt 0 ] && [ "$(cat "$SEEN" 2>/dev/null)" = "$today $print" ]; then
    echo "yang-plots: no new data since the last run today, plots kept"
    exit 0
fi

status=0
gpsplot run --area "$AREAS" --plots-dir "$PLOTS" --state areas "${changed[@]}" || status=$?
gpsplot run --product standard --no-prepare --plots-dir "$PLOTS" --state standard "${changed[@]}" || status=$?
# recorded even when some plots failed: a re-run would fail alike
mkdir -p "$(dirname "$SEEN")" && echo "$today $print" > "$SEEN"
exit $status
