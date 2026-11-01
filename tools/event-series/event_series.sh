#!/bin/bash
# event_series.sh OUTDIR FIRST LAST [gpsplot map velocity options ...]
#
# Weekly velocity maps of an event, after the fact: one map for each week
# ending FIRST, FIRST + 7 days, ... up to LAST (YYYY-MM-DD), each drawn with
# -e that day, titled with it and named OUTDIR/<NAME>-YYYYMMDD.png, then a
# contact sheet of them all (OUTDIR/<NAME>-contact.png, ImageMagick
# montage). The options after LAST go to every `gpsplot map velocity` (the
# windows, the stations' region, layers, ...); see README.md for the
# Seyðisfjörður set. NAME is OUTDIR's last part. Not an operational plot:
# plots.yaml has the daily map.
set -euo pipefail
[ $# -ge 3 ] || { sed -n '2,12p' "$0"; exit 2; }
out=$1 first=$2 last=$3
shift 3
name=$(basename "$out")
gpsplot=${GPSPLOT:-gpsplot}
mkdir -p "$out"
d=$first
while [[ ! "$d" > "$last" ]]; do
  e=${d//-/}
  if $gpsplot map velocity -e "$e" --title "week ending $d" --save png \
      -d "$out" --name "$name-$e" "$@" > "$out/log-$e.txt" 2>&1; then
    echo "$d ok"
  else
    echo "$d FAILED (see $out/log-$e.txt)"
  fi
  d=$(date -d "$d + 7 days" +%F)
done
montage "$out/$name"-2*.png -tile 5x -geometry 700x+4+4 "$out/$name-contact.png"
echo "$out/$name-contact.png"
