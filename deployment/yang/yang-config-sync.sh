#!/bin/bash
# yang-config-sync.sh: yang's config follows gps-config-data main, every 10
# min (the pattern of receivers' sync-config.sh on rek-d01).
#
# Convergent: every tick compares the clone with the live files, whether or
# not the fetch brought anything. A changed set is staged whole and checked
# with the installed gpsplot (--dry-run --check-files) against the new files;
# only if that passes do the files replace the live ones. A failed fetch, a
# diverged clone or a refused set leaves the live config as it was.
set -uo pipefail
: "${YANG_ROOT:=/mnt/data/gpsops}"

REPO=$HOME/git/gps-config-data
LIVE=$YANG_ROOT/config
STAGE=$YANG_ROOT/config.stage
ENV_FILE=$REPO/environments/yang-d01.vedur.is.env

# repo paths; deployed flat. Not database.cfg, receivers.cfg or anything else
# yang's plots do not read.
# rtk_thresholds.yaml: the RTK warning's calibrated edges (rtk-bench
# --calibrate on the laptop, copied into the repo by hand; BGÓ 2026-11-13)
FILES=(plots.yaml map_layers.yaml station_areas.yaml stations.cfg rtk_thresholds.yaml
    detrend_itrf2008.csv station-plate station_coord.xyz
    analysis-lane/analysis.yaml analysis-lane/detrend_params.itrf14.json
    analysis-lane/events.yaml analysis-lane/steps.yaml
    analysis-lane/protect_windows.csv analysis-lane/outlier_overrides.csv
    analysis-lane/excluded_epochs.csv analysis-lane/fit_windows.csv
    analysis-lane/segment_exclusions.csv analysis-lane/junction_offsets.csv
    analysis-lane/detrend_status.csv)

[ -d "$REPO/.git" ] || { echo "config-sync: no clone at $REPO (see SUDO-STEPS.md)" >&2; exit 1; }

git -C "$REPO" fetch --quiet 2>/dev/null || true
if [ "$(git -C "$REPO" rev-parse HEAD)" != "$(git -C "$REPO" rev-parse '@{u}' 2>/dev/null || git -C "$REPO" rev-parse HEAD)" ]; then
    git -C "$REPO" pull --ff-only --quiet ||
        echo "config-sync: pull failed (diverged?), using the clone as it is" >&2
fi
rev=$(git -C "$REPO" rev-parse --short HEAD)

rm -rf "${STAGE:?}"; mkdir -p "$STAGE"
for f in "${FILES[@]}"; do
    cp "$REPO/$f" "$STAGE/" || { echo "config-sync: $f missing at $rev, nothing changed" >&2; exit 1; }
done
python3 - "$REPO" "$ENV_FILE" "$STAGE/postprocess.cfg" <<'PY' || { echo "config-sync: postprocess.cfg did not render at $rev" >&2; exit 1; }
import contextlib, io, sys
from pathlib import Path
repo, env, out = (Path(a) for a in sys.argv[1:])
sys.path.insert(0, str(repo))
import deploy
config = deploy.read_environment_config(env)
# its "unset" notes are expected (keys yang has no use for), not news every tick
with contextlib.redirect_stdout(io.StringIO()):
    text = deploy.render_template(repo / "postprocess.cfg.template", config, env.name)
out.write_text(text)
PY

changed=()
for f in "$STAGE"/*; do
    name=$(basename "$f")
    cmp -s "$f" "$LIVE/$name" || changed+=("$name")
done
[ ${#changed[@]} -gt 0 ] || exit 0

if ! out=$(GPS_CONFIG_PATH=$STAGE gpsplot run "$STAGE/plots.yaml" --dry-run --check-files 2>&1); then
    echo "config-sync: $rev REFUSED, live config kept ($(echo "${changed[*]}")):" >&2
    echo "$out" | tail -n 20 >&2
    exit 1
fi

for name in "${changed[@]}"; do
    install -m 640 "$STAGE/$name" "$LIVE/.$name.new" && mv -f "$LIVE/.$name.new" "$LIVE/$name"
done
echo "config-sync: synced to $rev: ${changed[*]}"

# aflogun caches these until it restarts (a service not installed: no-op)
for name in "${changed[@]}"; do
    case $name in
        station_areas.yaml|stations.cfg|events.yaml|postprocess.cfg)
            systemctl --user try-restart aflogun.service 2>/dev/null || true
            echo "config-sync: aflogun restarted ($name)"
            break ;;
    esac
done
