#!/bin/bash
# yang-update.sh: bring yang's gpsplot code forward, on purpose. As gpsops:
#
#   deployment/yang/yang-update.sh [REF]     REF: a commit, branch or tag;
#                                            default: this branch's upstream
#
# Waits for a running plot run, moves the checkout, re-installs (pins.txt and
# requirements.lock come with it) and checks the live config with the new code.
# If the install or the check fails it goes back to the commit it came from.
# Config needs no command: yang-config-sync follows gps-config-data main.
set -uo pipefail

HERE=$(dirname "$(readlink -f "$0")")
REPO=$(readlink -f "$HERE/../..")
set -a; . "$HERE/yang.env"; set +a
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}

[ "$(id -un)" = gpsops ] || { echo "update: run as gpsops" >&2; exit 1; }
exec 9>"$XDG_RUNTIME_DIR/gpsplot-run.lock"
echo "update: waiting for a plot run to finish"
flock -w 3600 9 || { echo "update: a plot run still holds the lock" >&2; exit 1; }

git -C "$REPO" fetch -q origin || { echo "update: fetch failed, nothing changed" >&2; exit 1; }
from=$(git -C "$REPO" rev-parse HEAD)
branch=$(git -C "$REPO" symbolic-ref -q --short HEAD || true)
ref=${1:-@\{u\}}
to=$(git -C "$REPO" rev-parse --verify -q "$ref^{commit}") || { echo "update: no such ref: $ref" >&2; exit 1; }

goto() {  # the checkout at commit $1: forward on the branch, or back to it
    if [ -n "$branch" ] && git -C "$REPO" merge-base --is-ancestor "$(git -C "$REPO" rev-parse HEAD)" "$1"; then
        git -C "$REPO" merge -q --ff-only "$1"
    elif [ -n "$branch" ]; then
        git -C "$REPO" reset -q --hard "$1"
    else
        git -C "$REPO" -c advice.detachedHead=false checkout -q --detach "$1"
    fi
}

if [ "$from" = "$to" ]; then
    echo "update: already at $(git -C "$REPO" rev-parse --short "$to")"; exit 0
fi
echo "update: $(git -C "$REPO" rev-parse --short "$from") -> $(git -C "$REPO" rev-parse --short "$to")"
goto "$to" || { echo "update: cannot move the checkout" >&2; exit 1; }

if "$HERE/install.sh" && gpsplot run --dry-run --check-files >/dev/null; then
    echo "update: done"
    exit 0
fi
echo "update: FAILED, going back to $(git -C "$REPO" rev-parse --short "$from")" >&2
goto "$from" && "$HERE/install.sh"
exit 1
