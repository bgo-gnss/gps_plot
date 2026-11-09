#!/bin/bash
# install.sh: the gpsplot stack on yang-d01, as gpsops. Safe to re-run.
#
#   deployment/yang/install.sh [--enable]
#
# Run from ~gpsops/src/gps_plot, after SUDO-STEPS.md. The checkout it
# runs from is what gets installed; the siblings come from pins.txt and the
# third-party versions from requirements.lock (the laptop's versions, so the
# figures can be compared). --enable also starts the timers: leave it off
# until seed-from-laptop.sh has run and a by-hand run has worked.
set -euo pipefail

HERE=$(dirname "$(readlink -f "$0")")
ENABLE=0
case "${1:-}" in
    --enable) ENABLE=1 ;;
    "") ;;
    *) sed -n 2,10p "$0"; exit 64 ;;
esac
set -a; . "$HERE/yang.env"; set +a

UV_VERSION=0.8.22
UNITS=(svinafell-watch.service svinafell-watch.timer
    gpsplot-rtk.service gpsplot-rtk.timer
    gpsplot-rtk-long.service gpsplot-rtk-long.timer
    gpsplot-run.service gpsplot-run.timer
    gpsplot-run-update.service gpsplot-run-update.timer
    yang-config-sync.service yang-config-sync.timer)
SRC=$HOME/src  # the code; the venv, caches and data stay on /mnt/data

fail() { echo "install: $1" >&2; exit 1; }
[ "$(id -un)" = gpsops ] || fail "run as gpsops (sudo -iu gpsops)"
[ -w "$YANG_ROOT" ] || fail "$YANG_ROOT is not writable: see SUDO-STEPS.md"
[ "$(readlink -f "$HERE/../..")" = "$SRC/gps_plot" ] || fail "run from a checkout at $SRC/gps_plot"
for tool in gmt gs git curl latex dvipng; do
    command -v "$tool" >/dev/null || fail "$tool is missing: see SUDO-STEPS.md"
done
[ -e "$(gmt --show-library)" ] || fail "$(gmt --show-library) is missing: see SUDO-STEPS.md"
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}

mkdir -p "$YANG_ROOT"/{bin,config,cache/gps_analysis/noise,gg/tables,gis,plots,pos,velocities} "$SRC"
# seed-from-laptop.sh writes these as bgo, through the group
chmod 2775 "$YANG_ROOT" "$YANG_ROOT"/{config,cache,cache/gps_analysis,cache/gps_analysis/noise,gg,gg/tables,gis,pos,velocities}

# uv: the release binary, no installer script
if [ "$("$YANG_ROOT/bin/uv" --version 2>/dev/null | cut -d' ' -f2)" != "$UV_VERSION" ]; then
    name=uv-$(uname -m)-unknown-linux-gnu
    curl -fsSL "https://github.com/astral-sh/uv/releases/download/$UV_VERSION/$name.tar.gz" |
        tar -xz -C "$YANG_ROOT/bin" --strip-components=1 "$name/uv"
fi

# the siblings, at their pins
sibling_dirs=()
while read -r name url commit; do
    case $name in ''|\#*) continue ;; esac
    [ -d "$SRC/$name/.git" ] || git clone -q "$url" "$SRC/$name"
    git -C "$SRC/$name" fetch -q origin
    git -C "$SRC/$name" -c advice.detachedHead=false checkout -q --detach "$commit"
    sibling_dirs+=("$SRC/$name")
done < "$HERE/pins.txt"

# Editable with --no-deps, as on the laptop: the git+https requirements between
# the siblings would otherwise pull them a second time, at other commits.
uv python install -q 3.13
uv venv -q --allow-existing --python 3.13 "$YANG_ROOT/venv"
export VIRTUAL_ENV=$YANG_ROOT/venv
uv pip install -q -r "$HERE/requirements.lock"
editable=()
for dir in "${sibling_dirs[@]}" "$SRC/gps_plot"; do editable+=(-e "$dir"); done
uv pip install -q --no-deps --no-sources "${editable[@]}"

# the config path every tool defaults to, kept on the big disk
mkdir -p "$HOME/.config"
if [ -e "$HOME/.config/gpsconfig" ] && [ ! -L "$HOME/.config/gpsconfig" ]; then
    fail "$HOME/.config/gpsconfig is a real directory: move it away first"
fi
ln -sfn "$YANG_ROOT/config" "$HOME/.config/gpsconfig"

mkdir -p "$HOME/.config/systemd/user"
for unit in "${UNITS[@]}"; do
    ln -sf "$HERE/$unit" "$HOME/.config/systemd/user/$unit"
done
systemctl --user daemon-reload

python -c "import pygmt; pygmt.show_versions()" | grep -E 'version|GMT' | head -8
gpsplot run --help >/dev/null
gps-realign --help >/dev/null

if [ $ENABLE -eq 1 ]; then
    for unit in "${UNITS[@]}"; do
        [[ $unit == *.timer ]] && systemctl --user enable --now "$unit"
    done
fi
echo "install: done (gps_plot $(git -C "$SRC/gps_plot" rev-parse --short HEAD))"
