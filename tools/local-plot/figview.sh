#!/usr/bin/env bash
# figview.sh — render into a gitignored scratch figdir, then browse it in ranger.
#
# `plot-gps-timeseries` writes to the CWD when `-d` is omitted, which is how
# stray `{STA}-*.pdf` end up in the repo root. This wrapper always pins `-d` to
# a scratch dir (gitignored `gps_plot/tmp-figdir/` by default) and picks a
# preview-friendly `--save` set, so nothing lands in the project space.
#
# Requires the `gpslibrary` mamba env (all packages editable-installed there):
#   mamba activate gpslibrary && ./figview.sh ...
#
# Usage:
#   ./figview.sh SKHA --special year          # render, then list what was written
#   ./figview.sh -B SKHA BLON --special 90d   # ... and open ranger on the figdir
#   ./figview.sh --browse                     # just browse the figdir
#   ./figview.sh --last                       # open the newest figure directly
#   ./figview.sh --clean                      # empty the figdir
#
# Any other argument is passed through to `plot-gps-timeseries` verbatim, so the
# full CLI (--view, --ref, --start/--end, --ylim, --tType, -i ...) still applies.
# `-d` is always pinned to FIGDIR (a caller-supplied -d cannot escape it);
# `--save` is only defaulted, so passing your own --save still wins.
#
# Env overrides:
#   FIGDIR  scratch figure dir  (default <gps_plot>/tmp-figdir)
#   SAVE    save format(s)      (default png,pdf — png previews fast, pdf is the
#                               on-demand vector; eps is legacy, see README)
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PKG_ROOT="$(cd -- "$HERE/../.." && pwd)"

FIGDIR="${FIGDIR:-$PKG_ROOT/tmp-figdir}"
SAVE="${SAVE:-png,pdf}"

browse=0
plot_args=()

for arg in "$@"; do
  case "$arg" in
    -B|--browse) browse=1 ;;
    --last)
      newest="$(ls -1t "$FIGDIR"/*.{png,pdf,eps,ps} 2>/dev/null | head -1 || true)"
      [[ -n "$newest" ]] || { echo "!! no figures in $FIGDIR"; exit 1; }
      echo ">> opening $newest"
      exec rifle "$newest"
      ;;
    --clean)
      # The figdir is scratch, but it may hold figures rendered by hand rather
      # than by this script — list them before asking, never a blind y/N.
      doomed=$(ls -1 "$FIGDIR"/*.{png,pdf,eps,ps} 2>/dev/null || true)
      [[ -n "$doomed" ]] || { echo ">> $FIGDIR already empty"; exit 0; }
      echo "$doomed" | sed 's#.*/#   #'
      read -r -p "delete these $(echo "$doomed" | wc -l) figure(s) from $FIGDIR? [y/N] " reply
      [[ "$reply" == [yY] ]] || { echo "aborted"; exit 0; }
      rm -f -- "$FIGDIR"/*.{png,pdf,eps,ps}
      echo ">> cleaned $FIGDIR"
      exit 0
      ;;
    *) plot_args+=("$arg") ;;
  esac
done

mkdir -p "$FIGDIR"

open_ranger() {
  command -v ranger >/dev/null || { echo "!! ranger not on PATH"; return 1; }
  # ranger gates its kitty backend on `'kitty' in TERM` and nothing else — there
  # is no capability probe, so the string alone decides. herdr reports
  # TERM=xterm-256color while re-rendering kitty graphics itself (needs
  # [experimental] kitty_graphics = true + a kitty outer terminal), so inside a
  # herdr pane we override TERM for ranger only. Everything else keeps the real
  # TERM and just gets a warning.
  # herdr additionally rejects file-based transfer (`EINVAL: unsupported medium`
  # — it renders on the attached client, where a pane-side path is meaningless).
  # Ranger only understands OK/EBADF from that query, so the kitty_stream plugin
  # forces inline transfer; this flag is what switches the plugin on.
  if [[ "${HERDR_ENV:-}" == "1" ]]; then
    exec env TERM=xterm-kitty RANGER_KITTY_FORCE_STREAM=1 ranger "$FIGDIR"
  fi
  case "${TERM:-}" in
    *kitty*) : ;;
    *) echo "!! TERM=${TERM:-unset} — image previews need kitty, or herdr with kitty_graphics = true" ;;
  esac
  exec ranger "$FIGDIR"
}

if [[ ${#plot_args[@]} -eq 0 ]]; then
  open_ranger
fi

command -v plot-gps-timeseries >/dev/null \
  || { echo "!! plot-gps-timeseries not on PATH — run 'mamba activate gpslibrary' first"; exit 1; }

# `-d` is pinned unconditionally: argparse lets the last occurrence win (verified),
# so appending it last guarantees nothing escapes the scratch dir even if the
# caller passed their own -d/--figDir. Point FIGDIR elsewhere to redirect.
inject=(-d "$FIGDIR")
# `--save` is a real choice (png,pdf vs eps vs pdf only), so only default it.
case " ${plot_args[*]} " in
  *" --save "*|*" --save="*) ;;
  *) inject+=(--save "$SAVE") ;;
esac

echo ">> plot-gps-timeseries ${plot_args[*]} ${inject[*]}"
plot-gps-timeseries "${plot_args[@]}" "${inject[@]}"

echo ">> figures in $FIGDIR:"
ls -1t "$FIGDIR"/*.{png,pdf,eps,ps} 2>/dev/null | sed 's#.*/##' | head -20

(( browse )) && open_ranger
exit 0
