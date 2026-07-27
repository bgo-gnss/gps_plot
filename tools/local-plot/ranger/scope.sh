#!/usr/bin/env bash
# scope.sh — ranger preview delta for the figure-browsing workflow.
#
# NOT installed by this repo. Copy it into the ranger stow package (see
# ../README.md, "Terminal figure browsing"); ranger picks up
# ~/.config/ranger/scope.sh automatically when it exists — no rc.conf change.
#
# Ranger's packaged scope.sh ships the PDF raster branch COMMENTED OUT (PDFs
# fall through to a pdftotext dump, which is useless for a plot) and has no
# PostScript branch at all. Rather than fork the 300-line upstream script, this
# handles PDF/PS/EPS and delegates everything else to the packaged one, so it
# keeps working across ranger upgrades.
#
# Argument contract and exit codes are ranger's (1.9.x):
#   $1 path  $2 width  $3 height  $4 image cache path  $5 'True'|'False'
#   exit 6 = show the image at the cache path; 7 = show the file itself.

set -o noclobber -o noglob -o nounset -o pipefail

FILE_PATH="${1}"
IMAGE_CACHE_PATH="${4}"
PV_IMAGE_ENABLED="${5}"

# Hand off to ranger's own scope.sh for every type not handled below.
delegate() {
    local candidate
    for candidate in \
        ${RANGER_BUNDLED_SCOPE:-} \
        /usr/lib/python3/dist-packages/ranger/data/scope.sh \
        /usr/share/ranger/data/scope.sh \
        /usr/local/lib/python3*/dist-packages/ranger/data/scope.sh
    do
        [ -x "${candidate}" ] && exec "${candidate}" "$@"
    done
    exit 1   # no packaged scope.sh found — no preview rather than a wrong one
}

if [ "${PV_IMAGE_ENABLED}" = "True" ]; then
    # pdftoppm/gs both insist on appending their own extension, so render to a
    # scratch name and move the result onto the cache path ranger asked for.
    tmp="${IMAGE_CACHE_PATH}.tmp"

    case "$(printf '%s' "${FILE_PATH##*.}" | tr '[:upper:]' '[:lower:]')" in
        pdf)
            pdftoppm -png -r 120 -f 1 -l 1 -singlefile -- "${FILE_PATH}" "${tmp}" \
                && mv -f -- "${tmp}.png" "${IMAGE_CACHE_PATH}" \
                && exit 6
            exit 1
            ;;
        eps)
            # -dEPSCrop honours the BoundingBox; without it the plot sits in a
            # letter-sized page and previews as a stamp in the corner.
            gs -q -dSAFER -dBATCH -dNOPAUSE -dEPSCrop -sDEVICE=png16m -r120 \
               -sOutputFile="${IMAGE_CACHE_PATH}" "${FILE_PATH}" >/dev/null 2>&1 \
                && exit 6
            exit 1
            ;;
        ps)
            gs -q -dSAFER -dBATCH -dNOPAUSE -dFirstPage=1 -dLastPage=1 \
               -sDEVICE=png16m -r120 \
               -sOutputFile="${IMAGE_CACHE_PATH}" "${FILE_PATH}" >/dev/null 2>&1 \
                && exit 6
            exit 1
            ;;
    esac
fi

delegate "$@"
