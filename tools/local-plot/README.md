# local-plot — laptop 3-view time-series workflow

Prove and iterate on the static-plot lane locally, publishing nothing:

```
gps-globk-tot          join okada GLOBK segments  ->  ~/gps-data/TOT
gps-estimate-detrend   fit stored-detrend params  ->  <testcfg>/detrend_params.json
plot-gps-timeseries    render raw | cleaned | detrended  ->  ~/gps-data/figs
```

The wrapper reads a **layered test config** (`setup-testcfg.sh`) so the deployed
`~/.config/gpsconfig` is never touched: base config files are symlinked (no
credentials), the analysis-lane catalogs are copied from the `gps-config-data`
checkout, and `detrend_params.json` is generated into the test dir.

**All generated output lives outside this repo** (`~/gps-data/…`); only these
scripts are versioned.

## Quick start

```bash
mamba activate gpslibrary                # all packages editable here; scripts on PATH
cd tools/local-plot
./setup-testcfg.sh                       # build ~/gps-data/testcfg (once, or after catalog edits)
ESTIMATE=1 ./plot-views.sh SKHA BLON      # fit detrend, then render all 3 views
./plot-views.sh AKUR                      # plot only (reuse existing detrend_params.json)
SAVE=pdf,png VIEWS=detrended ./plot-views.sh SKHA
```

Output filenames are siblings: `{STA}-plate.png`, `{STA}-plate-cleaned.png`,
`{STA}-detrend.png` (+ any `--save` vector formats — `pdf`/`eps` work natively).

## Prerequisites

- **`mamba activate gpslibrary`** — the dev env where all nine packages are
  editable-installed. `gps-globk-tot`, `gps-estimate-detrend` and
  `plot-gps-timeseries` all come from there (both scripts merged to `main`
  2026-07-21). The scripts abort with a hint if the env is not active.
- **gps-config-data analysis-lane catalogs** including `segment_exclusions.csv`
  and `fit_windows.csv`. `setup-testcfg.sh` copies them from `GCD`
  (default `~/git/gps-config-data`, `main`).
- `~/gps-data/TOT` populated by `gps-globk-tot` (194 stations as of 2026-07-28).

## Refreshing `~/gps-data/TOT` (the joined dataset)

**Why it matters, not just how.** The production archive `/mnt_data/gpsdata`
holds *raw* GLOBK segments — overlapping in time, with per-segment datums that
are not reconciled. HVER's Up wraps by 10 m there (1746 epochs off-datum);
`geo_dataread.globk_join` de-wraps and min-σ-dedupes them, leaving 1. So
`postprocess.cfg` `totDir` points at the joined output and this refresh is a
MANUAL step.

**The local join is kept on purpose, not pending anything** (BGÓ, 2026-08-18).
This used to read "local only until it ships to production" — it shipped on
2026-08-17 (`globk_join` runs in okada's `tododaily`, layered after
`compGLOBK`), and `/mnt_data/gps_gmt_data/TOT` now carries joined series for
the `stations.cfg` stations. Do NOT switch to it: `gpslibrary` development
stays **isolated from production**, so the analysis lane keeps re-joining into
`~/gps-data/TOT` from the segments and nothing here depends on a production
pipeline that can change under it. A production-fed `totDir` would also make
every local figure irreproducible the moment okada's join is reverted — which
is one deleted line in `tododaily`.

```bash
mamba activate gpslibrary
cd ~/gps-data/TOT && ls | sed 's/^mb_//; s/_TOT.*//' | sort -u > /tmp/sta.txt
xargs -a /tmp/sta.txt gps-globk-tot \
    --pre /mnt_data/gps_gmt_data/pre \
    --rap /mnt_data/gps_gmt_data/rap \
    --out ~/gps-data/TOT
```

- **Use `xargs`, not `$(cat …)`** — this is zsh, where an unquoted parameter
  does NOT word-split, so the whole station list arrives as one argument and
  the join errors out on every component (harmlessly: it writes nothing).
- Segment exclusions (SEY1 name-clash, SUND rap-subset) resolve from the
  deployed `~/.config/gpsconfig/segment_exclusions.csv`. If that is missing the
  join warns and proceeds **without** them — copy it from
  `~/git/gps-config-data/analysis-lane/`.
- The station list above is inherited from what is already joined. Stations
  present in `pre`/`rap` but never joined locally stay missing; `pre` carries
  378 codes and `rap` 394, though most extras are global reference sites.
- Verify after: a joined series keeps its true height (RHOF Up sits near
  6.9 m, REYK 3.0 m), so `|U| > 1 m` is NOT a wrap test — check deviation from
  each station's own median instead.

## Fractional-year epochs are at NOON

`gtimes.TimetoYearf` returns the **daily-solution reference epoch**, i.e. noon:
May 29 2008 is `149.5/366 = 2008.40847`, not the naive midnight `149/366 =
2008.4071`. Everything in the lane agrees on noon — `steps.csv` rows,
`fit_windows.csv`, `detrend_params.json`, the workbench's `--event`. A
hand-computed midnight fractional year therefore sits **half a day early**,
which is enough to fail an epoch-coincidence check while changing nothing
visible: on daily data both values fall between the same pair of observations,
so a step term fitted at either epoch is identical. Convert dates with
`TimetoYearf`; never divide day-of-year by the year length yourself.

## Config knobs

Every path is env-overridable — see the header of each script. Common:

| var | default | meaning |
|-----|---------|---------|
| `TOT_DIR` | `~/gps-data/TOT` | local join output |
| `FIGDIR` | `~/gps-data/figs` | figure output (outside repo) |
| `TESTCFG` | `~/gps-data/testcfg` | layered test config |
| `SAVE` | `png` | `--save` formats (`pdf`, `eps,pdf,png`, …) |
| `VIEWS` | `raw cleaned detrended` | which views to render |
| `GCD` | `~/git/gps-config-data` | config-data checkout (setup only) |

### `GPS_CONFIG_PATH` is no longer needed (2026-07-29)

The analysis-lane catalogs — `steps.csv`, `protect_windows.csv`,
`outlier_overrides.csv`, `fit_windows.csv`, `analysis.yaml` — plus the
generated `detrend_params.json` are now **deployed to `~/.config/gpsconfig/`**,
which is where `gps_parser.catalog_path()` looks by default. `TESTCFG` survives
as an all-symlink mirror, so `plot-views.sh` still works, but a bare
`gps-detrend-workbench` / `plot-gps-timeseries` in any shell now resolves them.

Before this, forgetting the export was silent-but-costly rather than an error:
each catalog degrades to "nothing declared" with a warning, so SELF's 2008
Ölfus step went undeclared, full detection tripped the excess-candidate abort,
and the S0 fallback fitted a 150 mm step with a sinusoid — rms
`[34.1, 32.0, 20.8]` instead of `[1.97, 3.49, 6.54]` mm, seasonal inflated
~10×, north rate sign-flipped. If you ever see `no step catalog` or
`stages S0` on a station you know has a declared step, the catalogs are not
resolving.

> ⚠️ **Do not run `gps-config-data/deploy.py` against `~/.config/gpsconfig` on
> this laptop without re-checking `totDir`.** The repo's `postprocess.cfg` still
> has the production `totDir = /mnt_data/gpsdata/`, while the deployed copy is
> pointed at `~/gps-data/TOT` for the local join (HVER 10 m Up wraps). A deploy
> would silently revert that and put every plot back on unjoined segments.

## fit_windows.csv — per-station detrend windows

Most long-history stations have multi-year early gaps (sparse campaign era), so
the default `max_gap_years=0.5` gate rejects them: only ~20% of the network
passes untouched. The fix is a per-station **fit window** in
`gps-config-data/analysis-lane/fit_windows.csv`
(`sta,window_start,window_end,max_gap_years,min_epochs,min_span_years,steps,comment`)
that fits the stored-detrend trajectory on a recent continuous **pre-unrest**
window. Edit that catalog, re-run `setup-testcfg.sh`, then
`ESTIMATE=1 ./plot-views.sh <STA>`.

## figview.sh — scratch figdir + terminal browsing

`plot-gps-timeseries` writes to the CWD when `-d` is omitted, so running it from
the package root drops `{STA}-*.pdf` straight into the repo. `figview.sh` pins
`-d` to **`gps_plot/tmp-figdir/`** (gitignored) and defaults `--save png,pdf`:

```bash
./figview.sh SKHA --special year          # render, list what was written
./figview.sh -B SKHA BLON --special 90d   # ... and open ranger on the figdir
./figview.sh --browse                     # just browse
./figview.sh --last                       # open the newest figure (via rifle)
./figview.sh --clean                      # empty the figdir (confirms first)
```

Everything else is passed through to `plot-gps-timeseries` verbatim. `-d` is
appended **unconditionally** — argparse takes the last occurrence, so output
cannot escape the scratch dir even if you pass your own `-d`; redirect with
`FIGDIR=~/gps-data/figs` to share the `plot-views.sh` output dir instead.
`--save` is only defaulted, so `--save eps` on the command line still wins.

**Formats.** `png` is the published product and the fast preview; `pdf` is the
on-demand vector. `eps` is legacy (dropped as a product 2026-07-15: eps 442 ms +
ghostscript 470 ms vs png 83 ms) — it still renders, and previews, but there is
no reason to ask for it in new work.

### Terminal browsing (ranger + kitty)

| file | preview | `Enter` opens |
|------|---------|---------------|
| `.png` | native (kitty graphics) | `imv` |
| `.pdf` | needs `ranger/scope.sh` below | `zathura` |
| `.eps` / `.ps` | needs `ranger/scope.sh` below | needs `zathura-ps`, below |

PNG and PDF opening already work with a stock ranger; the two gaps are **image
previews for PDF/PS** and **opening PS/EPS**.

**1 — previews.** Ranger's packaged `scope.sh` ships the PDF raster branch
commented out and has no PostScript branch, so both fall back to a useless text
dump. `ranger/scope.sh` here handles PDF/PS/EPS and delegates everything else to
the packaged script (so it survives ranger upgrades). Install it into the ranger
stow package — ranger picks up `~/.config/ranger/scope.sh` automatically, no
`rc.conf` change:

```bash
cp tools/local-plot/ranger/scope.sh ~/.dotfiles/ranger/.config/ranger/scope.sh
chmod +x ~/.dotfiles/ranger/.config/ranger/scope.sh
```

(`~/.config/ranger` is a stow tree-fold symlink into `~/.dotfiles/ranger`, so
write the dotfiles path, not the `~/.config` one.)

**2 — opening PS/EPS.** Stock Ubuntu ships only `zathura-pdf-poppler`, so
zathura cannot open PostScript at all; rifle has no `.eps` rule and falls
through to `xdg-open`, so the fix is the desktop MIME default (no `rifle.conf`
fork needed):

```bash
sudo apt install zathura-ps
xdg-mime default org.pwmt.zathura-ps.desktop \
    image/x-eps image/eps application/eps application/x-eps application/postscript
```

All five types matter: shared-mime-info tags a `.eps` as **`image/x-eps`**, not
`application/postscript` — setting only the latter leaves `.eps` opening in
Inkscape. Verify with
`xdg-mime query default "$(xdg-mime query filetype some.eps)"`.

### Image previews inside herdr

Ranger gates its kitty backend on `'kitty' in TERM` and nothing else, so any
multiplexer that rewrites `TERM` disables previews. herdr reports
`TERM=xterm-256color`, but unlike tmux it can re-render kitty graphics itself:

```toml
# ~/.dotfiles/herdr/.config/herdr/config.toml
[experimental]
kitty_graphics = true
```

then `herdr server reload-config` (no session restart — `prefix+shift+r` is
bound to `herdr-refresh-env` here, so the default `reload_config` binding is
shadowed and the CLI is the way in), **then detach and re-attach**
(`prefix+d`, then `herdr`).

That last step is not optional. `reload-config` reaches only the server, while
the process that paints graphics onto the outer terminal is the *client*, which
reads its config when it attaches. Skipping it leaves a convincing half-working
state: the server's terminal emulation processes the image and replies `OK`,
the client never paints, and previews are blank. Panes survive a detach — the
server owns them. Compare `ps -eo pid,lstart,args | grep herdr` against the
config mtime if in doubt. `figview.sh` detects `HERDR_ENV=1` and
launches ranger with `TERM=xterm-kitty` so the gate passes.

That is necessary but **not sufficient**. Ranger picks its transfer mode from a
single startup query (`_late_init`): it asks about `t=f` — "open this path
yourself" — and understands only `OK` (use temp files) or `EBADF` (send pixels
inline). herdr renders on the attached *client*, so a pane-side path means
nothing to it and it replies `EINVAL: unsupported medium`. Neither answer, so
ranger disables previews even though inline transfer works.

`ranger/plugins/kitty_stream.py` forces the inline path and skips the query.

It also **changes what inline means**. Ranger's stream mode transmits the
flattened pixel buffer, so a 630×1000 preview is 3.2 MB (4 MB base64); herdr
caps the frame it forwards to the attached client and logs `dropping oversized
graphics payload for client frame` — the terminal replies `OK` and the image
never reaches the screen, which looks exactly like a blank preview. The plugin
sends PNG bytes inline instead (`f=100`), 406 KB for the same figure. It also
bounds the reply read, which ranger does with an unbounded `stdin.read(1)`.

Install it alongside `scope.sh`:

```bash
mkdir -p ~/.dotfiles/ranger/.config/ranger/plugins
cp tools/local-plot/ranger/plugins/kitty_stream.py \
   ~/.dotfiles/ranger/.config/ranger/plugins/
```

It is inert unless `RANGER_KITTY_FORCE_STREAM=1`, so a plain kitty terminal
keeps ranger's own autodetection. `figview.sh` sets it together with
`TERM=xterm-kitty` when it sees `HERDR_ENV=1`.

**Probe first if any of this changes.** `kitty-graphics-probe.py` tests each
medium (`d`/`f`/`t`) under a 3 s `select()` and reports the `TIOCGWINSZ` pixel
size, so nothing can hang the way ranger's unguarded `stdin.read(1)` would:

```bash
python3 tools/local-plot/kitty-graphics-probe.py     # must be a real pane, not a pipe
```

If it reports no pixel size, the plugin falls back to an 8×16 cell — override
with `RANGER_KITTY_CELL_PX=10x21` (matching your kitty font) if previews come
out mis-scaled.

For a **blank** preview (terminal says `OK`, nothing appears) the question is
transfer size, not capability. `kitty-transfer-test.py` draws the same image
both ways and prints each payload size, outside ranger:

```bash
python3 tools/local-plot/kitty-transfer-test.py ../../tmp-figdir/SKHA-plate-year.png
printf '\033_Ga=d\033\\'      # clear leftovers
```

Whichever block is visible is the medium that survives to the screen — an `OK`
reply only means herdr accepted the payload.

### If PNG previews render but PDF/EPS stay blank

The cause is a **pixel-count** ceiling, not encoded size and not the exit-6 path.
herdr/ghostty keeps decoded images as RGBA (`width*height*4`) and answers
`ENOMEM: out of memory` past roughly 8 MB. Measured here:

| image | megapixels | RGBA | payload | reply |
|---|---|---|---|---|
| PDF render 1302×2075 | 2.70 | 10.9 MB | 68.3 KB | `ENOMEM` |
| PNG figure 981×1556 | 1.53 | 6.1 MB | 169.8 KB | `OK` |

The 68 KB failure next to the 170 KB success is what rules out payload size.

Only converted files hit it because ranger downscales an image *only if it
overflows the preview box*. On a HiDPI cell (21×41 px here) the box is
1638×2173, so a full-page PDF render at 1302×2075 fits inside it and is sent at
native resolution; the taller PNG figures overflow and get scaled under the
limit by luck. `kitty_stream.py` therefore caps previews at 1.5 MP regardless of
the box — override with `RANGER_KITTY_MAX_PIXELS` if your terminal is more or
less generous.

`rc.conf` also sets `collapse_preview false`, which addresses a separate quirk
ranger flags in `core/actions.py::get_preview` ("Previews can break when
collapse_preview is on and the preview column is popping out … on e.g. a PDF
file"). That was **not** the cause here; toggle it with `zc` if suspected.

Confirm conversion is not the problem — every converted file gets a cache entry
(named `sha1(realpath).jpg`, PNG content despite the extension):

```bash
ls -la ~/.cache/ranger/ | head
file ~/.cache/ranger/*.jpg | head -3      # expect "PNG image data, 1300x2075"
```

Entries present for the PDFs but blank previews ⇒ display, not `scope.sh`.

`collapse_preview false` in `rc.conf` addresses ranger's documented quirk above,
but it is not always the cause. Two ways to see what is actually happening:

**Ranger's own errors.** `draw_image` funnels every failure into `fm.notify`,
which flashes in the status bar. Press **`W`** inside ranger to replay the log.

**Trace the draw path.** The plugin logs every call when `RANGER_KITTY_DEBUG`
points at a file (inherited straight through `figview.sh`):

```bash
RANGER_KITTY_DEBUG=/tmp/kdebug.log ./figview.sh --browse
# select a .png, then a .pdf, then quit
cat /tmp/kdebug.log
```

An `enter` line with no following `draw` line means decoding failed; a `draw`
line with no `reply` means the terminal went silent; **no lines at all for the
PDF** means ranger never called the displayer, so the fault is upstream in its
exit-6 plumbing rather than in transfer.

**Isolate exit-6 from the image itself.** The cache entries are `.jpg`-named, so
browsing `~/.cache/ranger/` previews those exact files through the *exit-7*
path. If they render there but not as PDF previews, the images and the transport
are both fine and only ranger's converted-preview wiring is at fault.

Note the override is broader than the graphics gate: `TERM=xterm-kitty` also
makes ranger load kitty's terminfo for keys, colours and cursor sequences, which
herdr's inner emulator then has to handle. The probe only covers the graphics
handshake — if previews render but keys or colours misbehave in ranger under
herdr, suspect terminfo, not the image path.

`ueberzugpp` is not an alternative fallback — installed but broken (missing
`libvips-cpp.so.42`).

Rendering a dense-scatter PDF preview costs ~0.5 s the first time (`pdftoppm` at
120 dpi); ranger caches it afterwards.
