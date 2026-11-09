# gpsplot on yang-d01

The plot production on `yang-d01.vedur.is`, as `gpsops`, everything under
`/mnt/data/gpsops`. The laptop stays the test bed. Plan: `PLAN-yang-deploy.md` in
the gpslibrary root.

First install:

1. [SUDO-STEPS.md](SUDO-STEPS.md): packages, data folder, NFS, nginx.
2. As gpsops: clone this repo to `/mnt/data/gpsops/src/gps_plot`, check out the
   commit to run, then `deployment/yang/install.sh`.
3. On the laptop: `deployment/yang/seed-from-laptop.sh` (GIS 26 GB, `.pos`, tables,
   config). `-n` shows what it would copy.
4. As gpsops, with `set -a; . deployment/yang/yang.env; set +a`:
   `deployment/yang/update-pos-yang.sh`, then
   `gpsplot run --area svinafellsheidi --plots-dir /mnt/data/gpsops/plots --dry-run`,
   then the same without `--dry-run`.
5. `deployment/yang/install.sh --enable` starts the timer. The figures are at
   `http://yang-d01.vedur.is/gps/locations/skridurogflod/svinafellsheidi/`.

`install.sh` is safe to re-run. `pins.txt` holds the sibling commits and
`requirements.lock` the third-party versions, compiled against the laptop's env.
