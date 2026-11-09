# yang-d01: the steps that need sudo or another host

Run once, by a person, before `install.sh`. All on yang unless stated.

## 1. Packages

```bash
sudo apt install gmt gmt-gshhg gmt-dcw ghostscript nginx nfs-common
# matplotlib usetex: the labels of the time-series figures
sudo apt install --no-install-recommends dvipng texlive-latex-extra texlive-fonts-recommended cm-super
# GMT and PyGMT open libgmt.so by that name. Ubuntu ships the link only in
# libgmt-dev, which would pull 42 packages onto a small / for this one link.
sudo ln -sf libgmt.so.6 /usr/lib/x86_64-linux-gnu/libgmt.so
```

## 2. The data folder and the account

```bash
sudo mkdir -p /mnt/data/gpsops
sudo chown gpsops:starfsmenn /mnt/data/gpsops
sudo chmod 2775 /mnt/data/gpsops
sudo loginctl enable-linger gpsops
```

Work as the account with `sudo -iu gpsops`.

## 3. Ghostscript and GMT (Ubuntu 26.04)

GMT's `gmt_<n>.ps-` scratch files are refused by the `gs` AppArmor profile
("error 79" from psconvert) without this:

```bash
echo 'owner @{HOME}/.gmt/** rwk,' | sudo tee -a /etc/apparmor.d/local/gs
sudo apparmor_parser -r /etc/apparmor.d/gs
```

## 4. NFS from okada, read-only

On okada, export both to yang (these are the paths the laptop mounts):

```
/mnt/sdc1/D    yang-d01.vedur.is(ro,no_subtree_check)
/data/D/WEEKS  yang-d01.vedur.is(ro,no_subtree_check)
```

On yang, add to `/etc/fstab`. `/mnt_data/okada` is what the code defaults to:

```
okada.vedur.is:/mnt/sdc1/D    /mnt_data/okada/D      nfs  ro,soft,timeo=50,retrans=2,nofail,_netdev,x-systemd.automount,x-systemd.mount-timeout=10,x-systemd.idle-timeout=600  0 0
okada.vedur.is:/data/D/WEEKS  /mnt_data/okada/WEEKS  nfs  ro,soft,timeo=50,retrans=2,nofail,_netdev,x-systemd.automount,x-systemd.mount-timeout=10,x-systemd.idle-timeout=600  0 0
```

```bash
sudo mkdir -p /mnt_data/okada/D /mnt_data/okada/WEEKS
sudo systemctl daemon-reload && sudo mount -a
ls /mnt_data/okada/D/SNX | head -3; ls /mnt_data/okada/WEEKS | tail -2
```

## 4b. NFS from rtk.vedur.is (the RTK plots)

Already exported to everyone. Short lines so a terminal does not break them:

```bash
sudo mkdir -p /mnt_data/rtgpsdata
O=ro,soft,timeo=50,retrans=2,nofail,_netdev
O=$O,x-systemd.automount
O=$O,x-systemd.mount-timeout=10
O=$O,x-systemd.idle-timeout=600
H=rtk.vedur.is:/home/gpsops/rtklib-run/data
echo "$H /mnt_data/rtgpsdata nfs $O 0 0" | sudo tee -a /etc/fstab
sudo systemctl daemon-reload && sudo mount -a
ls /mnt_data/rtgpsdata | head -3
```

## 5. nginx at `/gps/`

From the gps_plot checkout. Check the `allow` ranges in the file first.

```bash
sudo cp deployment/yang/nginx-gps.conf /etc/nginx/sites-available/gps
sudo ln -sf /etc/nginx/sites-available/gps /etc/nginx/sites-enabled/gps
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl reload nginx
```

## 5a. nginx: room for the aflogun install

```bash
sudo mkdir -p /etc/nginx/gps.d
sudo nginx -t && sudo systemctl reload nginx
```

## 5b. gps-config-data for gpsops (config sync, T4)

The config sync needs a clone that gpsops can pull. As gpsops, with a key
git.vedur.is accepts (if it has none, add one as a read-only deploy key on
`aut/gnss/gnss_configs/gps-config-data`):

```bash
mkdir -p ~/git
R=git@git.vedur.is:aut/gnss/gnss_configs/gps-config-data.git
git clone $R ~/git/gps-config-data
```

## 5c. gpsops may read the journal

On yang the logs of gpsops' units land in the system journal, which needs this
(for `gps-jobs -v` and looking at a failed run):

```bash
sudo gpasswd -a gpsops systemd-journal
```

## 6. Mail out (needed for T6)

postfix was installed as "Local only" (`default_transport = error`), so it
bounces everything leaving the host:

```bash
sudo postconf -e 'relayhost = [mail.vedur.is]'
sudo postconf -X default_transport relay_transport
sudo systemctl reload postfix
```
