"""
plotting libraries 
"""
def velomap(volcano, gpscoord, region, gpsvelo1=None, gpsvelo2=None, dem=True, showplot=False):
    """ """

    import pygmt
    from pathlib import Path
    import configparser

    # config section
    config = configparser.ConfigParser()
    config.read("config.cfg")
    config.sections()

    cptfile = Path(config["Paths"]["cptfile"])
    grdfile = Path(config["Paths"]["grdfile"])
    mapdir = Path(config["Paths"]["mapdir"])
    # ---------
    velo_vektors = [
        "longitude",
        "latitude",
        "east_velo",
        "north_velo",
        "east_sigma",
        "north_sigma",
        "coorelation_EN",
        "Station",
    ]

    fig = pygmt.Figure()

    # tmp = xarray.load_dataset(grdfile)
    # print(tmp)
    # grid = pygmt.load_dataarray(grdfile)

    # pygmt.makecpt(cmap=cptfile, series=[0, 1500, 100], continuous=True)

    # Plot original grid
    fig.basemap(region=region, projection="M12c", frame=["f", "+t{}".format(volcano)])
    # for making a dem background
    if dem:
        subgrid = pygmt.grdcut(grdfile, region=region)
        grad = pygmt.grdgradient(
            subgrid, azimuth=95 / 120, normalize="t1.5"
        )  # -Ggrad.grd -A45/320 -Nt1.5

        fig.grdimage(
            grid=subgrid,
            shading=grad,
            cmap=cptfile
        )
    else:
        fig.coast(land="lightgrey")

    fig.coast(water="lightblue", shorelines="0.2,black")
    votn = Path(mapdir, "map/votn.xy")
    fig.plot(data=votn, fill="lightblue")
    fig.plot(data=gpscoord, style="d7p", pen="1p,red", fill="yellow")
    fig.text(
        x=gpscoord["latitude"],
        y=gpscoord["longitude"],
        text=gpscoord.index,
        font="5p,Helvetica-Bold",
        justify="BL",
        fill="white",
        offset="J0.2",
    )
    if gpsvelo2 is not None:
        fig.velo(
            data=gpsvelo2[velo_vektors],
            pen="1.2p,blue",
            line="0.4p,gray",
            spec="e0.02c/0.95/0",
            vector="0.5c+p2p+e+gblue",
        )

    if gpsvelo1 is not None:
        fig.velo(
            data=gpsvelo1[velo_vektors],
            pen="1.2p,red",
            line="0.4p,black",
            spec="e0.02c/0.95/0",
            vector="0.5c+p2p+e+gred",
        )

    fig.savefig("volcano_test.pdf", show=showplot)

    return
