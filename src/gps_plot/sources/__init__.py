"""Data sources: thin adapters from the data libraries to :class:`~gps_plot.series.Series`.

A source reads (geo_dataread now, gps_api later) and returns a Series; it
does no analysis maths of its own. ``gps`` is the GNSS station/baseline
source; RTK and seismic sources follow (``docs/cli-subcommands-design.md``).
"""
