"""Reference CHMs and comparisons for validating alsdb.

- :mod:`alsdb.benchmark.lastools` - LAStools reference tiles from LAS/LAZ
  (``run``) and the in-memory ``las2dem`` adapter (``rasterize_lastools``).
- :mod:`alsdb.benchmark.lasr` - lasR (r-lidar) pitfree and spike-free
  reference tiles (``run``), via the optional ``lasr`` pixi environment.
- :mod:`alsdb.benchmark.compare` - matched scoring of alsdb stores against
  any folder of reference GeoTIFF tiles.
- :mod:`alsdb.benchmark.points` - all alsdb methods on one prepared point set.

External tools run as separate processes and are never installed by alsdb.
Command line: ``python -m alsdb.benchmark {lastools,lasr,compare,points} --help``.
"""
