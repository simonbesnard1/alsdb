"""Benchmarks and reference products for validating alsdb.

- :mod:`~alsdb.benchmark.lastools`, :mod:`~alsdb.benchmark.lasr`: reference
  CHMs from LAStools and lasR.
- :mod:`~alsdb.benchmark.compare`: matched scoring of stores against
  reference tiles.
- :mod:`~alsdb.benchmark.synthetic`, :mod:`~alsdb.benchmark.uls`,
  :mod:`~alsdb.benchmark.decimated`, :mod:`~alsdb.benchmark.lidarhd`: scenes
  with known truth.
- :mod:`~alsdb.benchmark.classification`: ground-classification scores.
- :mod:`~alsdb.benchmark.sensitivity`, :mod:`~alsdb.benchmark.timing`,
  :mod:`~alsdb.benchmark.points`: freeze distances, compute time, methods
  on one point set.

External tools run as separate processes and are never installed by alsdb.
Command line: ``python -m alsdb.benchmark --help``.
"""
