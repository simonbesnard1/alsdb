# Vendored: progressive TIN densification (PTD)

Source: https://github.com/r-lidar/PTD at commit
`ee9774b5f3613ccfac22aaf5f64107ab53fb7e03`, copied unchanged:

| Folder | What | Licence |
|---|---|---|
| `ptd/` | PTD ground classification (Axelsson 2000), Jean-Romain Roussel | MIT (`ptd/LICENSE.md`) |
| `hporro/` | Incremental Delaunay triangulation, Heinich Porro and Jean-Romain Roussel | MIT (`hporro/LICENSE.md`) |
| `nanoflann/` | k-d tree, Marius Muja, David G. Lowe, Jose Luis Blanco | BSD (`nanoflann/COPYING`) |

This is the same PTD that lasR and pyLASR run. alsdb compiles it into
`alsdb.methods._ptd_native` through `../ptd_binding.cpp`
(`pixi run -e spikefree build-spikefree` builds all native modules).

Changes made in alsdb are kept in `../ptd_binding.cpp`, not in these files,
so they can be updated from upstream by copying the folders again.

Axelsson, P. (2000). DEM generation from laser scanner data using adaptive TIN
models. International Archives of Photogrammetry and Remote Sensing, 33(B4),
110-117.
