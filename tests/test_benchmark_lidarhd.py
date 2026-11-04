"""alsdb.benchmark.lidarhd: strata, leaf types, tile checks and the draw (offline)."""

import csv
import itertools

import pytest

from alsdb.benchmark import lidarhd as L


def _square(x0, y0, size):
    return {"type": "Polygon", "coordinates": [[(x0, y0), (x0 + size, y0), (x0 + size, y0 + size),
                                                (x0, y0 + size), (x0, y0)]]}  # fmt: skip


def _feature(code, x0, y0, size):
    return {"geometry": _square(x0, y0, size), "properties": {"code_tfv": code}}


def test_slope_classes():
    assert [L.slope_class(s) for s in (0, 9.9, 10, 24.9, 25, 60)] == [
        "gentle", "gentle", "moderate", "moderate", "steep", "steep",
    ]  # fmt: skip


def test_leaf_codes_and_types():
    assert L.leaf_code("FF1-09-09") == "broadleaf" and L.leaf_code("FO1") == "broadleaf"
    assert L.leaf_code("FF2G61-61") == "conifer" and L.leaf_code("FF32") == "mixed"
    assert L.leaf_code("FF0") is None and L.leaf_code("LA4") is None
    shares = L.leaf_shares([_feature("FF2-51-51", 0, 0, 1000), _feature("FF1G01-01", 1000, 0, 200),
                            _feature("LA4", 0, 1000, 1000)], 0, 0)  # fmt: skip
    assert shares["conifer"] == pytest.approx(0.25) and shares["broadleaf"] == pytest.approx(0.01)
    assert shares["forest"] == pytest.approx(0.26)  # heath is not forest
    assert L.leaf_type(shares) == "conifer"  # 0.25 / 0.26 >= 0.75
    assert L.leaf_type({"broadleaf": 0.3, "conifer": 0.3, "mixed": 0.0}) == "mixed"
    assert L.leaf_type({"broadleaf": 0.0, "conifer": 0.0, "mixed": 0.0}) is None


def _item(x0, y0, method=L.CLASSIFIED, edition="2025-01-01"):
    return {"properties": {"proj:bbox": [x0, y0, 0, x0 + 1000, y0 + 1000, 1],
                           "lidarhd:procede_classement": method,
                           "lidarhd:date_edition": edition, "lidarhd:capteur": ["S1"],
                           "lidarhd:date_debut_acquisition": "2022-03-01", "pc:count": 10},
            "assets": {"data": {"href": f"https://x/{x0}_{y0}.copc.laz"}}}  # fmt: skip


def test_block_needs_four_hand_corrected_tiles():
    full = {(x, y): _item(x, y) for x in (0, 1000) for y in (0, 1000)}
    assert L.block_tiles_ok(full, 0, 0)
    assert not L.block_tiles_ok({k: v for k, v in full.items() if k != (0, 0)}, 0, 0)
    automatic = {**full, (0, 0): _item(0, 0, method="ST_AUTO")}
    assert not L.block_tiles_ok(automatic, 0, 0)


def test_draw_fills_strata_and_keeps_blocks_apart(tmp_path):
    frame = tmp_path / "frame.csv"
    rows = []
    for i in range(40):  # alternating leaf types along x, every 10 km
        rows.append({"block": f"b{i}", "x0": i * 10_000.0, "y0": 0.0, "france_share": 1.0,
                     "land_share": 1.0, "tree_share": 0.9, "slope_mean": 5.0,
                     "slope_class": "gentle"})  # fmt: skip
    rows.append({**rows[0], "block": "farm", "x0": 999_000.0, "tree_share": 0.1})  # not eligible
    L._write_csv(frame, rows, L.FRAME_FIELDS)
    codes = {"broadleaf": "FF1", "conifer": "FF2", "mixed": "FF31"}

    def forest(x0, y0):
        leaf = L.LEAF_TYPES[int(x0 // 10_000) % 3]
        return [_feature(codes[leaf], x0, y0, L.BLOCK)]

    def tiles(x0, y0):
        return {(x, y): _item(x, y) for x in (x0, x0 + 1000) for y in (y0, y0 + 1000)}

    selected, summary = L.draw(frame, tmp_path / "sel.csv", per_stratum=2, min_distance=25_000,
                               lookup_forest=forest, lookup_tiles=tiles)  # fmt: skip
    assert sorted(r["stratum"] for r in selected) == sorted(
        f"gentle_{leaf}" for leaf in L.LEAF_TYPES for _ in range(2)
    )
    xs = sorted(r["x0"] for r in selected)
    assert min(b - a for a, b in itertools.pairwise(xs)) >= 25_000
    assert summary["moderate_broadleaf"]["selected"] == 0  # empty stratum reported, not filled
    with open(tmp_path / "sel.csv") as handle:
        assert len(list(csv.DictReader(handle))) == 6
    again, _ = L.draw(frame, tmp_path / "sel2.csv", per_stratum=2, min_distance=25_000,
                      lookup_forest=forest, lookup_tiles=tiles)  # fmt: skip
    assert [r["block"] for r in again] == [r["block"] for r in selected]  # seeded


def test_cores_sit_inside_wooded_cells():
    def cover(x0, y0, size):
        return 1.0 if x0 >= 1000 else 0.0  # only the eastern half is wooded

    cores = L.choose_cores(0.0, 0.0, n_cores=4, seed=1, tree_cover=cover)
    assert len(cores) == 4 and len(set(cores)) == 4
    for x, y in cores:
        assert x >= 1000 and (x - 25) % L.CELL == 0 and (y - 25) % L.CELL == 0
        assert x - L.MARGIN >= 0 and x + L.CORE + L.MARGIN <= L.BLOCK


def test_corrected_blocks_and_leaf_season(tmp_path):
    rows = [{"x0": x, "y0": y, "method": L.CLASSIFIED, "acquired": f"2022-0{3 + i}-01",
             "edition": "2025", "id": f"t{i}"}
            for i, (x, y) in enumerate([(0, 0), (1000, 0), (0, 1000), (1000, 1000)])]  # fmt: skip
    rows += [{"x0": 2000, "y0": 0, "method": L.CLASSIFIED, "acquired": "2022-07-01", "edition": "2025",
              "id": "lonely"}]  # fmt: skip
    rows += [{"x0": x, "y0": y, "method": "IGN_AUTO_V5", "acquired": "2023-01-01", "edition": "2025",
              "id": "auto"} for x, y in [(4000, 0), (5000, 0), (4000, 1000), (5000, 1000)]]  # fmt: skip
    index = tmp_path / "tiles.csv"
    L._write_csv(index, rows, L.INDEX_FIELDS)
    assert L.corrected_blocks(index) == {(0.0, 0.0): "2022-03-01"}
    assert L.leaf_season("2022-03-01") == "leaf-off" and L.leaf_season("2022-07-14") == "leaf-on"
