"""alsdb.benchmark.gedi: block loop, skips and summaries (ingest and pipeline faked)."""

import pandas as pd

from alsdb.benchmark import gedi


class FakeDB:
    def __init__(self, storage_type, uri):
        self.uri = uri

    def create(self, crs=None, overwrite=False):
        pass

    def ingest_many(self, files, n_workers=1):
        self.files = files


def fake_run(db, out_dir, *, year, years, leaf_on, n_workers, rng):
    if out_dir.name == "empty":
        raise ValueError("no usable GEDI shot falls inside the ALS extent")
    assert year is None and years == (2019, 2023) and leaf_on is False
    profiles = pd.DataFrame({"relative_height": [0, 1], "ratio_gedi": [0.3, 0.4]})
    fits = {"waveform": {"beta": 0.1, "amplitude": 0.9, "n": 5},
            "gedi": {"beta": 0.2, "amplitude": 0.3, "n": 5}}  # fmt: skip
    return {
        "profiles": profiles,
        "fits": fits,
        "n_footprints": 5,
        "shots": pd.DataFrame(index=range(6)),
    }


def test_run_blocks_skips_and_summarises(tmp_path, monkeypatch):
    import alsdb
    import alsdb.products.gedi_attenuation as ga

    monkeypatch.setattr(alsdb, "ALSDatabase", FakeDB)
    monkeypatch.setattr(ga, "run_gedi_attenuation", fake_run)
    data = tmp_path / "data"
    for block in ("ok", "empty", "off"):
        (data / block).mkdir(parents=True)
        (data / block / "t.laz").write_bytes(b"")
    (data / "notiles").mkdir()
    sel = tmp_path / "sel.csv"
    sel.write_text("block,year,leaf_season\nok,2021,leaf-on\nempty,2021,leaf-on\n"
                   "off,2021,leaf-off\nnotiles,2021,leaf-on\nok,2022,leaf-on\n")  # fmt: skip
    out = tmp_path / "out"
    gedi.run_blocks(out, [sel], data)
    blocks = pd.read_csv(out / "blocks.csv", dtype={"block": str}).set_index("block")
    assert list(blocks.index) == [
        "ok",
        "empty",
        "off",
        "notiles",
    ]  # first occurrence of a block wins
    assert blocks.loc["ok", "status"] == "ok" and blocks.loc["ok", "gedi_amplitude"] == 0.3
    assert blocks.loc["empty", "status"].startswith("no result")
    assert blocks.loc["off", "status"] == "skipped: ALS leaf-off"
    assert blocks.loc["notiles", "status"] == "skipped: no tiles"
    profiles = pd.read_csv(out / "profiles.csv", dtype={"block": str})
    assert set(profiles.block) == {"ok"} and len(profiles) == 2
    gedi.run_blocks(out, [sel], data)  # rerun: every block done
    assert len(pd.read_csv(out / "blocks.csv")) == 4
