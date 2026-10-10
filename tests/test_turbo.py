"""Turbo read-side MT (issue #32): parallel reproject warp + process-pool ingest.

- parallel _reproject_band (n_threads=4) vs serial (n_threads=1): output
  file sha256 equal, band arrays equal (A6 unit, design S3.3)
- parallel ingest vs single whole-file parse: bit-identical float64 arrays,
  original row order preserved, on a fixture with edge rows (int64 vs
  float64 inference, |x| >= 2**53, NaN/inf, quoted fields, non-adjacent
  columns) (S3.5, review F3)
- fallbacks: small files and quoted-newline files stay on the serial parse
- S3 bit-identity for all-integer columns |x| >= 2**58 (audit P0-2):
  serial vs chunk `_read_points` bit-identical; the serial float64 pin
  follows the C-parser float path per token (stateless per-token reference)
- n_threads=1 regression: the default code path is the serial one (A8)
"""

import bz2
import gzip
import hashlib
import io
import lzma
import math
import multiprocessing as mp
import shutil
import struct
import zipfile
from concurrent.futures import ProcessPoolExecutor as _ProcessPoolExecutor
from functools import partial
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin
from rasterio.warp import calculate_default_transform

from ppgrid import pipeline as pipeline_mod
from ppgrid import turbop
from ppgrid.pipeline import (
    NODATA,
    OUT_CRS,
    TILE_PX,
    WORK_CRS,
    Pipeline,
    _build_parser,
    _csv_chunk_tasks,
    _reproject_band,
    _reproject_band_array,
    _tif_parse_ifd,
    _turbo_write_parallel,
    _turbo_write_stock_parallel,
)
from ppgrid.pullpush import bin_points, box_count_banded

DATA_CSV = Path(pipeline_mod.__file__).resolve().parent.parent / "data" / "melb_houses.csv"


def _sha(p: str | Path) -> str:
    """sha256 hex digest of a file."""
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


_WANTED = ["value", "longitude", "latitude"]


def _write_src_tif(path: Path, width: int, height: int, res: float = 100.0, seed: int = 0) -> None:
    """Write a deterministic int16 source raster in the work CRS (512px blocks)."""
    rng = np.random.default_rng(seed)
    a = ((np.arange(width, dtype=np.int64) * 13 + np.arange(height, dtype=np.int64)[:, None] * 7) % 9000 + 1000).astype(
        np.int16
    )
    a[rng.random((height, width)) < 0.05] = NODATA
    transform = from_origin(0.0, height * res, res, res)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=1,
        dtype="int16",
        crs=f"EPSG:{WORK_CRS}",
        transform=transform,
        nodata=NODATA,
        compress="ZSTD",
        tiled=True,
        blockxsize=TILE_PX,
        blockysize=TILE_PX,
        predictor=2,
    ) as dst:
        dst.write(a, 1)
        dst.scales = (1.0 / 100.0,)


def test_reproject_band_mt_sha256_equal(tmp_path: Path) -> None:
    """The warp thread pool must not change a single output byte (A6 unit)."""
    src = tmp_path / "src.tif"
    _write_src_tif(src, 5000, 4000)
    with rasterio.open(src) as ds:
        dst_crs = f"EPSG:{OUT_CRS}"
        dst_transform, dst_width, dst_height = calculate_default_transform(
            ds.crs, dst_crs, ds.width, ds.height, *ds.bounds
        )
    assert dst_width > 2048  # >= 2 warp tiles per row band
    assert dst_height > 2048
    profile = {
        "driver": "GTiff",
        "dtype": "int16",
        "nodata": NODATA,
        "compress": "ZSTD",
        "tiled": True,
        "blockxsize": TILE_PX,
        "blockysize": TILE_PX,
        "predictor": 2,
        "count": 1,
    }
    out1 = tmp_path / "out1.tif"
    out4 = tmp_path / "out4.tif"
    _reproject_band(str(src), str(out1), profile, dst_crs, dst_transform, dst_width, dst_height)
    _reproject_band(str(src), str(out4), profile, dst_crs, dst_transform, dst_width, dst_height, n_threads=4)
    h1 = hashlib.sha256(out1.read_bytes()).hexdigest()
    h4 = hashlib.sha256(out4.read_bytes()).hexdigest()
    assert h1 == h4
    with rasterio.open(out1) as a, rasterio.open(out4) as b:
        assert np.array_equal(a.read(1), b.read(1))


def _write_edge_csv(path: Path, n: int) -> None:
    """Write n rows covering the per-chunk inference edge cases (review F3).

    value: floats plus int-looking rows, 2**53 +/- 1, an empty (NaN) and an
    inf row. longitude: floats plus an int-looking row, a 17-significant-
    digit row, and an empty (NaN) row. latitude: integers only (a whole-file
    parse infers int64, the chunk parses must still agree). Columns are
    non-adjacent and non-leading, with a quoted-comma note column, so the
    reader must map by name.
    """
    rng = np.random.default_rng(123)
    value = rng.normal(0.0, 100.0, n).astype(object)
    value[0] = 2**53 + 1
    value[1] = -(2**53 + 1)
    value[2] = 42
    value[3] = ""
    value[4] = np.inf
    longitude = rng.uniform(144.0, 145.0, n).astype(object)
    longitude[5] = ""
    longitude[6] = 144.123456789012345
    latitude = rng.integers(-38, -37, n)
    pd.DataFrame(
        {
            "id": np.arange(n),
            "value": value,
            "longitude": longitude,
            "latitude": latitude,
            "note": ["a,b" if i % 97 == 0 else "plain" for i in range(n)],
        },
    ).to_csv(path, index=False)


def test_ingest_mt_matches_single_parse(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Parallel ingest must be bit-identical to the single whole-file parse."""
    # Pin the pool start method to forkserver (review m11): the pytest parent
    # is multithreaded, and fork() from a multithreaded process raises
    # DeprecationWarning on Python 3.13+. forkserver parses identically
    # (same _read_csv_chunk function, same dtypes) with zero warnings.
    monkeypatch.setattr(
        pipeline_mod,
        "ProcessPoolExecutor",
        partial(_ProcessPoolExecutor, mp_context=mp.get_context("forkserver")),
    )
    csv = tmp_path / "edge.csv"
    n = 500_000
    _write_edge_csv(csv, n)
    tasks = _csv_chunk_tasks(str(csv), _WANTED, 4)
    assert tasks is not None  # the fixture must actually drive the pool
    assert len(tasks[0]) == 4  # 250k rows -> 4 chunks at n_threads=4

    # Reference: the single whole-file parse, exactly as the serial path does it.
    df = pd.read_csv(csv, usecols=_WANTED)
    v_ref = df["value"].to_numpy(dtype=np.float64)
    lon_ref = df["longitude"].to_numpy(dtype=np.float64)
    lat_ref = df["latitude"].to_numpy(dtype=np.float64)
    good = np.isfinite(v_ref) & np.isfinite(lon_ref) & np.isfinite(lat_ref)
    v_ref, lon_ref, lat_ref = v_ref[good], lon_ref[good], lat_ref[good]
    tr = Transformer.from_crs(4326, 6933, always_xy=True)
    x_ref, y_ref = tr.transform(lon_ref, lat_ref)

    prev = None
    for nt in (1, 4):
        p = Pipeline(str(csv), "value", "longitude", "latitude", str(tmp_path / f"out{nt}"), n_threads=nt)
        p.ingest()
        assert p.n == v_ref.size
        assert np.array_equal(p.v, v_ref)  # bit-identical values, original row order
        assert np.array_equal(p.x, np.asarray(x_ref))
        assert np.array_equal(p.y, np.asarray(y_ref))
        if prev is not None:
            assert np.array_equal(prev[0], p.v)
            assert np.array_equal(prev[1], p.x)
            assert np.array_equal(prev[2], p.y)
        prev = (p.v, p.x, p.y)


def test_ingest_mt_fallbacks(tmp_path: Path) -> None:
    """Small files stay on the serial parse and still ingest correctly."""
    rng = np.random.default_rng(7)
    small = tmp_path / "small.csv"
    pd.DataFrame(
        {
            "value": rng.normal(0.0, 1.0, 100),
            "longitude": rng.uniform(144.0, 145.0, 100),
            "latitude": rng.uniform(-38.0, -37.0, 100),
        },
    ).to_csv(small, index=False)
    assert _csv_chunk_tasks(str(small), _WANTED, 4) is None  # below the min chunk size
    p = Pipeline(str(small), "value", "longitude", "latitude", str(tmp_path / "o1"), n_threads=4)
    p.ingest()
    assert p.n == 100


def test_ingest_mt_quoted_newline_refuses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A quoted field with an embedded newline must refuse the split (serial parse)."""
    monkeypatch.setattr(pipeline_mod, "_INGEST_MIN_CHUNK_ROWS", 1)
    qnl = tmp_path / "qnl.csv"
    qnl.write_text(
        'value,longitude,latitude,note\n1.5,144.1,-37.5,"line1\nline2"\n2.5,144.2,-37.6,plain\n',
        encoding="utf-8",
    )
    assert _csv_chunk_tasks(str(qnl), _WANTED, 4) is None  # chunk boundary splits a quoted field
    p = Pipeline(str(qnl), "value", "longitude", "latitude", str(tmp_path / "o2"), n_threads=4)
    p.ingest()
    assert p.n == 2  # the serial parse reads both rows
    assert np.array_equal(p.v, np.array([1.5, 2.5]))


def _write_value_csv(path: Path, value_toks: list[str]) -> None:
    """Write a 3-column CSV (value, longitude, latitude).

    Value column holds the given raw tokens; coordinates are small
    deterministic floats (|x| < 2**53), off the audit P0-2 pattern.
    """
    rng = np.random.default_rng(31)
    n = len(value_toks)
    lon = rng.uniform(144.0, 145.0, n)
    lat = rng.uniform(-38.0, -37.0, n)
    lines = ["value,longitude,latitude"]
    for t, lo, la in zip(value_toks, lon, lat, strict=True):
        lines.append(f"{t},{float(lo)!r},{float(la)!r}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _big_int_value_toks() -> list[str]:
    """Return the audit P0-2 (2026-10-09) pattern value tokens.

    All-integer column in 2**58-2048..2**58+2047 (where pandas' C-parser
    float path is not correctly rounded) plus one 2**63-1 literal.
    """
    return [str(x) for x in range(2**58 - 2048, 2**58 + 2048)] + [str(2**63 - 1)]


def test_ingest_bigint_serial_bit_identical_to_chunk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """S3 regression pin (audit P0-2, 2026-10-09, decision (a)).

    All-integer value column with |x| >= 2**58: n_threads=1 must be
    bit-identical to n_threads=4 (np.array_equal, NOT allclose). Pre-fix,
    the serial read inferred int64 (correctly rounded int64->f64) while the
    chunk read pinned float64 (C-parser float path), so the audit measured
    90,165 of 250k rows 1-2 ULP apart at n_threads=1 vs 4.
    """
    monkeypatch.setattr(
        pipeline_mod,
        "ProcessPoolExecutor",
        partial(_ProcessPoolExecutor, mp_context=mp.get_context("forkserver")),
    )
    monkeypatch.setattr(pipeline_mod, "_INGEST_MIN_CHUNK_ROWS", 1)
    csv = tmp_path / "bigint.csv"
    toks = _big_int_value_toks()
    _write_value_csv(csv, toks)
    tasks = _csv_chunk_tasks(str(csv), _WANTED, 4)
    assert tasks is not None  # the fixture must actually drive the pool
    assert len(tasks[0]) == 4

    p1 = Pipeline(str(csv), "value", "longitude", "latitude", str(tmp_path / "o1"), n_threads=1)
    v1, lon1, lat1 = p1._read_points(_WANTED)  # ruff: ignore[private-member-access]
    p4 = Pipeline(str(csv), "value", "longitude", "latitude", str(tmp_path / "o4"), n_threads=4)
    v4, lon4, lat4 = p4._read_points(_WANTED)  # ruff: ignore[private-member-access]

    assert v1.shape == (len(toks),)
    assert np.array_equal(v1, v4)  # bit-identical values, original row order
    assert np.array_equal(lon1, lon4)
    assert np.array_equal(lat1, lat4)
    assert v4[-1] == 2**63 + 2048  # the 2**63-1 literal lands last, C-path value


def test_ingest_bigint_serial_takes_c_float_path(tmp_path: Path) -> None:
    """Serial read pins float64: n_threads=1 takes the C-parser float path.

    Per token, exactly as the chunk workers do (audit P0-2, 2026-10-09,
    decision (a)). Deliberate baseline change vs 0.4.1: pre-fix, an
    all-integer column inferred int64 and converted with correct rounding,
    so the serial value was float(token) for every literal. Post-fix, both
    thread counts parse through the C-parser float path, which is not
    correctly rounded in parts of the int64 range: on this fixture 1478 of
    4097 literals (pandas 2.3.3) differ from float(token) by 1-2 ULP, e.g.
    the 2**63-1 literal now parses to 0x43e0000000000001 (2**63+2048)
    instead of the correctly rounded 0x43e0000000000000 (2**63).
    """
    csv = tmp_path / "bigint.csv"
    toks = _big_int_value_toks()
    _write_value_csv(csv, toks)

    p = Pipeline(str(csv), "value", "longitude", "latitude", str(tmp_path / "o1"), n_threads=1)
    v, _, _ = p._read_points(_WANTED)  # ruff: ignore[private-member-access]

    # Reference: the C-parser float path on each token in isolation (one
    # token per file, fresh tokenizer — xstrtod is stateless per token, so
    # this is the value the column parse must produce).
    per_tok = np.empty(len(toks), dtype=np.float64)
    for i, t in enumerate(toks):
        per_tok[i] = pd.read_csv(io.BytesIO(f"value\n{t}\n".encode()), dtype={"value": np.float64})["value"].iloc[0]
    assert np.array_equal(v, per_tok)

    exact = np.array([float(t) for t in toks], dtype=np.float64)
    ulp = np.abs(v.view(np.int64) - exact.view(np.int64))
    assert ulp.max() <= 2  # C path is at most 2 ULP off correct rounding here
    assert int((v != exact).sum()) == 1478  # pandas 2.3.3: the announced change set
    assert v[-1] == 2**63 + 2048  # audit headline: pre-fix the serial gave 2**63


def test_ingest_float_pin_no_change_off_pattern(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Guard: the float64 pin changes nothing off the rare pattern.

    (i) One decimal among integers: pre-fix, the serial parse already
    inferred float64 and took the C float path on every token, so the pin
    is a no-op — nt=1 and nt=4 must equal the plain unpinned parse.
    (ii) All-integer column with |x| < 2**53: int64->f64 is exact there and
    the C float path agrees with it, so the pin is a no-op — nt=1 values
    must equal float(token) (the 0.4.1 baseline) and nt=4 must match nt=1.
    """
    monkeypatch.setattr(
        pipeline_mod,
        "ProcessPoolExecutor",
        partial(_ProcessPoolExecutor, mp_context=mp.get_context("forkserver")),
    )
    monkeypatch.setattr(pipeline_mod, "_INGEST_MIN_CHUNK_ROWS", 1)

    # (i) one decimal among integers
    toks = ["1.5", "2", "3", "4", "5", "288230376151711744.0", "9223372036854775807", "-7", "0", "42"]
    mixed = tmp_path / "mixed.csv"
    _write_value_csv(mixed, toks)
    v_ref = pd.read_csv(mixed, usecols=_WANTED)["value"].to_numpy(dtype=np.float64)
    p1 = Pipeline(str(mixed), "value", "longitude", "latitude", str(tmp_path / "m1"), n_threads=1)
    v1, _, _ = p1._read_points(_WANTED)  # ruff: ignore[private-member-access]
    p4 = Pipeline(str(mixed), "value", "longitude", "latitude", str(tmp_path / "m4"), n_threads=4)
    v4, _, _ = p4._read_points(_WANTED)  # ruff: ignore[private-member-access]
    assert np.array_equal(v1, v_ref)  # no behaviour change vs the 0.4.1 serial parse
    assert np.array_equal(v1, v4)

    # (ii) all-integer, |x| < 2**53 (plus the 2**53 boundary literals)
    rng = np.random.default_rng(21)
    small_toks = [str(x) for x in rng.integers(-(2**53), 2**53, 2000).tolist()] + [
        str(2**53 - 1),
        str(2**53),
        str(2**53 + 1),
    ]
    scsv = tmp_path / "smallint.csv"
    _write_value_csv(scsv, small_toks)
    p1 = Pipeline(str(scsv), "value", "longitude", "latitude", str(tmp_path / "s1"), n_threads=1)
    v1, _, _ = p1._read_points(_WANTED)  # ruff: ignore[private-member-access]
    p4 = Pipeline(str(scsv), "value", "longitude", "latitude", str(tmp_path / "s4"), n_threads=4)
    v4, _, _ = p4._read_points(_WANTED)  # ruff: ignore[private-member-access]
    assert np.array_equal(v1, np.array([float(t) for t in small_toks], dtype=np.float64))
    assert np.array_equal(v1, v4)


# ---------------------------------------------------------------------------
# Issue #80: compressed CSV + planner MemoryError
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def big_csv_pair(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """250k-row CSV (raw + gzipped) shared by the #80 fallback tests."""
    n = 250_000
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "value": rng.random(n),
            "longitude": 144.0 + rng.random(n) * 0.2,
            "latitude": -38.0 + rng.random(n) * 0.2,
        },
    )
    d = tmp_path_factory.mktemp("bigcsv")
    raw = d / "big.csv"
    df.to_csv(raw, index=False)
    gz = d / "big.csv.gz"
    with raw.open("rb") as fin, gzip.open(gz, "wb") as fout:
        shutil.copyfileobj(fin, fout)
    return raw, gz


def test_csv_gz_chunk_planner_sniffs_magic(big_csv_pair: tuple[Path, Path]) -> None:
    """#80: gzip magic in the planner -> None (serial fallback), not a byte-offset plan."""
    raw, gz = big_csv_pair
    assert _csv_chunk_tasks(str(gz), _WANTED, 4) is None
    # The uncompressed file still plans (guard against an over-broad sniff).
    tasks = _csv_chunk_tasks(str(raw), _WANTED, 4)
    assert tasks is not None
    assert tasks[1] == 250_000


def test_csv_gz_n_threads_falls_back_to_serial(tmp_path: Path, big_csv_pair: tuple[Path, Path]) -> None:
    """#80: .csv.gz + n_threads=4 parses via the serial fallback (was UnicodeDecodeError).

    Row count and values must match the n_threads=1 serial read exactly.
    """
    _raw, gz = big_csv_pair
    n = 250_000
    p4 = Pipeline(str(gz), "value", "longitude", "latitude", str(tmp_path / "o4"), n_threads=4)
    p4.ingest()
    p1 = Pipeline(str(gz), "value", "longitude", "latitude", str(tmp_path / "o1"), n_threads=1)
    p1.ingest()
    assert p4.n == n
    assert p1.n == n
    np.testing.assert_array_equal(p4.v, p1.v)
    np.testing.assert_array_equal(p4.x, p1.x)
    np.testing.assert_array_equal(p4.y, p1.y)


@pytest.fixture(scope="module")
def compressed_csv_formats(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """CSV compressed as gzip/bzip2/zip/xz (review M-1: xz sniff was dead code)."""
    n = 5_000
    rng = np.random.default_rng(7)
    df = pd.DataFrame(
        {
            "value": rng.random(n),
            "longitude": 144.0 + rng.random(n) * 0.2,
            "latitude": -38.0 + rng.random(n) * 0.2,
        },
    )
    d = tmp_path_factory.mktemp("csvfmt")
    raw = d / "data.csv"
    df.to_csv(raw, index=False)
    rb = raw.read_bytes()
    out: dict[str, Path] = {}
    out["raw"] = raw
    out["gz"] = raw.parent / "data.csv.gz"
    out["gz"].write_bytes(gzip.compress(rb))
    out["bz2"] = d / "data.csv.bz2"
    out["bz2"].write_bytes(bz2.compress(rb))
    out["zip"] = d / "data.zip"
    with zipfile.ZipFile(out["zip"], "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("data.csv", rb)
    out["xz"] = d / "data.csv.xz"
    out["xz"].write_bytes(lzma.compress(rb, preset=1))
    return out


@pytest.mark.parametrize("fmt", ["gz", "bz2", "zip", "xz"])
def test_csv_compressed_planner_sniffs_magic(compressed_csv_formats: dict[str, Path], fmt: str) -> None:
    """#80 (review M-1): every compressed format must fall back to serial.

    The xz magic is 5 bytes (fd 37 7a 58 5a); the original check compared it
    against data[:4], so xz files passed the planner and crashed in the worker.
    """
    d = compressed_csv_formats
    assert _csv_chunk_tasks(str(d[fmt]), _WANTED, 4) is None


def test_csv_xz_n_threads_falls_back_to_serial(tmp_path: Path, compressed_csv_formats: dict[str, Path]) -> None:
    """#80 (review M-1): .csv.xz + n_threads=4 parses via the serial fallback.

    End-to-end proof the dead xz sniff is fixed: pre-fix this raised
    UnicodeDecodeError on byte 0xfd (the xz magic) in the chunk worker.
    """
    xz = compressed_csv_formats["xz"]
    n = 5_000
    p4 = Pipeline(str(xz), "value", "longitude", "latitude", str(tmp_path / "o4"), n_threads=4)
    p4.ingest()
    p1 = Pipeline(str(xz), "value", "longitude", "latitude", str(tmp_path / "o1"), n_threads=1)
    p1.ingest()
    assert p4.n == n == p1.n
    np.testing.assert_array_equal(p4.v, p1.v)
    np.testing.assert_array_equal(p4.x, p1.x)
    np.testing.assert_array_equal(p4.y, p1.y)


def test_chunk_planner_memoryerror_falls_back_to_serial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, big_csv_pair: tuple[Path, Path]
) -> None:
    """#80: MemoryError during planning (2x file-size allocation) -> serial parse, no crash."""
    raw, _gz = big_csv_pair
    n = 250_000
    real_read_bytes = Path.read_bytes
    state = {"raised": False}

    def boom(self: Path) -> bytes:
        if not state["raised"] and str(self) == str(raw):
            state["raised"] = True
            raise MemoryError
        return real_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", boom)
    p = Pipeline(str(raw), "value", "longitude", "latitude", str(tmp_path / "omem"), n_threads=4)
    p.ingest()
    assert state["raised"]  # the planner really hit the MemoryError
    assert p.n == n


def test_pipeline_mt_end_to_end_sha256_equal(tmp_path: Path) -> None:
    """n_threads=4 must not change a single output byte of a full run (A8)."""
    rng = np.random.default_rng(1)
    n = 40
    csv = tmp_path / "pts.csv"
    pd.DataFrame(
        {
            "value": rng.uniform(1.0, 10.0, n),
            "longitude": 144.6 + rng.uniform(-0.05, 0.05, n),
            "latitude": -37.7 + rng.uniform(-0.05, 0.05, n),
        },
    ).to_csv(csv, index=False)

    digests = {}
    for nt in (1, 4):
        out = tmp_path / f"o{nt}"
        p = Pipeline(
            str(csv),
            "value",
            "longitude",
            "latitude",
            str(out),
            res=500.0,
            cap_km=10.0,
            workers=1,
            skip_calibration=True,
            n_threads=nt,
        )
        vpath, spath = p.run()
        digests[nt] = (
            hashlib.sha256(Path(vpath).read_bytes()).hexdigest(),
            hashlib.sha256(Path(spath).read_bytes()).hexdigest(),
        )
    assert digests[1] == digests[4]


def test_pipeline_n_threads_validation() -> None:
    """n_threads must be >= 1."""
    with pytest.raises(ValueError, match="n_threads"):
        Pipeline("in.csv", "value", "longitude", "latitude", "out", n_threads=0)


# ---------------------------------------------------------------------------
# Issue #33: turbo write path (S3.4a) + budget cap (S3.3) + CLI (S3.6)
# ---------------------------------------------------------------------------

# A9 anchor geometry (design 3.6.1): full-AU 16M pts @ 100 m, cap 25 km.
_A9_CELLS = 1_592_524_800  # Wc 41472 x ny_padded 38400
_A9_WC = 41472
_A9_RADIUS = 250  # cap_km 25 -> radius cells


def _identity_cap(preset: float | None, *_args: float | None, **_kwargs: float | None) -> tuple[float | None, str]:
    """resolve_cap stub: the explicit preset stands, source labelled 'explicit'."""
    return preset, "explicit"


def _a9_pipeline(tmp_path: Path, cap: float, *, strict: bool = False) -> Pipeline:
    """Build a full-AU-geometry Pipeline with the A9 anchor grid (no run)."""
    p = Pipeline(
        str(DATA_CSV),
        "price",
        "longitude",
        "latitude",
        str(tmp_path / f"o_{cap}_{int(strict)}"),
        res=100.0,
        cap_km=25.0,
        turbo=True,
        turbo_cap_gb=cap,
        turbo_strict=strict,
    )
    p.n, p.nx_padded, p.ny_padded, p.res, p.cap_km_val = (
        400_000,
        _A9_WC,
        _A9_CELLS // _A9_WC,
        100.0,
        25,
    )
    return p


def test_a9_pin_20gb_regime_c() -> None:
    """20 GB full-AU: regime C, c=2, b=2, est ~16.6 GB, shared, exit 0."""
    p = turbop.plan(20.0, _A9_CELLS, _A9_WC, _A9_RADIUS)
    assert p.regime == "C"
    assert p.box_chunks == 2
    assert p.descent_bands == 2
    assert p.workers == 2
    assert abs(p.est_peak_bytes / 1e9 - 16.6) < 0.2
    d = turbop.precheck(p)
    assert d.path == "shared"
    assert d.exit_code == 0
    assert "--turbo: cap" in d.summary  # the budget value used is printed
    assert "budget 0.85*C = 17.0 GB" in d.summary


def test_a9_pin_16gb_per_box() -> None:
    """16 GB full-AU: per_box, est ~4.7 GB; non-strict warns + exit 0, strict exit 3."""
    p = turbop.plan(16.0, _A9_CELLS, _A9_WC, _A9_RADIUS)
    assert p.regime == "per_box"
    assert abs(p.est_peak_bytes / 1e9 - 4.7) < 0.2
    d = turbop.precheck(p)
    assert d.path == "per_box"
    assert d.exit_code == 0  # [warn] + continue on the per-box path
    d3 = turbop.precheck(p, strict=True)
    assert d3.exit_code == 3


def test_a9_pin_32gb_regime_b() -> None:
    """32 GB full-AU: regime B, c=3, b=4, workers 3, est ~27.2 GB, exit 0."""
    p = turbop.plan(32.0, _A9_CELLS, _A9_WC, _A9_RADIUS)
    assert p.regime == "B"
    assert p.box_chunks == 3
    assert p.descent_bands == 4
    assert p.workers == 3
    assert abs(p.est_peak_bytes / 1e9 - 27.2) < 0.2
    assert turbop.precheck(p).exit_code == 0


def test_turbo_precheck_strict_exit3(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--turbo-strict: shared infeasible (per_box) -> SystemExit(3)."""
    monkeypatch.setattr(pipeline_mod.turbop, "resolve_cap", _identity_cap)
    p = _a9_pipeline(tmp_path, 16.0, strict=True)
    with pytest.raises(SystemExit) as ei:
        p._turbo_precheck()  # ruff: ignore[private-member-access]
    assert ei.value.code == 3
    assert p._turbo_decision is not None  # ruff: ignore[private-member-access]
    assert p._turbo_decision.path == "per_box"  # ruff: ignore[private-member-access]


def test_turbo_precheck_nonstrict_continues(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-strict: per_box warns and continues; the 0.85*C budget is stored."""
    monkeypatch.setattr(pipeline_mod.turbop, "resolve_cap", _identity_cap)
    p = _a9_pipeline(tmp_path, 16.0, strict=False)
    p._turbo_precheck()  # ruff: ignore[private-member-access] - must not raise
    assert p._turbo_decision is not None  # ruff: ignore[private-member-access]
    assert p._turbo_decision.path == "per_box"  # ruff: ignore[private-member-access]
    assert p._turbo_decision.exit_code == 0  # ruff: ignore[private-member-access]
    assert p._turbo_budget_bytes == turbop.precheck_budget_bytes(16.0)  # ruff: ignore[private-member-access]


def test_resolve_turbo_cap_cgroup_wired(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """M-2: the precheck resolves the cap with physical + cgroup (clamp-aware)."""
    monkeypatch.setattr(pipeline_mod.turbop, "_read_physical_gb", lambda: 134.9)
    monkeypatch.setattr(pipeline_mod.turbop, "_read_cgroup_max_gb", lambda: 24.0)
    p = Pipeline(str(DATA_CSV), "price", "longitude", "latitude", str(tmp_path / "o"), turbo=True, turbo_cap_gb=32.0)
    cap, source = p._resolve_turbo_cap()  # ruff: ignore[private-member-access]
    assert cap == 16.0  # min(134.9, 24) - 8: the slice limit binds
    assert "cgroup 24 GB" in source
    p.turbo_cap_gb = None  # auto path, same wired values
    cap, source = p._resolve_turbo_cap()  # ruff: ignore[private-member-access]
    assert cap == 16.0
    assert source.startswith("auto")


def test_turbo_precheck_per_box_peak(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """M-3: the precheck hands the worst-case per-box est to turbop.precheck."""
    monkeypatch.setattr(pipeline_mod.turbop, "resolve_cap", _identity_cap)
    seen: dict[str, object] = {}
    orig = pipeline_mod.turbop.precheck

    def spy(plan: turbop.TurboPlan, **kw: object) -> Any:
        seen.update(kw)
        return orig(plan, **kw)

    monkeypatch.setattr(pipeline_mod.turbop, "precheck", spy)
    p = _a9_pipeline(tmp_path, 16.0)
    # Pre-grid: box geometry unknown -> no estimate (legacy message path).
    p._turbo_precheck()  # ruff: ignore[private-member-access]
    assert seen["per_box_peak_bytes"] is None
    # Post-grid: worst snapped box x in-flight + points + in-RAM DN field.
    p.bsize, p.halo, p.step = 2048, 256, 256
    p.nx, p.ny = _A9_WC, _A9_CELLS // _A9_WC
    p.tasks = [(i, 0) for i in range(1000)]
    p._turbo_precheck()  # ruff: ignore[private-member-access]
    worst = -(-(2048 + 2 * 256) // 256) * 256  # 2560
    expect = turbop.per_box_peak_bytes(
        worst * worst, in_flight=min(p.workers, 1000), extra_bytes=p.n * 32 + p.nx * p.ny * 4
    )
    assert seen["per_box_peak_bytes"] == expect
    assert p._turbo_decision is not None  # ruff: ignore[private-member-access]
    assert p._turbo_decision.path == "per_box"  # ruff: ignore[private-member-access]


def test_turbo_cap_gb_validation() -> None:
    """turbo_cap_gb must be > 0."""
    with pytest.raises(ValueError, match="turbo_cap_gb"):
        Pipeline(str(DATA_CSV), "price", "longitude", "latitude", "out", turbo=True, turbo_cap_gb=0.0)


def _demo_field(width: int, height: int, res: float, seed: int) -> tuple[np.ndarray, Any]:
    """Deterministic int16 work-CRS field with a 5% NODATA scatter."""
    rng = np.random.default_rng(seed)
    a = ((np.arange(width, dtype=np.int64) * 13 + np.arange(height, dtype=np.int64)[:, None] * 7) % 9000 + 100).astype(
        np.int16
    )
    a[rng.random((height, width)) < 0.05] = NODATA
    return a, from_origin(0.0, height * res, res, res)


def _dst_grid(width: int, height: int, res: float) -> tuple[Any, int, int]:
    """Default output-CRS grid for a work-CRS field of the given size."""
    work_crs, dst_crs = f"EPSG:{WORK_CRS}", f"EPSG:{OUT_CRS}"
    return calculate_default_transform(work_crs, dst_crs, width, height, 0.0, 0.0, width * res, height * res)


def _demo_profile() -> dict[str, Any]:
    """Return the base write profile (zstd tiled int16), identical to the pipeline's."""
    return {
        "driver": "GTiff",
        "dtype": "int16",
        "nodata": NODATA,
        "compress": "ZSTD",
        "tiled": True,
        "blockxsize": TILE_PX,
        "blockysize": TILE_PX,
        "predictor": 2,
        "count": 1,
    }


def test_array_reproject_matches_file_reproject(tmp_path: Path) -> None:
    """_reproject_band_array == _reproject_band on the same work-CRS file (A6 unit)."""
    a, xform = _demo_field(3000, 2400, 100.0, seed=42)
    dst_transform, dst_width, dst_height = _dst_grid(3000, 2400, 100.0)
    work_crs, dst_crs = f"EPSG:{WORK_CRS}", f"EPSG:{OUT_CRS}"
    tags = {"transform": "identity", "res_m": "100.0"}
    scales, offsets = (1.0 / 100.0,), (0.0,)
    work = tmp_path / "work.tif"
    with rasterio.open(
        work,
        "w",
        driver="GTiff",
        width=a.shape[1],
        height=a.shape[0],
        count=1,
        dtype="int16",
        crs=work_crs,
        transform=xform,
        nodata=NODATA,
        compress="ZSTD",
        tiled=True,
        blockxsize=TILE_PX,
        blockysize=TILE_PX,
        predictor=2,
    ) as dst:
        dst.write(a, 1)
        dst.update_tags(**tags)
        dst.scales = scales
        dst.offsets = offsets
    out_file = tmp_path / "out_file.tif"
    out_arr = tmp_path / "out_arr.tif"
    _reproject_band(str(work), str(out_file), _demo_profile(), dst_crs, dst_transform, dst_width, dst_height)
    _reproject_band_array(
        a,
        xform,
        work_crs,
        str(out_arr),
        _demo_profile(),
        dst_crs,
        dst_transform,
        dst_width,
        dst_height,
        tags=tags,
        scales=scales,
        offsets=offsets,
        n_threads=4,
    )
    assert _sha(out_file) == _sha(out_arr)


def _stock_frames(path: Path) -> list[bytes]:
    """Read the stored (compressed) tile frames of a tiled GTiff in raster-scan order."""
    data = path.read_bytes()
    info = _tif_parse_ifd(data)
    n = info["n_tiles"]
    # Per-tag slot widths (Y-1): BigTIFF 324 is LONG8 (8 B), 325 is LONG (4 B).
    offs = struct.unpack_from(info["e"] + f"{n}{'Q' if info['sz324'] == 8 else 'I'}", data, info["t324"][2])
    sizes = struct.unpack_from(info["e"] + f"{n}{'Q' if info['sz325'] == 8 else 'I'}", data, info["t325"][2])
    frames = [data[o : o + s] for o, s in zip(offs, sizes, strict=True)]
    assert len(frames) == n
    assert all(f[:4] == b"\x28\xb5\x2f\xfd" for f in frames)  # zstd magic
    return frames


def test_turbo_write_parallel_matches_serial(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Patched-head parallel assembly matches the serial array reproject.

    Simulates an oracle pass: compress_tiles returns the exact stock codec
    frames, so the assembled file must be byte-identical (A6 unit).
    """
    a, xform = _demo_field(1600, 1400, 100.0, seed=11)
    dst_transform, dst_width, dst_height = _dst_grid(1600, 1400, 100.0)
    work_crs, dst_crs = f"EPSG:{WORK_CRS}", f"EPSG:{OUT_CRS}"
    tags = {"transform": "identity", "res_m": "100.0"}
    scales, offsets = (1.0 / 100.0,), (0.0,)
    out_ser = tmp_path / "ser.tif"
    _reproject_band_array(
        a,
        xform,
        work_crs,
        str(out_ser),
        _demo_profile(),
        dst_crs,
        dst_transform,
        dst_width,
        dst_height,
        tags=tags,
        scales=scales,
        offsets=offsets,
        n_threads=1,
    )
    frames = iter(_stock_frames(out_ser))

    def stub(tiles: list[np.ndarray], n_threads: int = 1) -> list[bytes]:  # ruff: ignore[unused-function-argument]
        """Stock-frame stand-in: consume the real frames in raster-scan order."""
        return [next(frames) for _ in tiles]

    monkeypatch.setattr(pipeline_mod.zstdmt, "compress_tiles", stub)
    out_par = tmp_path / "par.tif"
    ok = _turbo_write_parallel(
        a,
        xform,
        work_crs,
        str(out_par),
        _demo_profile(),
        dst_crs,
        dst_transform,
        dst_width,
        dst_height,
        tags=tags,
        scales=scales,
        offsets=offsets,
        n_threads=4,
    )
    assert ok
    assert _sha(out_ser) == _sha(out_par)
    with rasterio.open(out_par) as ds:
        assert ds.scales == scales
        for k, v in tags.items():  # GDAL auto-adds AREA_OR_POINT; ours must be present
            assert ds.tags()[k] == v


def test_turbo_write_parallel_bigtiff_matches_serial(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """BigTIFF heads patch per-tag slot widths; output is byte-identical (Y-1).

    Same oracle pass as the classic test, but the reference head is a
    real BigTIFF: tag 324 is LONG8 (8-byte slots) while 325 is LONG
    (4-byte slots). Patching both at one stride corrupts the 325 region
    (the pre-fix bug); per-tag strides keep the file byte-identical.

    """
    a, xform = _demo_field(1600, 1400, 100.0, seed=11)
    dst_transform, dst_width, dst_height = _dst_grid(1600, 1400, 100.0)
    work_crs, dst_crs = f"EPSG:{WORK_CRS}", f"EPSG:{OUT_CRS}"
    tags = {"transform": "identity", "res_m": "100.0"}
    scales, offsets = (1.0 / 100.0,), (0.0,)
    profile = dict(_demo_profile(), BIGTIFF="YES")
    out_ser = tmp_path / "ser.tif"
    _reproject_band_array(
        a,
        xform,
        work_crs,
        str(out_ser),
        profile,
        dst_crs,
        dst_transform,
        dst_width,
        dst_height,
        tags=tags,
        scales=scales,
        offsets=offsets,
        n_threads=1,
    )
    info = _tif_parse_ifd(out_ser.read_bytes())
    assert info["big"] is True  # the reference head really is BigTIFF
    assert info["sz324"] == 8  # 324 = LONG8
    assert info["sz325"] == 4  # 325 = LONG
    frames = iter(_stock_frames(out_ser))

    def stub(tiles: list[np.ndarray], n_threads: int = 1) -> list[bytes]:  # ruff: ignore[unused-function-argument]
        """Stock-frame stand-in: consume the real frames in raster-scan order."""
        return [next(frames) for _ in tiles]

    monkeypatch.setattr(pipeline_mod.zstdmt, "compress_tiles", stub)
    out_par = tmp_path / "par.tif"
    ok = _turbo_write_parallel(
        a,
        xform,
        work_crs,
        str(out_par),
        profile,
        dst_crs,
        dst_transform,
        dst_width,
        dst_height,
        tags=tags,
        scales=scales,
        offsets=offsets,
        n_threads=4,
    )
    assert ok
    assert _sha(out_ser) == _sha(out_par)


def test_turbo_write_parallel_edge_tiles_padded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Audit #81 F-1: partial edge tiles are zero-padded to TILE_PX^2.

    GDAL pads a 300-wide edge tile to the full 512^2 block with zeros
    before the codec runs; the turbo zstd writer used to compress the
    unpadded slice, so its frames decompress to less than the declared
    tile size (S3 bit-identity break on non-512-multiple rasters). The
    stock reference is a direct GDAL write of the same content; the
    turbo assembly (stubbed with the stock frames) must match it byte
    for byte, and every tile reaching the compressor must be full-size
    with the exact zero padding.

    """
    height, width = 700, 900  # 2x2 tile grid: right + bottom edges partial
    assert width % TILE_PX != 0
    assert height % TILE_PX != 0
    a, xform = _demo_field(width, height, 100.0, seed=31)
    crs = f"EPSG:{WORK_CRS}"
    # Stock reference: same write recipe as the turbo reference head
    # (_reproject_core), identity warp so the stored content is exactly a.
    out_ser = tmp_path / "ser.tif"
    _reproject_band_array(
        a,
        xform,
        crs,
        str(out_ser),
        _demo_profile(),
        crs,
        xform,
        width,
        height,
        tags={},
        scales=(1.0,),
        offsets=(0.0,),
        n_threads=1,
    )
    with rasterio.open(out_ser) as ds:
        assert np.array_equal(ds.read(1), a)  # identity warp is bit-exact
    stock_frames = _stock_frames(out_ser)
    assert len(stock_frames) == 4  # 2x2 tile grid, all zstd
    frames = iter(stock_frames)
    raw_seen: list[np.ndarray] = []

    def spy_predictor2(t: np.ndarray) -> np.ndarray:
        raw_seen.append(t)
        return t  # content-agnostic: the stubbed frames carry the bytes

    def stub(tiles: list[np.ndarray], n_threads: int = 1) -> list[bytes]:  # ruff: ignore[unused-function-argument]
        return [next(frames) for _ in tiles]

    monkeypatch.setattr(pipeline_mod.zstdmt, "predictor2", spy_predictor2)
    monkeypatch.setattr(pipeline_mod.zstdmt, "compress_tiles", stub)
    out_par = tmp_path / "par.tif"
    ok = _turbo_write_parallel(
        a,
        xform,
        crs,
        str(out_par),
        _demo_profile(),
        crs,
        xform,
        width,
        height,
        tags={},
        scales=(1.0,),
        offsets=(0.0,),
        n_threads=2,
    )
    assert ok
    assert _sha(out_ser) == _sha(out_par)
    # 2x2 tile grid, raster-scan order; all four reach the compressor
    # as full TILE_PX^2 blocks (identity warp: content == a's slices).
    assert len(raw_seen) == 4
    assert all(t.shape == (TILE_PX, TILE_PX) for t in raw_seen)
    # Full top-left tile: unchanged by the padding.
    assert np.array_equal(raw_seen[0], a[:TILE_PX, :TILE_PX])
    # Top-right tile: partial width 388, zero-padded to 512.
    t = raw_seen[1]
    w = width - TILE_PX
    assert np.array_equal(t[:, :w], a[:TILE_PX, TILE_PX : TILE_PX + w])
    assert (t[:, w:] == 0).all()
    # Bottom-right tile: partial in BOTH axes, zero-padded.
    t = raw_seen[3]
    h = height - TILE_PX
    w = width - TILE_PX
    assert np.array_equal(t[:h, :w], a[TILE_PX : TILE_PX + h, TILE_PX : TILE_PX + w])
    assert (t[h:, :] == 0).all()
    assert (t[:, w:] == 0).all()


def test_turbo_writers_wrap_source_array_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Audit #78 P1-3: both sibling writers share ONE no-copy MemoryDataset.

    The pass-3 fix (shared MemoryDataset, copy=False, MultiBand tuple
    form) landed only in _turbo_write_stock_parallel; _turbo_write_
    parallel and _reproject_band_array still passed bare ndarrays, so
    reproject copied the whole band into a fresh MEM dataset per warp
    call (3.1 GB each at full-AU). Each writer now wraps once before
    the band loop: one construction per call, and the outputs stay
    bit-identical across thread counts.

    """
    orig = pipeline_mod.MemoryDataset
    counts = {"n": 0}

    class Counting(orig):  # type: ignore[misc]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            counts["n"] += 1
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(pipeline_mod, "MemoryDataset", Counting)
    # Spy the reproject source form: the MultiBand tuple form consumes the
    # shared handle; a bare ndarray makes reproject copy the whole band
    # into a fresh MEM dataset per warp call (the pre-P1-3 behaviour).
    src_forms: list[Any] = []
    orig_reproject = pipeline_mod.reproject

    def spy_reproject(src: Any, *args: Any, **kwargs: Any) -> Any:
        src_forms.append(src)
        return orig_reproject(src, *args, **kwargs)

    monkeypatch.setattr(pipeline_mod, "reproject", spy_reproject)
    a, xform = _demo_field(1600, 1400, 100.0, seed=23)
    dst_transform, dst_width, dst_height = _dst_grid(1600, 1400, 100.0)
    work_crs, dst_crs = f"EPSG:{WORK_CRS}", f"EPSG:{OUT_CRS}"
    tags, scales, offsets = {}, (1.0,), (0.0,)
    out1 = tmp_path / "arr1.tif"
    out4 = tmp_path / "arr4.tif"
    _reproject_band_array(
        a,
        xform,
        work_crs,
        str(out1),
        _demo_profile(),
        dst_crs,
        dst_transform,
        dst_width,
        dst_height,
        tags=tags,
        scales=scales,
        offsets=offsets,
        n_threads=1,
    )
    assert counts["n"] == 1  # one wrap, not one per warp tile
    assert all(isinstance(s, tuple) for s in src_forms)  # tuple form: shared handle, no copy
    src_forms.clear()
    _reproject_band_array(
        a,
        xform,
        work_crs,
        str(out4),
        _demo_profile(),
        dst_crs,
        dst_transform,
        dst_width,
        dst_height,
        tags=tags,
        scales=scales,
        offsets=offsets,
        n_threads=4,
    )
    assert counts["n"] == 2  # one more wrap for the whole second call
    assert all(isinstance(s, tuple) for s in src_forms)
    src_forms.clear()
    assert _sha(out1) == _sha(out4)  # bit-identity still holds

    frames = iter(_stock_frames(out1))

    def stub(tiles: list[np.ndarray], n_threads: int = 1) -> list[bytes]:  # ruff: ignore[unused-function-argument]
        return [next(frames) for _ in tiles]

    monkeypatch.setattr(pipeline_mod.zstdmt, "compress_tiles", stub)
    out_par = tmp_path / "par.tif"
    ok = _turbo_write_parallel(
        a,
        xform,
        work_crs,
        str(out_par),
        _demo_profile(),
        dst_crs,
        dst_transform,
        dst_width,
        dst_height,
        tags=tags,
        scales=scales,
        offsets=offsets,
        n_threads=4,
    )
    assert ok
    assert counts["n"] == 3  # the zstd writer wraps once too
    assert all(isinstance(s, tuple) for s in src_forms)
    assert _sha(out1) == _sha(out_par)


def test_stock_parallel_reference_head_failure_not_masked(tmp_path: Path) -> None:
    """Audit #78 P1-1: a failed reference head raises the root error.

    Negative dst dimensions make np.zeros raise before `zero` binds;
    the old `finally: del zero` replaced the ValueError with an
    UnboundLocalError, masking the root cause.

    """
    a, xform = _demo_field(800, 700, 100.0, seed=14)
    with pytest.raises(ValueError, match="negative dimensions"):
        _turbo_write_stock_parallel(
            a,
            xform,
            f"EPSG:{WORK_CRS}",
            str(tmp_path / "x.tif"),
            _demo_profile(),
            f"EPSG:{OUT_CRS}",
            xform,
            -5,
            -5,
            tags={},
            scales=(1.0,),
            offsets=(0.0,),
            n_threads=1,
            scratch_dir=str(tmp_path),
        )


def test_turbo_zstd_ok_catches_available_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Audit #81 F-3: an OSError from zstdmt.available() is caught.

    The gate must fail closed (stock parallel writer), never crash:
    the available() call sits inside _turbo_zstd_ok's try/except.

    """
    p = Pipeline(
        str(DATA_CSV),
        "price",
        "longitude",
        "latitude",
        str(tmp_path / "o"),
        res=500.0,
        cap_km=10.0,
        workers=1,
        skip_calibration=True,
    )

    def boom() -> bool:
        msg = "libgdal mapping vanished"
        raise OSError(msg)

    monkeypatch.setattr(pipeline_mod.zstdmt, "available", boom)
    assert p._turbo_zstd_ok() is False  # ruff: ignore[private-member-access]


def test_turbo_precheck_passes_memory_ceiling(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Audit #78 P1-2: _turbo_precheck hands min(physical, cgroup) to precheck."""
    monkeypatch.setattr(pipeline_mod.turbop, "resolve_cap", _identity_cap)
    monkeypatch.setattr(pipeline_mod.turbop, "_read_physical_gb", lambda: 134.9)
    monkeypatch.setattr(pipeline_mod.turbop, "_read_cgroup_max_gb", lambda: 6.0)
    seen: dict[str, object] = {}
    orig = pipeline_mod.turbop.precheck

    def spy(plan: turbop.TurboPlan, **kw: object) -> Any:
        seen.update(kw)
        return orig(plan, **kw)

    monkeypatch.setattr(pipeline_mod.turbop, "precheck", spy)
    p = _a9_pipeline(tmp_path, 8.0)  # floor cap on full-AU: per_box, budget 6.8 GB
    p._turbo_precheck()  # ruff: ignore[private-member-access]
    assert seen["available_gb"] == 6.0  # the cgroup slice binds
    assert p._turbo_decision is not None  # ruff: ignore[private-member-access]
    assert p._turbo_decision.message is not None  # ruff: ignore[private-member-access]
    assert "exceeds the process memory ceiling 6.0 GB" in p._turbo_decision.message  # ruff: ignore[private-member-access]


def test_turbo_write_parallel_4gib_guard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Classic head + assembled size beyond 4 GiB -> False (serial fallback)."""
    a, xform = _demo_field(800, 700, 100.0, seed=12)
    dst_transform, dst_width, dst_height = _dst_grid(800, 700, 100.0)
    work_crs, dst_crs = f"EPSG:{WORK_CRS}", f"EPSG:{OUT_CRS}"
    monkeypatch.setattr(pipeline_mod, "_TIFF_CLASSIC_MAX_BYTES", 1024)  # any classic file now "exceeds"
    ok = _turbo_write_parallel(
        a,
        xform,
        work_crs,
        str(tmp_path / "big.tif"),
        _demo_profile(),
        dst_crs,
        dst_transform,
        dst_width,
        dst_height,
        tags={},
        scales=(1.0,),
        offsets=(0.0,),
        n_threads=2,
    )
    assert ok is False


def test_tif_parse_ifd_classic_and_bigtiff(tmp_path: Path) -> None:
    """Classic and BigTIFF heads both parse; tile count and offsets resolve."""
    rng = np.random.default_rng(13)
    width, height = 1200, 900
    a = (np.arange(width * height, dtype=np.int64) % 9000).reshape(height, width).astype(np.int16)
    a[rng.random((height, width)) < 0.1] = NODATA
    xform = from_origin(0.0, height * 100.0, 100.0, 100.0)
    n_tiles = math.ceil(width / TILE_PX) * math.ceil(height / TILE_PX)  # 3*2
    for big, fname in ((False, "c.tif"), (True, "b.tif")):
        path = tmp_path / fname
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=width,
            height=height,
            count=1,
            dtype="int16",
            crs=f"EPSG:{WORK_CRS}",
            transform=xform,
            nodata=NODATA,
            compress="ZSTD",
            tiled=True,
            blockxsize=TILE_PX,
            blockysize=TILE_PX,
            predictor=2,
            BIGTIFF="YES" if big else "IF_SAFER",
        ) as dst:
            dst.write(a, 1)
        info = _tif_parse_ifd(path.read_bytes())
        assert info["big"] is big
        assert info["n_tiles"] == n_tiles
        assert info["t324"][1] == n_tiles
        assert info["t325"][1] == n_tiles
        # Y-1: per-tag slot widths. BigTIFF 324 = LONG8 (8 B), 325 = LONG (4 B);
        # classic stores both as LONG.
        if big:
            assert info["t324"][0] == 16
            assert info["sz324"] == 8
            assert info["t325"][0] == 4
            assert info["sz325"] == 4
        else:
            assert info["t324"][0] == 4
            assert info["sz324"] == 4
            assert info["t325"][0] == 4
            assert info["sz325"] == 4
        d = path.read_bytes()
        offs = [struct.unpack_from(info["fmt324"], d, info["t324"][2] + i * info["sz324"])[0] for i in range(n_tiles)]
        sizes = [struct.unpack_from(info["fmt325"], d, info["t325"][2] + i * info["sz325"])[0] for i in range(n_tiles)]
        # The parsed first offset is the first stored tile, offsets run
        # monotonically (raster-scan order), and each frame is a zstd stream.
        assert offs[0] == info["first"]
        assert all(o2 > o1 for o1, o2 in pairwise(offs))
        for o, s in zip(offs, sizes, strict=True):
            assert d[o : o + 4] == b"\x28\xb5\x2f\xfd"  # zstd magic at each tile
            assert o + s <= len(d)
        # The 324/325 arrays are disjoint and sit before the first tile
        # (the head-cut assumption of the parallel writer).
        a0, a1 = info["t324"][2], info["t324"][2] + info["sz324"] * n_tiles
        b0, b1 = info["t325"][2], info["t325"][2] + info["sz325"] * n_tiles
        assert a1 <= b0 or b1 <= a0
        assert min(a0, b0) < info["first"]
        assert info["n_tiles"] == n_tiles
        assert info["t324"][1] == n_tiles
        assert info["t325"][1] == n_tiles
        assert info["first"] > 0


def test_turbo_e2e_inram_matches_nonturbo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Full turbo run is byte-identical to the non-turbo run (A6).

    In-RAM DN field, oracle mismatch -> serial array reproject, MT
    descent/warp driven by the turbo plan.
    """
    monkeypatch.setattr(pipeline_mod.turbop, "resolve_cap", _identity_cap)
    p0 = Pipeline(str(DATA_CSV), "price", "longitude", "latitude", str(tmp_path / "ref"), res=50.0, workers=4)
    v0, s0 = p0.run()
    p1 = Pipeline(
        str(DATA_CSV),
        "price",
        "longitude",
        "latitude",
        str(tmp_path / "turbo"),
        res=50.0,
        workers=4,
        turbo=True,
        turbo_cap_gb=64.0,
    )
    v1, s1 = p1.run()
    assert _sha(v0) == _sha(v1)
    assert _sha(s0) == _sha(s1)
    assert p1._turbo_decision is not None  # ruff: ignore[private-member-access]
    assert p1._turbo_decision.path == "shared"  # ruff: ignore[private-member-access]
    assert p1._turbo_plan is not None  # ruff: ignore[private-member-access]
    assert p1._turbo_plan.workers >= 2  # ruff: ignore[private-member-access] - MT paths actually taken


def test_turbo_budget_gate_falls_back_to_serial(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """DN field above half the budget -> file-based serial write; bytes still match."""
    monkeypatch.setattr(pipeline_mod.turbop, "resolve_cap", _identity_cap)
    p0 = Pipeline(str(DATA_CSV), "price", "longitude", "latitude", str(tmp_path / "ref"), res=50.0, workers=4)
    v0, s0 = p0.run()
    p1 = Pipeline(
        str(DATA_CSV),
        "price",
        "longitude",
        "latitude",
        str(tmp_path / "turbo"),
        res=50.0,
        workers=4,
        turbo=True,
        turbo_cap_gb=64.0,
    )
    p1.ingest()
    p1.calibrate()
    p1.grid()
    p1._turbo_precheck()  # ruff: ignore[private-member-access]
    p1._turbo_budget_bytes = 1000  # ruff: ignore[private-member-access] - force the DN field gate
    v1, s1 = p1._write_rasters()  # ruff: ignore[private-member-access]
    assert _sha(v0) == _sha(v1)
    assert _sha(s0) == _sha(s1)


def test_turbo_out_crs_equals_work_crs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """out_crs == work_crs: the block-write file is final; turbo adds nothing."""
    monkeypatch.setattr(pipeline_mod.turbop, "resolve_cap", _identity_cap)
    p0 = Pipeline(
        str(DATA_CSV),
        "price",
        "longitude",
        "latitude",
        str(tmp_path / "ref"),
        res=50.0,
        workers=4,
        out_crs=WORK_CRS,
    )
    v0, s0 = p0.run()
    p1 = Pipeline(
        str(DATA_CSV),
        "price",
        "longitude",
        "latitude",
        str(tmp_path / "turbo"),
        res=50.0,
        workers=4,
        out_crs=WORK_CRS,
        turbo=True,
        turbo_cap_gb=64.0,
    )
    v1, s1 = p1.run()
    assert _sha(v0) == _sha(v1)
    assert _sha(s0) == _sha(s1)


def test_box_count_banded_mt_matches_serial() -> None:
    """box_count_banded n_threads=4 == n_threads=1 (the _prepare_shared wiring)."""
    rng = np.random.default_rng(3)
    nx = ny = 800
    x = rng.integers(0, nx, 20_000)
    y = rng.integers(0, ny, 20_000)
    tv = rng.uniform(0.0, 10.0, 20_000)
    _s0, c0 = bin_points(x, y, tv, nx, ny)
    cap_cells = 5
    near1 = np.zeros((nx, ny), dtype=bool)
    near4 = np.zeros((nx, ny), dtype=bool)
    box_count_banded(c0, cap_cells, near1, n_threads=1)
    box_count_banded(c0, cap_cells, near4, n_threads=4)
    assert np.array_equal(near1, near4)
    assert near1.any()
    assert not near1.all()


def test_cli_turbo_flags() -> None:
    """--turbo [preset], --max-ram, --ram-gb, --turbo-strict parse and validate."""
    base = ["x.csv", "--value-col", "v", "--lng-col", "l", "--lat-col", "la"]
    parser = _build_parser()
    a = parser.parse_args([*base, "--turbo", "16"])
    assert a.turbo == "16"
    a = parser.parse_args([*base, "--turbo"])
    assert a.turbo == "auto"
    a = parser.parse_args([*base, "--max-ram", "20"])
    assert a.max_ram == 20.0
    for extra in (
        ["--turbo", "24"],
        ["--max-ram", "16", "--ram-gb", "32"],
        ["--turbo-strict"],
        ["--max-ram", "0"],
    ):
        with pytest.raises(SystemExit):
            pipeline_mod.main([*base, *extra])


def test_cli_turbo_wiring(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--turbo 16 + --turbo-strict on an infeasible grid exits 3 (A9 CLI path)."""
    monkeypatch.setattr(pipeline_mod.turbop, "resolve_cap", _identity_cap)
    # The melb grid at 100 m is far too small to be infeasible under 16 GB;
    # swap in the A9 anchor geometry at the pre-check point.
    orig = Pipeline._turbo_precheck  # ruff: ignore[private-member-access]

    def a9_hack(self: Pipeline) -> None:
        """Inject the A9 anchor grid, then run the real pre-check."""
        self.n, self.nx_padded, self.ny_padded, self.res, self.cap_km_val = (
            400_000,
            _A9_WC,
            _A9_CELLS // _A9_WC,
            100.0,
            25,
        )
        orig(self)

    monkeypatch.setattr(Pipeline, "_turbo_precheck", a9_hack)
    with pytest.raises(SystemExit) as ei:
        pipeline_mod.main(
            [
                str(DATA_CSV),
                "--value-col",
                "price",
                "--lng-col",
                "longitude",
                "--lat-col",
                "latitude",
                "--res",
                "100",
                "--cap-km",
                "25",
                "--turbo",
                "16",
                "--turbo-strict",
                "-o",
                str(tmp_path / "o"),
            ]
        )
    assert ei.value.code == 3


# ---------------------------------------------------------------------------
# Issue #40 (M1): partial/stale output handling
# ---------------------------------------------------------------------------


def _small_csv(tmp_path: Path, n: int = 400, seed: int = 7) -> Path:
    """Write a small deterministic CSV around Melbourne (fast e2e runs)."""
    rng = np.random.default_rng(seed)
    lon = 144.9 + rng.random(n) * 0.1
    lat = -37.8 - rng.random(n) * 0.1
    v = rng.integers(100_000, 2_000_000, n).astype(float)
    p = tmp_path / "small.csv"
    pd.DataFrame({"value": v, "longitude": lon, "latitude": lat}).to_csv(p, index=False)
    return p


def _staging_leftovers(out: Path) -> list[str]:
    """List non-final .tif names in an output dir (staging that survived)."""
    return sorted(p.name for p in out.iterdir() if p.suffix == ".tif" and p.name not in ("value.tif", "support_km.tif"))


def _cli_base(out: Path, *extra: str) -> list[str]:
    """CLI args for a small-CSV run (value col 'value')."""
    args = [
        "--value-col",
        "value",
        "--lng-col",
        "longitude",
        "--lat-col",
        "latitude",
        "-o",
        str(out),
        "--skip-calibration",
    ]
    args.extend(extra)
    return args


def test_stale_outputs_warning_after_failed_run(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Reviewer repro (#40): earlier run's value.tif + failed new run -> warn.

    A run that fails before the write phase (bad --value-col) must exit
    non-zero AND warn that the rasters in the out dir are from an earlier
    run; the earlier files stay byte-identical.
    """
    csv = _small_csv(tmp_path)
    out = tmp_path / "out"
    pipeline_mod.main([str(csv), *_cli_base(out)])
    capsys.readouterr()
    assert _staging_leftovers(out) == []  # success leaves no .tmp / _tmp staging
    sha_v = _sha(out / "value.tif")
    sha_s = _sha(out / "support_km.tif")

    with pytest.raises(SystemExit) as ei:
        pipeline_mod.main([str(csv), *_cli_base(out, "--value-col", "nonexistent", "--force")])
    assert ei.value.code == 2
    err = capsys.readouterr().err
    assert "failed before writing outputs" in err
    assert "value.tif" in err
    assert "support_km.tif" in err
    assert _sha(out / "value.tif") == sha_v
    assert _sha(out / "support_km.tif") == sha_s


def _boom(*_args: object, **_kwargs: object) -> None:
    """Raise a mid-write crash (signature matches the writer callables).

    Raises:
        RuntimeError: Always, to simulate the mid-write failure.

    """
    msg = "simulated mid-write crash"
    raise RuntimeError(msg)


def test_midwrite_failure_keeps_finals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Gate (d): a crash during the serial write leaves finals untouched.

    The finals are only ever published via atomic rename after the band
    completes; a mid-reproject crash must leave the earlier run's
    value.tif / support_km.tif byte-identical, warn, and clean up all
    staging files.
    """
    csv = _small_csv(tmp_path)
    out = tmp_path / "out"
    pipeline_mod.main([str(csv), *_cli_base(out)])
    capsys.readouterr()
    sha_v = _sha(out / "value.tif")
    sha_s = _sha(out / "support_km.tif")

    monkeypatch.setattr(pipeline_mod, "_reproject_band", _boom)
    with pytest.raises(RuntimeError, match="simulated mid-write crash"):
        pipeline_mod.main([str(csv), *_cli_base(out, "--force")])
    err = capsys.readouterr().err
    assert "failed during write" in err
    assert "failed before writing outputs" in err
    assert _sha(out / "value.tif") == sha_v
    assert _sha(out / "support_km.tif") == sha_s
    assert _staging_leftovers(out) == []


def test_turbo_midwrite_failure_keeps_finals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """M1(b): a crash in the turbo final-write phase leaves finals untouched.

    The turbo dnfill + write loop is covered by the same
    except BaseException as the serial path: the rename publish is the last
    step, so a mid-write crash keeps the earlier run's finals
    byte-identical, warns, and cleans up staging + scratch.
    """
    csv = _small_csv(tmp_path)
    out = tmp_path / "out"
    pipeline_mod.main([str(csv), *_cli_base(out, "--turbo", "32")])
    capsys.readouterr()
    assert _staging_leftovers(out) == []
    sha_v = _sha(out / "value.tif")
    sha_s = _sha(out / "support_km.tif")

    # Whichever writer the stack picks (zstd parallel, stock parallel, or
    # the serial array reproject), make it fail mid-write.
    monkeypatch.setattr(pipeline_mod, "_turbo_write_parallel", _boom)
    monkeypatch.setattr(pipeline_mod, "_turbo_write_stock_parallel", _boom)
    monkeypatch.setattr(pipeline_mod, "_reproject_band_array", _boom)
    with pytest.raises(RuntimeError, match="simulated mid-write crash"):
        pipeline_mod.main([str(csv), *_cli_base(out, "--turbo", "32", "--force")])
    err = capsys.readouterr().err
    assert "failed during write" in err
    assert "failed before writing outputs" in err
    assert _sha(out / "value.tif") == sha_v
    assert _sha(out / "support_km.tif") == sha_s
    assert _staging_leftovers(out) == []
    assert not list(out.glob("_stock_scratch_*.tif"))
