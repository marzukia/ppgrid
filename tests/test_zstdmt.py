"""Tests for ppgrid.zstdmt (issue #29: turbo zstdmt module).

Covers CPL pfn resolution and the availability self-check, compress_tile
frame output and roundtrip through the CPL decompressor, predictor2
against a direct re-implementation of libtiff horDiff16, compress_tiles
determinism across thread counts (n_threads=1 must be byte-identical to
serial), and the calibration oracle: byte-equal verdict, forced-mismatch
fallback signal via an injectable compressor, and consistency with the
real CPL pfn on the committed stock-path fixture.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio

from ppgrid import zstdmt
from ppgrid.zstdmt import (
    ZstdmtError,
    compress_tile,
    compress_tiles,
    decompress_tile,
    oracle_check,
    predictor2,
)

FIXTURE = Path(__file__).parent / "fixtures" / "zstd_oracle_512.tif"
TILE_PX = 512
_ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"


def _tile(seed: int) -> np.ndarray:
    """Return one 512x512 int16 random tile from a seeded generator."""
    return np.random.default_rng(seed).integers(-32768, 32767, size=(TILE_PX, TILE_PX), dtype=np.int16)


def _hordiff16_ref(a: np.ndarray) -> np.ndarray:
    """Re-implement libtiff horDiff16 directly (row-wise, uint16 wrap)."""
    out = a.copy()
    for r in range(a.shape[0]):
        prev = int(int(a[r, 0]) & 0xFFFF)
        for c in range(1, a.shape[1]):
            cur = int(int(a[r, c]) & 0xFFFF)
            d = (cur - prev) % 65536
            out[r, c] = d - 65536 if d > 32767 else d
            prev = cur
    return out


def _assert_frames_equal(left: list[bytes], right: list[bytes]) -> None:
    """Compare frame lists without difflib (fast failure on 512 KB frames).

    Raises:
        AssertionError: If the lists differ in length or in any frame.

    """
    assert len(left) == len(right)
    for i, (lf, rf) in enumerate(zip(left, right, strict=True)):
        if lf != rf:
            n = min(len(lf), len(rf))
            j = next((k for k in range(n) if lf[k] != rf[k]), n)
            msg = f"tile {i}: {len(lf)} B vs {len(rf)} B, first diff byte {j}"
            raise AssertionError(msg)


# ---------------------------------------------------------------- compressor


def test_available_true() -> None:
    """available() is True on a GDAL build with the zstd CPL compressor."""
    if not zstdmt.available():
        pytest.skip("CPL zstd compressor not resolvable in this GDAL build")
    assert zstdmt.available() is True


def test_compress_tile_produces_zstd_frame_and_roundtrips() -> None:
    """compress_tile emits a ZSTD frame that the CPL decompressor inverts."""
    raw = predictor2(_tile(1)).tobytes()
    frame = compress_tile(raw)
    assert frame[:4] == _ZSTD_MAGIC
    assert len(frame) > len(raw)  # incompressible data: frame overhead only
    assert decompress_tile(frame) == raw


def test_compress_tile_deterministic() -> None:
    """Repeated compress_tile calls on the same bytes give identical frames."""
    raw = predictor2(_tile(2)).tobytes()
    assert compress_tile(raw) == compress_tile(raw)


def test_compress_tiles_matches_serial_and_deterministic_across_thread_counts() -> None:
    """compress_tiles output is byte-identical for n_threads 1..16."""
    tiles = [predictor2(_tile(s)).tobytes() for s in range(32)]
    serial = [compress_tile(t) for t in tiles]
    for n in (1, 2, 4, 8, 16):  # 16 > tile count: more workers than tiles
        _assert_frames_equal(compress_tiles(tiles, n_threads=n), serial)


def test_compress_tiles_rejects_bad_n_threads() -> None:
    """compress_tiles raises ValueError for n_threads < 1."""
    with pytest.raises(ValueError, match="n_threads"):
        compress_tiles([b"abc"], n_threads=0)
    with pytest.raises(ValueError, match="n_threads"):
        compress_tiles([b"abc"], n_threads=-1)


def test_compress_tiles_rejects_bad_tile_type() -> None:
    """compress_tiles raises TypeError for non-bytes/non-array tiles."""
    with pytest.raises(TypeError, match="bytes-like or a numpy array"):
        compress_tiles([12345])


def test_compress_tiles_accepts_ndarray_inputs() -> None:
    """compress_tiles accepts int16 numpy arrays as tile inputs."""
    pa, pb = predictor2(_tile(3)), predictor2(_tile(4))
    _assert_frames_equal(
        compress_tiles([pa, pb], n_threads=2),
        [compress_tile(pa.tobytes()), compress_tile(pb.tobytes())],
    )


# ---------------------------------------------------------------- predictor2


def test_predictor2_matches_libtiff_hordiff16() -> None:
    """predictor2 matches a direct horDiff16 re-implementation (wrap cases)."""
    a = np.array([[30000, -30000, 5, -5], [0, 32767, -32768, 7]], dtype=np.int16)
    expected = np.array([[30000, 5536, 30005, -10], [0, 32767, 1, -32761]], dtype=np.int16)
    assert predictor2(a).tolist() == expected.tolist()
    rng = np.random.default_rng(7)
    m = rng.integers(-32768, 32767, size=(8, 5), dtype=np.int16)
    assert (predictor2(m) == _hordiff16_ref(m)).all()


def test_predictor2_rejects_bad_shape_and_dtype() -> None:
    """predictor2 raises ValueError for non-2-D or non-int16 input."""
    with pytest.raises(ValueError, match="2-D"):
        predictor2(np.zeros(4, dtype=np.int16))
    with pytest.raises(ValueError, match="int16"):
        predictor2(np.zeros((2, 2), dtype=np.int32))


# ---------------------------------------------------------------- oracle


def test_oracle_byte_equal_on_fixture() -> None:
    """A byte-reproducing compressor verdicts ok and sees predictor2 input.

    Raises:
        AssertionError: If the turbo input is not predictor2 of the stored tile.

    """
    info = zstdmt._tif_info(FIXTURE)  # ruff: ignore[private-member-access]
    stock = info.tile_bytes[0]
    seen: dict[str, bytes] = {}

    def spy_compress(data: bytes) -> bytes:
        seen["data"] = data
        return stock

    verdict = oracle_check(FIXTURE, tile=0, compress=spy_compress)
    assert verdict.ok is True
    assert verdict.turbo_size == len(stock)
    assert verdict.stock_size == len(stock)
    with rasterio.open(FIXTURE) as ds:
        arr = ds.read(1)
    expected = predictor2(arr).tobytes()
    if seen["data"] != expected:
        n = min(len(seen["data"]), len(expected))
        j = next((k for k in range(n) if seen["data"][k] != expected[k]), n)
        msg = f"turbo input differs from predictor2(stored tile): first byte {j}"
        raise AssertionError(msg)
    assert stock[:4] == _ZSTD_MAGIC


def test_oracle_forced_mismatch_returns_fallback_signal() -> None:
    """A drifting compressor verdicts mismatch and names the serial fallback."""
    info = zstdmt._tif_info(FIXTURE)  # ruff: ignore[private-member-access]
    stock = info.tile_bytes[0]

    def broken_compress(_data: bytes) -> bytes:
        return stock + b"\x00"

    verdict = oracle_check(FIXTURE, tile=0, compress=broken_compress)
    assert verdict.ok is False
    assert verdict.turbo_size == len(stock) + 1
    assert verdict.stock_size == len(stock)
    assert "mismatch" in verdict.detail
    assert "stock parallel" in verdict.detail  # fallback is the stock parallel writer, not serial (issue #40 m10)


def test_oracle_real_cpl_pfn_consistent_with_byte_comparison() -> None:
    """The oracle verdict equals an independent turbo-vs-stock byte compare.

    On the current stack (GDAL 3.12.4 / libtiff 6.2 / libzstd 1.5.7) the
    CPL pfn frames differ from the libtiff streaming frames, so the oracle
    reports mismatch: the designed serial-fallback signal.
    """
    info = zstdmt._tif_info(FIXTURE)  # ruff: ignore[private-member-access]
    stock = info.tile_bytes[0]
    with rasterio.open(FIXTURE) as ds:
        arr = ds.read(1)
    turbo = compress_tile(predictor2(arr).tobytes())
    verdict = oracle_check(FIXTURE, tile=0)
    assert verdict.ok is (turbo == stock)
    assert verdict.turbo_size == len(turbo)
    assert verdict.stock_size == len(stock)
    if turbo != stock:
        assert "mismatch" in verdict.detail
        assert "stock parallel" in verdict.detail  # fallback is the stock parallel writer, not serial (issue #40 m10)


def test_oracle_tile_out_of_range() -> None:
    """oracle_check raises ZstdmtError for a tile index past the grid."""
    with pytest.raises(ZstdmtError, match="out of range"):
        oracle_check(FIXTURE, tile=1, compress=lambda data: data)


def _write_tmp_raster(path: Path, *, tiled: bool, predictor: int, dtype: str = "int16") -> None:
    """Write a small 64x64 ZSTD GeoTIFF (tiled or strip) for oracle edge cases."""
    rng = np.random.default_rng(9)
    arr = rng.integers(0, 1000, size=(64, 64), dtype=dtype)
    xform = rasterio.transform.from_origin(0.0, 640.0, 10.0, 10.0)
    kwargs: dict[str, object] = {
        "width": 64,
        "height": 64,
        "count": 1,
        "dtype": dtype,
        "crs": "EPSG:3857",
        "transform": xform,
        "compress": "ZSTD",
        "nodata": -1,
    }
    if tiled:
        kwargs.update(tiled=True, blockxsize=32, blockysize=32, predictor=predictor)
    with rasterio.open(path, "w", driver="GTiff", **kwargs) as dst:
        dst.write(arr, 1)


def test_oracle_rejects_strip_raster(tmp_path: Path) -> None:
    """oracle_check raises ZstdmtError on a strip (non-tiled) raster."""
    p = tmp_path / "strip.tif"
    _write_tmp_raster(p, tiled=False, predictor=2)
    with pytest.raises(ZstdmtError, match="tiled"):
        oracle_check(p, compress=lambda data: data)


def test_oracle_rejects_unsupported_dtype(tmp_path: Path) -> None:
    """oracle_check raises ZstdmtError on non-16-bit data."""
    p = tmp_path / "int32.tif"
    _write_tmp_raster(p, tiled=True, predictor=2, dtype="int32")
    with pytest.raises(ZstdmtError, match="16-bit"):
        oracle_check(p, compress=lambda data: data)


def test_oracle_predictor1_identity_path(tmp_path: Path) -> None:
    """With predictor 1 the turbo side is fed raw tile bytes, no differencing."""
    p = tmp_path / "nopred.tif"
    _write_tmp_raster(p, tiled=True, predictor=1)
    info = zstdmt._tif_info(p)  # ruff: ignore[private-member-access]
    stock = info.tile_bytes[0]
    seen: dict[str, bytes] = {}

    def spy_compress(data: bytes) -> bytes:
        seen["data"] = data
        return stock

    verdict = oracle_check(p, tile=0, compress=spy_compress)
    assert verdict.ok is True
    with rasterio.open(p) as ds:
        arr = ds.read(1)
    # tile 0 is the top-left 32x32 block; raw bytes, no differencing
    assert seen["data"] == arr[0:32, 0:32].tobytes()


# ---------------------------------------------------------------- BigTIFF parser


def _write_bigtiff(path: Path) -> None:
    """Write a 512x512 tiled ZSTD predictor-2 BigTIFF for the parser tests."""
    rng = np.random.default_rng(11)
    arr = rng.integers(0, 1000, size=(TILE_PX, TILE_PX), dtype="int16")
    xform = rasterio.transform.from_origin(0.0, 5120.0, 10.0, 10.0)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=TILE_PX,
        height=TILE_PX,
        count=1,
        dtype="int16",
        crs="EPSG:3857",
        transform=xform,
        tiled=True,
        blockxsize=TILE_PX,
        blockysize=TILE_PX,
        compress="ZSTD",
        predictor=2,
        bigtiff="YES",
        nodata=-1,
    ) as dst:
        dst.write(arr, 1)


def test_tif_info_parses_bigtiff(tmp_path: Path) -> None:
    """_tif_info reads 20-byte BigTIFF IFD entries without struct.error.

    Raises:
        AssertionError: If the written file is not a little-endian BigTIFF.

    """
    p = tmp_path / "big.tif"
    _write_bigtiff(p)
    head = p.read_bytes()[:4]
    if head != b"II\x2b\x00":
        msg = f"file is not a little-endian BigTIFF: {head!r}"
        raise AssertionError(msg)
    info = zstdmt._tif_info(p)  # ruff: ignore[private-member-access]
    assert info.width == TILE_PX
    assert info.height == TILE_PX
    assert info.tilew == TILE_PX
    assert info.tileh == TILE_PX
    assert info.bps == 16
    assert info.spp == 1
    assert info.predictor == 2
    assert len(info.tile_bytes) == 1
    assert info.tile_bytes[0][:4] == _ZSTD_MAGIC


def test_oracle_runs_on_bigtiff(tmp_path: Path) -> None:
    """oracle_check on a BigTIFF runs end-to-end without crashing the parser.

    On the current stack the CPL pfn frames differ from the libtiff
    streaming frames, so the verdict is expected to be mismatch, the same
    frame-level behavior as the classic fixture.
    """
    p = tmp_path / "big.tif"
    _write_bigtiff(p)
    verdict = oracle_check(p, tile=0)
    assert verdict.turbo_size > 0
    assert verdict.stock_size > 0
    assert "oracle" in verdict.detail
    if not verdict.ok:
        assert "mismatch" in verdict.detail
        assert "stock parallel" in verdict.detail  # fallback is the stock parallel writer, not serial (issue #40 m10)


def test_tif_info_corrupt_body_raises_zstdmterror(tmp_path: Path) -> None:
    """A valid TIFF magic with a truncated IFD raises ZstdmtError, not struct.error."""
    p = tmp_path / "corrupt.tif"
    # classic header claiming the first IFD at offset 8, but the file ends there
    p.write_bytes(b"II\x2a\x00\x08\x00\x00\x00")
    with pytest.raises(ZstdmtError, match="corrupt TIFF"):
        zstdmt._tif_info(p)  # ruff: ignore[private-member-access]
