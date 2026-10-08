"""Parallel per-tile ZSTD compression via the GDAL CPL compressor registry.

GDAL's GTiff driver compresses 512px tiles serially, one libtiff ZSTD call
per tile. The CPL registry (CPLGetCompressor) exposes the same ZSTD backend
as a per-call function: fresh ZSTD_CCtx per call, ZSTD_compress2 at the
registry default level. That is thread-safe and can be driven from a Python
thread pool for per-tile parallelism (turbo design S3.4 part 2).

This module does not write the raster. It:

* exposes the CPL "zstd" compressor through ctypes against the libgdal that
  rasterio already loaded (no new dependency, no GDAL version assumption
  beyond what rasterio vendors),
* compresses a batch of raw (post-predictor) tile buffers in a thread
  pool, preserving input order,
* provides the calibration oracle (design doc A.7): compress one tile the
  way the turbo path would and byte-compare it against the tile bytes
  actually stored in a raster written by the stock GDAL path. On mismatch
  the caller writes with the stock path (parallel; the serial per-band
  write is the last resort); the byte delta is the version-drift signal
  for libtiff/libzstd upgrades.

Byte-identity note (measured 2026-10-06, GDAL 3.12.4, libtiff 6.2,
libzstd 1.5.7, rasterio 1.5.1): the stock GTiff path compresses through
the libtiff ZSTD codec (streaming Ctx, default level 3, no frame content
size in the header) while the CPL pfn uses ZSTD_compress2 at the registry
default level 13 (frame carries a content size field and a different
window descriptor). Frames therefore do not compare byte-equal on a
512px tile in this stack, and the oracle reports mismatch: the designed
stock-fallback signal.
"""

from __future__ import annotations

import ctypes
import struct
import threading
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

__all__ = [
    "OracleVerdict",
    "ZstdmtError",
    "available",
    "compress_tile",
    "compress_tiles",
    "decompress_tile",
    "oracle_check",
    "predictor2",
]

# ZSTD magic (RFC 8878). Any other leading bytes mean the pfn is not a zstd
# compressor (ABI drift); fail closed before writing a file.
_ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"
_COMPRESSOR_ID = b"zstd"

# Field offsets in the GDAL CPLCompressor struct (port/cpl_compressor.h;
# layout verified against GDAL 3.12.4). An ABI change is caught by the
# executable-region check and the magic-byte self-check, not by these
# numbers.
_OFF_PFN = 32
_OFF_USER_DATA = 40

# CPLCompressorFunc: bool(const void *input, size_t input_size,
#   void **output, size_t *output_size, CSLConstList options, void *user_data).
# With *output == NULL the C side allocates the buffer (VSI malloc); the
# caller frees it with VSIFree. The zstd decompressor shares this signature.
_CPL_FN = ctypes.CFUNCTYPE(
    ctypes.c_bool,
    ctypes.c_char_p,
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.POINTER(ctypes.c_size_t),
    ctypes.c_void_p,
    ctypes.c_void_p,
)


class ZstdmtError(RuntimeError):
    """The turbo zstd path is unavailable, or has drifted from stock bytes."""


@dataclass(frozen=True)
class OracleVerdict:
    """Result of the A.7 calibration oracle for one tile.

    ok=True means the turbo-compressed bytes are byte-equal to the bytes
    the stock GDAL path stored, so a turbo write of that tile reproduces
    the stock file exactly. ok=False means the caller must write that
    tile (and every tile) with the stock path (parallel; the serial
    per-band write is the last resort).
    """

    ok: bool
    turbo_size: int
    stock_size: int
    detail: str


class _State:
    """Lazy cache for the libgdal handle and CPL zstd compressor."""

    __slots__ = ("gdal_path", "lib", "pfn", "user_data")

    def __init__(self) -> None:
        self.gdal_path = ""
        self.lib: ctypes.CDLL | None = None
        self.pfn: _CPL_FN | None = None
        self.user_data = 0


_STATE = _State()
_LOCK = threading.RLock()
_DECOMP_CACHE: dict[str, object] = {}
_DECOMP_LOCK = threading.RLock()


def _gdal_path() -> str:
    """Return the on-disk path of the libgdal that rasterio already loaded.

    Raises:
        ZstdmtError: If /proc/self/maps is unreadable or maps no libgdal.

    """
    try:
        maps = Path("/proc/self/maps").read_text(encoding="utf-8")
    except OSError as exc:
        msg = "cannot read /proc/self/maps; zstdmt needs a Linux /proc"
        raise ZstdmtError(msg) from exc
    sizes: dict[str, int] = {}
    for line in maps.splitlines():
        parts = line.split(None, 5)
        if len(parts) < 6 or "libgdal" not in parts[5]:
            continue
        lo, hi = parts[0].split("-", 1)
        sizes[parts[5]] = max(sizes.get(parts[5], 0), int(hi, 16) - int(lo, 16))
    if not sizes:
        msg = "libgdal not found in /proc/self/maps; import rasterio first"
        raise ZstdmtError(msg)
    return max(sizes, key=sizes.get)


def _exec_ranges(path: str) -> list[tuple[int, int]]:
    """Return the executable (r-x) address ranges mapped from path."""
    ranges: list[tuple[int, int]] = []
    for line in Path("/proc/self/maps").read_text(encoding="utf-8").splitlines():
        parts = line.split(None, 5)
        if len(parts) < 6 or parts[5] != path or "x" not in parts[1]:
            continue
        lo, hi = parts[0].split("-", 1)
        ranges.append((int(lo, 16), int(hi, 16)))
    return ranges


def _in_exec_ranges(addr: int, ranges: Sequence[tuple[int, int]]) -> bool:
    """Return True if addr falls inside any [lo, hi) executable range."""
    return any(lo <= addr < hi for lo, hi in ranges)


def _lib() -> ctypes.CDLL:
    """Load (via the already-loaded mapping) and prepare the libgdal handle."""
    with _LOCK:
        if _STATE.lib is None:
            _STATE.gdal_path = _gdal_path()
            lib = ctypes.CDLL(_STATE.gdal_path)
            lib.GDALAllRegister()
            lib.CPLGetCompressor.restype = ctypes.c_void_p
            lib.CPLGetCompressor.argtypes = [ctypes.c_char_p]
            lib.CPLGetDecompressor.restype = ctypes.c_void_p
            lib.CPLGetDecompressor.argtypes = [ctypes.c_char_p]
            lib.VSIFree.argtypes = [ctypes.c_void_p]
            _STATE.lib = lib
    return _STATE.lib


def _extract_fn(cpl: int, what: str) -> tuple[_CPL_FN, int]:
    """Read the function pointer out of a CPLCompressor struct, with guards.

    Raises:
        ZstdmtError: If the pfn is NULL or outside the libgdal exec mappings.

    """
    pfn_addr = ctypes.c_void_p.from_address(cpl + _OFF_PFN).value
    user_data = ctypes.c_void_p.from_address(cpl + _OFF_USER_DATA).value
    if not pfn_addr:
        msg = f"CPLCompressor.{what} has no pfnFunc (struct ABI drift?)"
        raise ZstdmtError(msg)
    if not _in_exec_ranges(pfn_addr, _exec_ranges(_STATE.gdal_path)):
        msg = f"{what} pfn at {pfn_addr:#x} is outside libgdal exec mappings (ABI drift?)"
        raise ZstdmtError(msg)
    return _CPL_FN(pfn_addr), user_data or 0


def _call_fn(lib: ctypes.CDLL, pfn: _CPL_FN, user_data: int, data: bytes, what: str) -> bytes:
    """Run one CPL compress/decompress call and free the C-side output.

    Raises:
        ZstdmtError: If the pfn returns FALSE or an empty buffer.

    """
    out = ctypes.c_void_p(None)
    out_size = ctypes.c_size_t(0)
    if not pfn(data, len(data), ctypes.byref(out), ctypes.byref(out_size), None, user_data):
        msg = f"CPL {what} returned FALSE"
        raise ZstdmtError(msg)
    if not out or out_size.value == 0:
        msg = f"CPL {what} returned an empty buffer"
        raise ZstdmtError(msg)
    buf = ctypes.string_at(out, out_size.value)
    lib.VSIFree(out)
    return buf


def _compressor() -> tuple[ctypes.CDLL, _CPL_FN, int]:
    """Resolve the CPL zstd compressor once, with ABI drift guards.

    Returns:
        The (lib, pfn, user_data) triple.

    Raises:
        ZstdmtError: If the compressor is missing or the self-checks fail.

    """
    with _LOCK:
        if _STATE.pfn is None:
            lib = _lib()
            cpl = lib.CPLGetCompressor(_COMPRESSOR_ID)
            if not cpl:
                msg = 'CPLGetCompressor("zstd") returned NULL (libgdal built without ZSTD?)'
                raise ZstdmtError(msg)
            pfn, user_data = _extract_fn(cpl, "zstd")
            probe = _call_fn(lib, pfn, user_data, b"ppgrid-zstdmt-probe", "compressor")
            if not probe.startswith(_ZSTD_MAGIC):
                msg = f"zstd pfn output lacks zstd magic (got {probe[:4].hex()!r})"
                raise ZstdmtError(msg)
            _STATE.lib, _STATE.pfn, _STATE.user_data = lib, pfn, user_data
        return _STATE.lib, _STATE.pfn, _STATE.user_data


def _decompressor() -> tuple[ctypes.CDLL, _CPL_FN, int]:
    """Resolve the CPL zstd decompressor once (same guards, no magic probe).

    Returns:
        The (lib, pfn, user_data) triple.

    Raises:
        ZstdmtError: If the decompressor is missing or the guards fail.

    """
    with _DECOMP_LOCK:
        if "pfn" in _DECOMP_CACHE:
            return _DECOMP_CACHE["lib"], _DECOMP_CACHE["pfn"], _DECOMP_CACHE["user_data"]
        lib = _lib()
        cpl = lib.CPLGetDecompressor(_COMPRESSOR_ID)
        if not cpl:
            msg = 'CPLGetDecompressor("zstd") returned NULL (libgdal built without ZSTD?)'
            raise ZstdmtError(msg)
        pfn, user_data = _extract_fn(cpl, "zstd-decompressor")
        _DECOMP_CACHE["lib"], _DECOMP_CACHE["pfn"], _DECOMP_CACHE["user_data"] = lib, pfn, user_data
        return lib, pfn, user_data


def available() -> bool:
    """Return True if the CPL zstd compressor resolves and passes the self-checks."""
    try:
        _compressor()
    except ZstdmtError:
        return False
    return True


def compress_tile(data: bytes) -> bytes:
    """Compress one raw (post-predictor) tile buffer with the CPL zstd pfn.

    Raises ZstdmtError if the compressor is unavailable or the call fails.

    Args:
        data: Raw tile bytes (e.g. an int16 tile with predictor 2 applied).

    Returns:
        One self-contained ZSTD frame. Byte-identical across calls and
        across threads for the same input.

    """
    lib, pfn, user_data = _compressor()
    return _call_fn(lib, pfn, user_data, data, "compressor")


def decompress_tile(data: bytes) -> bytes:
    """Decompress one ZSTD frame through the CPL zstd decompressor pfn.

    The CPL decompressor sizes the output from the frame content size
    field, so the frame must carry one (all compress_tile output does).
    Raises ZstdmtError if the decompressor is unavailable or the call fails.

    Args:
        data: One self-contained ZSTD frame.

    Returns:
        The decompressed bytes.

    """
    lib, pfn, user_data = _decompressor()
    return _call_fn(lib, pfn, user_data, data, "decompressor")


def _to_bytes(tile: bytes | bytearray | memoryview | np.ndarray) -> bytes:
    """Coerce one tile input (bytes-like or numpy array) to raw bytes.

    Raises:
        TypeError: If the input is neither bytes-like nor a numpy array.

    """
    if isinstance(tile, (bytes, bytearray, memoryview)):
        return bytes(tile)
    if isinstance(tile, np.ndarray):
        return np.ascontiguousarray(tile).tobytes()
    msg = f"tile must be bytes-like or a numpy array, got {type(tile)!r}"
    raise TypeError(msg)


def compress_tiles(
    tiles: Sequence[bytes | bytearray | memoryview | np.ndarray],
    n_threads: int = 1,
) -> list[bytes]:
    """Compress a batch of raw tile buffers, preserving input order.

    Args:
        tiles: Raw tile buffers (bytes-like or int16 numpy arrays).
        n_threads: Pool size. 1 runs a plain serial loop; the result is
            byte-identical to calling compress_tile per tile.

    Returns:
        One ZSTD frame per input tile, in input order.

    Raises:
        ValueError: If n_threads < 1.

    """
    if n_threads < 1:
        msg = f"n_threads must be >= 1, got {n_threads}"
        raise ValueError(msg)
    raw = [_to_bytes(t) for t in tiles]
    if n_threads == 1 or len(raw) <= 1:
        return [compress_tile(d) for d in raw]
    with ThreadPoolExecutor(max_workers=min(n_threads, len(raw))) as pool:
        return list(pool.map(compress_tile, raw))


def predictor2(tile: np.ndarray) -> np.ndarray:
    """Apply the TIFF horizontal predictor (2) to one int16 tile.

    Mirrors libtiff horDiff16: each row is differenced left to right, the
    first sample of each row is unchanged, arithmetic wraps mod 2^16.

    Args:
        tile: 2-D int16 array, one tile (height, width).

    Returns:
        New int16 array with the horizontal difference applied.

    Raises:
        ValueError: If the tile is not a 2-D int16 array.

    """
    if tile.ndim != 2:
        msg = f"expected a 2-D tile, got {tile.ndim}-D"
        raise ValueError(msg)
    if tile.dtype != np.int16:
        msg = f"expected int16, got {tile.dtype}"
        raise ValueError(msg)
    out = np.empty_like(tile)
    out[:, 0] = tile[:, 0]
    out[:, 1:] = tile[:, 1:] - tile[:, :-1]
    return out


@dataclass(frozen=True)
class _TifInfo:
    """What the oracle needs from the first image IFD of a tiled GeoTIFF."""

    width: int
    height: int
    bps: int
    spp: int
    predictor: int
    tilew: int
    tileh: int
    tile_bytes: tuple[bytes, ...]


def _decode_ints(raw: bytes, e: str, typ: int, count: int) -> tuple[int, ...]:
    """Decode a TIFF value array (SHORT, LONG, or LONG8) to ints."""
    fmt = {3: "H", 4: "I", 16: "Q"}[typ]
    width = {"H": 2, "I": 4, "Q": 8}[fmt]
    return tuple(struct.unpack_from(e + fmt, raw, i * width)[0] for i in range(count))


def _tif_info(path: Path) -> _TifInfo:
    """Parse the first image IFD of a classic or BigTIFF file.

    Reads only what the oracle needs: image size, tile geometry,
    predictor, and the stored bytes of every tile. Follows the IFD chain
    to the first IFD that carries TileOffsets.

    Raises:
        ZstdmtError: On a non-tiled or unsupported layout.

    """
    data = path.read_bytes()
    if len(data) < 8:
        msg = f"file too small to be a TIFF ({len(data)} B)"
        raise ZstdmtError(msg)
    if data[:2] == b"II":
        e = "<"
    elif data[:2] == b"MM":
        e = ">"
    else:
        msg = f"bad TIFF byte order mark {data[:2]!r}"
        raise ZstdmtError(msg)
    try:
        magic = struct.unpack_from(e + "H", data, 2)[0]
        if magic == 42:
            offset_size = 4
            ifd_off = struct.unpack_from(e + "I", data, 4)[0]
        elif magic == 43:
            offset_size = struct.unpack_from(e + "H", data, 4)[0]
            if offset_size != 8:
                msg = f"unsupported BigTIFF offset size {offset_size}"
                raise ZstdmtError(msg)
            ifd_off = struct.unpack_from(e + "Q", data, 8)[0]
        else:
            msg = f"unknown TIFF magic {magic}"
            raise ZstdmtError(msg)
        off_fmt = "I" if offset_size == 4 else "Q"
        entry_size = 12 if offset_size == 4 else 20
        cnt_size = 2 if offset_size == 4 else 8  # IFD entry-count field width
        while True:
            tags = _parse_ifd_tags(data, e, off_fmt, cnt_size, entry_size, ifd_off)
            if 324 in tags and 325 in tags:
                return _tif_info_from_tags(data, e, tags)
            next_off = _next_ifd(data, e, off_fmt, cnt_size, entry_size, ifd_off)
            if not next_off:
                msg = "no tiled IFD found (strip TIFF?)"
                raise ZstdmtError(msg)
            ifd_off = next_off
    except struct.error as exc:
        msg = f"corrupt TIFF structure: {exc}"
        raise ZstdmtError(msg) from exc


def _parse_ifd_tags(
    data: bytes,
    e: str,
    off_fmt: str,
    cnt_size: int,
    entry_size: int,
    ifd_off: int,
) -> dict[int, tuple[int, int, bytes]]:
    """Return {tag: (type, count, raw bytes)} for one IFD."""
    cnt_fmt = "H" if cnt_size == 2 else "Q"
    entries = struct.unpack_from(e + cnt_fmt, data, ifd_off)[0]
    tags: dict[int, tuple[int, int, bytes]] = {}
    for i in range(entries):
        off = ifd_off + cnt_size + i * entry_size
        tag = struct.unpack_from(e + "H", data, off)[0]
        typ = struct.unpack_from(e + "H", data, off + 2)[0]
        count = struct.unpack_from(e + off_fmt, data, off + 4)[0]
        # Classic entries are 12 B (tag2 type2 count4 value4); BigTIFF entries
        # are 20 B (tag2 type2 count8 value8), so the value field sits at +12.
        val_off = off + (12 if cnt_size == 8 else 8)
        per = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 11: 4, 12: 8, 16: 8}.get(typ)
        if per is None:
            continue
        total = per * count
        val_size = 4 if cnt_size == 2 else 8
        if total <= val_size:
            raw = data[val_off : val_off + total]
        else:
            ptr = struct.unpack_from(e + off_fmt, data, val_off)[0]
            raw = data[ptr : ptr + total]
        tags[tag] = (typ, count, raw)
    return tags


def _next_ifd(
    data: bytes,
    e: str,
    off_fmt: str,
    cnt_size: int,
    entry_size: int,
    ifd_off: int,
) -> int:
    """Return the next-IFD offset after the IFD at ifd_off (0 if none)."""
    cnt_fmt = "H" if cnt_size == 2 else "Q"
    entries = struct.unpack_from(e + cnt_fmt, data, ifd_off)[0]
    return struct.unpack_from(e + off_fmt, data, ifd_off + cnt_size + entries * entry_size)[0]


def _tif_info_from_tags(data: bytes, e: str, tags: dict[int, tuple[int, int, bytes]]) -> _TifInfo:
    def ints(tag: int) -> tuple[int, ...]:
        typ, count, raw = tags[tag]
        return _decode_ints(raw, e, typ, count)

    if 256 not in tags or 257 not in tags or 322 not in tags or 323 not in tags:
        msg = "tiled TIFF must carry width, height, TileWidth, TileLength"
        raise ZstdmtError(msg)
    width = ints(256)[0]
    height = ints(257)[0]
    bps = ints(258)[0] if 258 in tags else 0
    spp = ints(277)[0] if 277 in tags else 1
    predictor = ints(317)[0] if 317 in tags else 1
    offs = ints(324)
    counts = ints(325)
    if len(offs) != len(counts):
        msg = f"TileOffsets/TileByteCounts length mismatch ({len(offs)} vs {len(counts)})"
        raise ZstdmtError(msg)
    tile_bytes = tuple(data[o : o + s] for o, s in zip(offs, counts, strict=True))
    return _TifInfo(
        width=width,
        height=height,
        bps=bps,
        spp=spp,
        predictor=predictor,
        tilew=ints(322)[0],
        tileh=ints(323)[0],
        tile_bytes=tile_bytes,
    )


def oracle_check(
    path: str | Path,
    tile: int = 0,
    compress: Callable[[bytes], bytes] | None = None,
) -> OracleVerdict:
    """Byte-compare one turbo-compressed tile against the stock tile stored in a raster.

    Reads one tile's raw array from the raster (stock decode path),
    applies the TIFF predictor exactly as the stock encoder does,
    compresses the result with `compress` (default: the CPL zstd pfn),
    and byte-compares against the tile bytes actually stored in the file
    by the stock GDAL write path.

    Args:
        path: A GeoTIFF written by the stock GDAL path (tiled, int16,
            predictor 1 or 2).
        tile: Tile index (row-major over the tile grid), default 0.
        compress: Injectable compressor for tests; default compress_tile.

    Returns:
        OracleVerdict with ok=True only on full byte equality.

    Raises:
        ZstdmtError: If the raster is not a single-band 16-bit tiled
            int16 GeoTIFF with predictor 1 or 2, or if the tile index
            is out of range.

    """
    tif = _tif_info(Path(path))
    if tif.predictor not in (1, 2):
        msg = f"predictor {tif.predictor} not supported by the oracle (need 1 or 2)"
        raise ZstdmtError(msg)
    if tif.bps != 16 or tif.spp != 1:
        msg = f"oracle expects single-band 16-bit data, got {tif.bps}-bit x{tif.spp} band"
        raise ZstdmtError(msg)
    n_tiles = len(tif.tile_bytes)
    if not 0 <= tile < n_tiles:
        msg = f"tile {tile} out of range (raster has {n_tiles} tiles)"
        raise ZstdmtError(msg)
    ntiles_x = (tif.width + tif.tilew - 1) // tif.tilew
    x0 = (tile % ntiles_x) * tif.tilew
    y0 = (tile // ntiles_x) * tif.tileh
    tw = min(tif.tilew, tif.width - x0)
    th = min(tif.tileh, tif.height - y0)
    with rasterio.open(path) as ds:
        arr = ds.read(1, window=Window(x0, y0, tw, th))
    pre = predictor2(arr).tobytes() if tif.predictor == 2 else arr.tobytes()
    fn = compress or compress_tile
    turbo = fn(pre)
    stock = tif.tile_bytes[tile]
    if turbo == stock:
        detail = f"zstd oracle: turbo {len(turbo)} B == stock {len(stock)} B (tile {tile})"
        return OracleVerdict(ok=True, turbo_size=len(turbo), stock_size=len(stock), detail=detail)
    detail = (
        f"zstd oracle mismatch: turbo {len(turbo)} B != stock {len(stock)} B (tile {tile}); "
        "write with the stock path (parallel; serial per-band write is the last resort)"
    )
    return OracleVerdict(ok=False, turbo_size=len(turbo), stock_size=len(stock), detail=detail)
