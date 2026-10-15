"""C-accelerated linear interpolation, bit-identical to ``np.interp``.

``PercentileTransform.fwd``/``inv`` (ppgrid.calibrate) run one
``np.interp`` per block in the hot write path. ``ppgrid.cinterp.c`` clones
the element arithmetic of numpy 2.5.x ``compiled_base.c`` (BSD-3) with the
same IEEE op order (``slope * (x - xp[j]) + fp[j]``, no FMA at -O3 x86-64
baseline), so the output floats are bit-identical to ``np.interp`` for any
xp/fp/x. The search is where the win comes from: numpy's persistent
warm-start guess only hits when consecutive elements land in the same node
bin; on real interpolated blocks the rank index moves several nodes per
element, so most elements pay the ~10-step bisection (~50-60 ns/element
under load). This kernel processes the input in 256-element chunks: two
bisections per chunk bound the whole chunk's j window, then each element
gets a guess-hit fast path or a bisection restricted to that window (zero
searches when a chunk sits in a single node bin). Measured 1.05-1.65x per
block on real pipeline captures (geomean ~1.2x); non-monotonic xp falls
back to a verbatim numpy search clone, so bit-identity holds everywhere.

The extension is a plain CPython module using only the buffer protocol
(no numpy C API), so it builds with one ``cc`` line against the
interpreter headers. ``_load`` is called once at import time:

1. Use a prebuilt ``_cinterp_c-<src-hash>.so`` next to this package or in
   ``~/.cache/ppgrid`` (source hash in the name: a stale build never wins).
2. Otherwise JIT-compile ``cinterp.c`` into the first writable of those
   locations.
3. Otherwise fall back to ``np.interp`` (the fallback is bit-identical by
   construction: it IS np.interp).

User opt-out (issue #109): set ``PPGRID_CINTERP=0`` in the environment
BEFORE importing this module to force the ``np.interp`` fallback — no JIT
build, no C kernel. ``PPGRID_CINTERP=1`` (or the variable unset) keeps the
default. The backend is resolved once at import; the variable is read at
that moment only. Both modes are bit-identical, so the switch never
changes pipeline output bytes.

The GIL is released inside the C loop, so block worker threads
interpolate in parallel (matching numpy's NPY_BEGIN_THREADS behaviour).
"""

from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.util
import os
import subprocess
import sys
import sysconfig
import tempfile
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path

import numpy as np

__all__ = ["backend", "interp", "reason"]

_SO_SUFFIX: str = importlib.machinery.EXTENSION_SUFFIXES[0]
_SRC: Path = Path(__file__).with_name("cinterp.c")

_mod: object | None = None  # the imported C module, or None (fallback)
_reason: str | None = None  # why the C module is unavailable (None = available)
_build_err: str | None = None  # last compiler stderr, for diagnostics
_TMPDIR_ROOT: Path = Path(tempfile.gettempdir()) / "ppgrid-cinterp"  # shared-tmpdir cache


def _candidate_paths() -> list[Path]:
    """Where a compiled module may live, or be built (first writable wins)."""
    if not _SRC.is_file():
        return []
    digest = hashlib.md5(_SRC.read_bytes()).hexdigest()[:12]  # ruff: ignore[hashlib-insecure-hash-function]
    name = f"_cinterp_c-{digest}{_SO_SUFFIX}"
    paths = [_SRC.parent / name]
    with suppress(OSError, RuntimeError):
        paths.append(Path.home() / ".cache" / "ppgrid" / name)
    paths.append(_TMPDIR_ROOT / name)
    return paths


def _tmpdir_owned(path: Path) -> bool:
    """Shared-tmpdir guard: the cache dir must be owned by our effective uid.

    A sticky /tmp lets any local user create ``ppgrid-cinterp/`` first and
    pre-place a file with our exact .so name; importing it would run their
    code as the importing user. On mismatch, treat the candidate as absent.
    """
    if path.parent != _TMPDIR_ROOT:
        return True
    geteuid = getattr(os, "geteuid", None)
    if geteuid is None:
        return True  # no uid concept (non-POSIX): guard not applicable
    try:
        return path.parent.stat().st_uid == geteuid()
    except OSError:
        return True  # missing dir: the build loop creates it


def _try_import(path: Path) -> bool:
    """Import the built module; True on success.

    Raises:
        ImportError: If the module is missing expected functions.

    """
    global _mod, _reason
    try:
        spec = importlib.util.spec_from_file_location("_cinterp_c", str(path))
        if spec is None or spec.loader is None:
            msg = f"no loader for {path}"
            raise ImportError(msg)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        mod = module
        if not callable(getattr(mod, "interp_f64", None)) or not callable(getattr(mod, "interp_f32", None)):
            msg = "module lacks interp_f64/interp_f32"
            raise ImportError(msg)
        _mod = mod
        _reason = None
        return True
    except Exception as exc:  # ruff: ignore[blind-except] - any import failure means fallback
        _mod = None
        _reason = f"import failed ({exc.__class__.__name__})"
        return False


def _build(src: Path, dst: Path) -> bool:
    """Compile src into dst; True on success. Never raises."""
    global _build_err
    inc = sysconfig.get_paths().get("include", "")
    cmd_base = ["-O3", "-ffp-contract=off", "-fPIC", "-shared", "-fvisibility=hidden"]
    if sys.platform == "darwin":
        # macOS ld resolves the Python C-API symbols at LOAD time (the
        # interpreter maps libpython into the extension), not at link time.
        # Raw cc omits the -undefined dynamic_lookup that
        # distutils/setuptools add, so the link dies with
        # "symbol(s) not found for architecture arm64" (issue #123).
        cmd_base += ["-undefined", "dynamic_lookup"]
    compilers = [os.environ.get("CC", "")] if os.environ.get("CC") else []
    compilers += ["cc", "gcc", "clang"]
    for compiler in compilers:
        if not compiler:
            continue
        cmd = [compiler, *cmd_base]
        if inc:
            cmd.append(f"-I{inc}")
        cmd += [str(src), "-o", str(dst)]
        try:
            proc = subprocess.run(cmd, capture_output=True, timeout=300)
        except (OSError, subprocess.SubprocessError) as exc:
            _build_err = f"{compiler}: {exc.__class__.__name__}"
            continue
        if proc.returncode == 0:
            return True
        _build_err = f"{compiler}: {proc.stderr.decode(errors='replace')[-400:]}"
    return False


def _load() -> None:
    """Locate or build the C module exactly once (import-time, main thread).

    PPGRID_CINTERP=0 (documented user opt-out, see module docstring)
    forces the np.interp fallback before any compiler work happens.
    """
    global _reason
    if _mod is not None or _reason is not None:
        return
    if os.environ.get("PPGRID_CINTERP") == "0":
        _reason = "PPGRID_CINTERP=0"
        return
    cands = _candidate_paths()
    if not cands:
        _reason = "cinterp.c not shipped with the package"
        return
    # 1. existing build (delete a broken one and rebuild below)
    for path in cands:
        if path.is_file():
            if not _tmpdir_owned(path):
                continue  # shared tmpdir dir not owned by us: treat as absent
            if _try_import(path):
                return
            # Broken build: delete and rebuild below. On Windows a pyd
            # already loaded by another process stays locked for that
            # process's lifetime (the parent's import keeps the DLL
            # mapped), so it cannot be unlinked and a rebuild cannot
            # replace it either; the next candidate (or the np.interp
            # fallback, which is bit-identical) takes over. A pool child
            # must not crash its import over a file it does not own (CI
            # run 38207801346 W1-B: BrokenProcessPool at this unlink).
            with suppress(OSError):
                path.unlink(missing_ok=True)
    # 2. JIT build into the first writable location
    for path in cands:
        if not _tmpdir_owned(path):
            continue
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            continue
        if not os.access(path.parent, os.W_OK):
            continue
        if _build(_SRC, path) and _try_import(path):
            return
        # Normally a fresh file nobody else holds; suppress anyway: on
        # Windows a build into a location another process has open fails
        # at the gcc step and the cleanup unlink would then raise (CI run
        # 38207801346 W1-B).
        with suppress(OSError):
            path.unlink(missing_ok=True)
    _reason = "no working C compiler (falling back to np.interp)" + (f"; {_build_err}" if _build_err else "")


def _as_contig(a: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(a)


def interp(x: np.ndarray, xp: np.ndarray, fp: np.ndarray, slopes: np.ndarray) -> np.ndarray:
    """``np.interp(x, xp, fp)`` bits, fast.

    Args:
        x: f64 or f32 array (any shape); other dtypes fall back to np.interp.
        xp: 1-D C-contiguous f64 sample points; non-f64 falls back to np.interp.
        fp: 1-D C-contiguous f64 values, same length as xp; non-f64 falls back.
        slopes: 1-D C-contiguous f64, ``(fp[1:]-fp[:-1]) / (xp[1:]-xp[:-1])``,
            precomputed by the caller; non-f64 falls back to np.interp.

    Returns:
        New f64 ndarray, same shape as x, bit-identical to
        ``np.interp(x, xp, fp)`` with default left/right.

    """
    mod = _mod
    if mod is None:
        return np.interp(x, xp, fp)
    xc = _as_contig(x)
    out = np.empty(xc.shape, np.float64)
    call: Callable[..., None]
    if xc.dtype == np.float64:
        call = mod.interp_f64  # type: ignore[attr-defined]
    elif xc.dtype == np.float32:
        call = mod.interp_f32  # type: ignore[attr-defined]
    else:
        return np.interp(x, xp, fp)
    if xp.dtype != np.float64 or fp.dtype != np.float64 or slopes.dtype != np.float64:
        return np.interp(x, xp, fp)  # C kernel takes f64 tables only
    xp_c = _as_contig(xp)
    fp_c = _as_contig(fp)
    sl_c = _as_contig(slopes)
    call(xc, xp_c, fp_c, sl_c, out)
    return out


def available() -> bool:
    """Report whether the C kernel is active (False = np.interp fallback).

    False when PPGRID_CINTERP=0 was set at import time (the documented
    user opt-out), or when no working compiler/build existed.
    """
    return _mod is not None


def backend() -> str:
    """Human-readable backend tag for logs/tests.

    'cinterp (C)' when the kernel is active, else 'np.interp (<reason>)'
    where <reason> is 'PPGRID_CINTERP=0' for the documented user opt-out
    or the compiler/build diagnostic.
    """
    return "cinterp (C)" if _mod is not None else f"np.interp ({_reason or 'disabled'})"


def reason() -> str | None:
    """Why the C kernel is unavailable (None when available).

    'PPGRID_CINTERP=0' when the documented user opt-out forced the
    np.interp fallback; otherwise the compiler/build diagnostic.
    """
    return _reason


_load()
