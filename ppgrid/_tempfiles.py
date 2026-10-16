"""Shared temp-file helpers for the pipeline and pullpush (issue #15/#45).

Leaves both ppgrid.pipeline and ppgrid.pullpush importable: the import
direction is pipeline -> pullpush, so the helper lives here instead.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np


def _open_fresh_memmap(
    path: Path,
    dtype: type | np.dtype[Any],
    shape: tuple[int, ...],
) -> np.memmap:
    """Create `path` as a fresh 0600 regular file, then open it for memmap write.

    The path is unlinked first, so a pre-planted symlink is replaced, never
    followed (its target must not be overwritten). unlink-then-O_EXCL is not
    atomic: in a two-creator race exactly one wins and the loser gets a
    plain FileExistsError instead of silently writing into a foreign file
    (issue #15). Mode 0600: temps hold raw coordinates + values and should
    not be world-readable (the default 0644 leaked them).

    Args:
        path: Destination path in the output dir.
        dtype: Memmap element type.
        shape: Memmap shape.

    Returns:
        Writable memmap over the fresh file.

    Raises:
        ValueError: If path exists as a directory (cannot become a file).
        PermissionError: A genuine permission failure on a regular file
            (propagates from unlink on every platform).

    A FileExistsError from os.open means another creator won the
    unlink/open race (O_EXCL made it visible); it propagates as-is.

    """
    try:
        path.unlink(missing_ok=True)
    except IsADirectoryError:
        # Linux: unlink() on a directory -> EISDIR.
        msg = f"cannot create fresh temp file {path}: path is a directory"
        raise ValueError(msg) from None
    except PermissionError:
        # macOS/BSD: unlink() on a directory -> EPERM (Operation not
        # permitted), not EISDIR. A genuine permission error on a regular
        # file is re-raised untouched.
        if path.is_dir():
            msg = f"cannot create fresh temp file {path}: path is a directory"
            raise ValueError(msg) from None
        raise
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
    os.close(fd)
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
