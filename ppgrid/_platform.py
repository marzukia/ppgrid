"""Small platform predicate shared by the tests and the anchor gate.

The committed byte anchors (tests/fixtures/*, examples/*) were generated on
x86_64 Linux. On other platforms the vendored PROJ/GEOS/GDAL/zstd wheels emit
different (functionally equivalent) tile bytes, so byte-identity gates must
apply only on the canonical platform; elsewhere the gates compare decoded
arrays + metadata instead.

"""

from __future__ import annotations

import platform
import sys

__all__ = ["is_canonical_platform"]


def is_canonical_platform() -> bool:
    """Return True on the anchor-generation platform (x86_64 Linux).

    aarch64 Linux is treated as non-canonical (conservative): the manylinux
    aarch64 wheels happen to match the x86 anchors today, but the byte gate
    is only guaranteed where the anchors were actually cut.

    """
    return sys.platform == "linux" and platform.machine().lower() in ("x86_64", "amd64")
