# Support

## Supported Versions

- **Python 3.11, 3.12, 3.13, 3.14.** All four are tested in CI on every PR
  (`.github/workflows/ci.yml`, `requires-python = ">=3.11"`, `pyproject.toml`
  classifiers 3.11-3.14). Versions outside 3.11-3.14 are unsupported: the
  lockfile is only verified to resolve there, and no fixes will be targeted at
  them.

## Operating Systems

Every OS in the CI matrix runs the full test suite in CI on all four
Pythons (Windows is currently excluded from the matrix — see below), plus an
anchor gate (`bench/check_anchors.py`) that re-runs the CI-safe configs
(melb10 + melb500, 2.4 GB peak) and fails the leg if the outputs drift
from the committed `examples/` anchors: byte-exact on the canonical
x86_64 Linux platform, functional-equivalence (identical grid/metadata,
near-identical decoded pixels) on the other legs.

- **Linux** (primary target)
  - C interp JIT active (cc/gcc); np.interp fallback if no compiler.
  - CI: ubuntu-latest, all 4 Pythons + anchor gate.
  - equakes-scale runs verified here (8.1-8.4 GB peak RSS on hydrogen;
    too big for the CI runners).
- **macOS** (arm64 CI)
  - C interp JIT active (clang); np.interp fallback if no compiler.
  - CI: macos-latest, all 4 Pythons + anchor gate (proves the cross-OS
    C-build bytes).
- **Windows** (NOT in the CI matrix — excluded 2026-10-12, Unix-only scope)
  - Portability fixes from this wave are in the tree (CDLL guard,
    O_NOFOLLOW/ELOOP defaults, memmap-holder releases, platform-aware
    gates) as the re-enable base. Last windows-latest run: all legs passed
    except one anchor test (melb10 DN data drift on the 3.11 pyproj wheel).
  - Re-enable: add windows-latest back to the CI matrix once that anchor
    question is resolved (grid-origin quantization or a non-canonical DN
    bound).

The interpolation math is backend-agnostic by construction (the C kernel is
bit-identical to np.interp, pinned by the test suite), so results do not
depend on which OS or backend produced them.

## Getting Help

- **Bug reports**: Open an issue on GitHub: https://github.com/marzukia/ppgrid/issues
- **Questions**: Open a discussion on GitHub: https://github.com/marzukia/ppgrid/discussions
- **Email**: andryo@mrzk.io

## Scope

This project is maintained in my spare time. I will do my best to respond to issues within a week, but no SLA is guaranteed.
