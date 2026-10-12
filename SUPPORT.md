# Support

## Supported Versions

- **Python 3.11, 3.12, 3.13, 3.14.** All four are tested in CI on every PR
  (`.github/workflows/ci.yml`, `requires-python = ">=3.11"`, `pyproject.toml`
  classifiers 3.11-3.14). Versions outside 3.11-3.14 are unsupported: the
  lockfile is only verified to resolve there, and no fixes will be targeted at
  them.
- **OS**: Linux is the primary target and is covered by CI on every PR; macOS
  is a supported target but has no CI leg yet (declare it in release notes only
  after a macOS leg is green). The C-interp extension
  JIT-builds via `cc`/`gcc`/`clang` and falls back to `np.interp` (bit-identical)
  when no compiler is available.

## Getting Help

- **Bug reports**: Open an issue on GitHub: https://github.com/marzukia/ppgrid/issues
- **Questions**: Open a discussion on GitHub: https://github.com/marzukia/ppgrid/discussions
- **Email**: andryo@mrzk.io

## Scope

This project is maintained in my spare time. I will do my best to respond to issues within a week, but no SLA is guaranteed.
