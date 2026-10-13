#!/usr/bin/env bash
# Clean-venv artifact gate (issue #98) — shared by .github/workflows/ci.yml
# (artifact job) and .github/workflows/publish.yml (pre-publish gate).
#
# Installs a built artifact (wheel or sdist) in a fresh stdlib venv with
# plain pip (no uv-managed env, no dev extras), then verifies:
#   1. the artifact installs with a fresh pip resolution on this OS
#   2. ppgrid --version == pyproject version (s2 P0)
#   3. the installed console script completes a realistic e2e run OUTSIDE
#      the repo tree (melb 500m, the committed melb500 anchor config,
#      --skip-calibration because the anchors were cut with it)
#   4. the e2e outputs match the committed anchors: byte-exact on canonical
#      x86_64 Linux, functional-equivalent elsewhere (bench/check_artifact.py)
#   5. (parquet) a realistic run from a Parquet input, anchor-gated too
#   6. pip check passes in the venv (s2 P1)
#
# Usage:
#   scripts/check_artifact.sh <artifact> [parquet]
#     <artifact>  path to dist/<wheel> or dist/<sdist> (single file)
#     parquet     optional: also run the Parquet-input e2e (wheel mode;
#                 the wheel venv always carries the [parquet] extra so the
#                 pinned tree matches uv.lock on every leg)
#
# Exit 0 = all checks passed; 1 = any check failed.
#
# WHY THE PIN STEP: the e2e venv first installs the wheel with a fresh
# resolution (item 1), then pins runtime deps to the uv.lock versions
# before the anchor gate. The committed anchors were cut on that exact
# toolchain: a fresh pip resolution today pulls newer rasterio/pyproj/
# affine, which drifts the grid transform by ~1e-15 and changes the output
# bytes on EVERY OS (decoded pixels stay identical). Byte-exact is only
# reproducible on the anchor-cut toolchain. The sdist test does not pin:
# its acceptance is "install + console script works", not byte-identity.
set -euo pipefail

ARTIFACT=${1:?usage: check_artifact.sh <artifact> [parquet]}
PARQUET=${2:-}

REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"

case "$ARTIFACT" in
  *.whl)    MODE=wheel; EXTRAS=[parquet] ;;
  *.tar.gz) MODE=sdist; EXTRAS= ;;
  *) echo "[fail] unsupported artifact '$ARTIFACT' (want dist/<wheel> or dist/<sdist>)" >&2; exit 1 ;;
esac

# Fresh stdlib venv (uv only provides the CPython 3.13 interpreter; the
# venv itself is plain `python -m venv` + pip, per acceptance item 1).
VENV=/tmp/ppgrid-artifact-venv
rm -rf "$VENV"
PY=$(uv python find 3.13)
"$PY" -m venv "$VENV"
PIP="$VENV/bin/pip"

echo "[info] $MODE: fresh pip install of $ARTIFACT"
"$PIP" install --no-cache-dir --quiet "${ARTIFACT}${EXTRAS}"

# Item 2: installed console-script version == pyproject version.
WANT=$("$VENV/bin/python" -c 'import sys, tomllib; print(tomllib.load(open(sys.argv[1], "rb"))["project"]["version"])' "$REPO/pyproject.toml")
GOT=$("$VENV/bin/ppgrid" --version)
if [ "$GOT" != "ppgrid $WANT" ]; then
  echo "[fail] ppgrid --version: got '$GOT', want 'ppgrid $WANT'" >&2
  exit 1
fi
echo "[ok]   ppgrid --version == $WANT"

if [ "$MODE" = wheel ]; then
  echo "[info] pinning runtime deps to uv.lock (anchor-cut toolchain)"
  PINNED=/tmp/ppgrid-pinned-reqs.txt
  uv export --frozen --all-extras --no-extra dev --no-emit-project --no-hashes --format requirements-txt > "$PINNED"
  "$PIP" install --no-cache-dir --quiet -r "$PINNED"
fi

# Item 3: realistic e2e run, CWD outside the repo tree (so the installed
# package is what runs, never the checkout).
OUT=/tmp/ppgrid-artifact-e2e
rm -rf "$OUT"
cd /
"$VENV/bin/ppgrid" "$REPO/data/melb_houses.csv" --value-col price --res 500 --cap-km 10 --skip-calibration --out "$OUT"
test -f "$OUT/value.tif"
test -f "$OUT/support_km.tif"
echo "[ok]   e2e run completed: $OUT"

if [ "$MODE" = wheel ]; then
  "$VENV/bin/python" "$REPO/bench/check_artifact.py" --out "$OUT"

  if [ -n "$PARQUET" ]; then
    PQ=/tmp/ppgrid-artifact-melb.parquet
    OUT2=/tmp/ppgrid-artifact-e2e-parquet
    rm -f "$PQ"
    rm -rf "$OUT2"
    "$VENV/bin/python" -c 'import pandas as pd, sys; pd.read_csv(sys.argv[1]).to_parquet(sys.argv[2], index=False)' \
      "$REPO/data/melb_houses.csv" "$PQ"
    "$VENV/bin/ppgrid" "$PQ" --value-col price --res 500 --cap-km 10 --skip-calibration --out "$OUT2"
    test -f "$OUT2/value.tif"
    test -f "$OUT2/support_km.tif"
    echo "[ok]   parquet e2e run completed: $OUT2"
    "$VENV/bin/python" "$REPO/bench/check_artifact.py" --out "$OUT2"
  fi
fi

# Item 4: no broken requirements in the venv.
"$PIP" check
echo "[ok]   artifact gate passed: $ARTIFACT"
