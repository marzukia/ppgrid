# Contributing to ppgrid

Read [AGENTS.md](AGENTS.md) first — it carries the repo rules (the do-not-touch
list, the byte-identity discipline, release flow) and is written for humans and
agents alike. This file is the short path for a first PR.

## Setup

Python 3.11+, `uv` managed:

```bash
git clone https://github.com/marzukia/ppgrid
cd ppgrid
uv sync --extra dev
uv run pytest                                    # full suite
uv run ruff check . && uv run ruff format --check .
```

## Code style

- `ruff` for lint + format (line-length 120, target py311). The pre-commit
  config runs both.
- No new dependencies without a stated reason in the PR description. CI does
  `pip check` in a clean venv, and the artifact gate pins runtime deps to
  `uv export --frozen` (fresh pip resolution drifts rasterio/pyproj ~1e-15 off
  the lock and moves output bytes — see AGENTS.md).

## The byte-identity discipline (this repo's core contract)

ppgrid output is anchor-tested:

- `bench/check_anchors.py` re-runs the pipeline against committed rasters.
- The `artifact` CI job installs the built wheel in a clean stdlib venv and
  gate-checks its outputs byte-exact on canonical x86_64 Linux
  (functional-equivalence elsewhere).

If your change alters output **bytes** (not just speed or CLI text):

1. Say so in the PR description, with why the bytes must change.
2. Regenerate the anchors (`examples/run_melb.py` for the melb set; the
   committed rasters under `examples/` ARE the anchors — `bench/check_anchors.py`
   derives the shas at run time).
3. Byte-exact anchors reproduce only on the locked toolchain — compare on the
   same `uv.lock` you built against.

Two more invariants the tests pin:

- **Library surface**: `Pipeline(...).run()` with no CLI (`ui=None`) must stay
  byte-identical across CLI/presentation changes.
- **CLI contract**: exit codes (documented in the `main()` docstring) and the
  plain non-TTY renderer (zero escapes, one line per phase) are tested.
  `--plan --json` and `run_summary.json` are machine-readable surfaces with
  `schema_version` — changes to their key sets need a version bump discussion,
  not silent renames.

## Tests

- `uv run pytest` — the full suite must pass; new behaviour ships with tests.
- Prefer parse/behaviour assertions over string snapshots
  (see `tests/test_plan_json.py` for the pattern).
- When you touch CLI exit paths, add or extend the asserting tests for every
  documented exit code.

## PR flow

- Branch -> PR -> required review -> CI -> Andryo merges. Nothing lands on
  `main` directly.
- CI: test matrix (ubuntu + macos, Python 3.11–3.14) + artifact gate
  (clean-venv wheel install, e2e run outside the repo tree, anchor check,
  `pip check`) + vuln scan (`pip-audit`, zero suppressions — triage means
  bumping the dep and relocking).
- Keep `--help` output stable unless the PR is about help; it is snapshot-
  tested.

## Docs

- The README is the PyPI `long_description`: **every image must be an absolute
  `https://raw.githubusercontent.com/marzukia/ppgrid/main/examples/...` URL**
  (relative paths 404 on PyPI's static host). Details in AGENTS.md.
- Keep the CHANGELOG in step with every version bump (`pyproject.toml` version
  is the source of truth; `publish.yml` uploads the wheel after the clean-venv
  gate).
