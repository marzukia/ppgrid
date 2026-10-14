"""Public API surface + library-mode guard tests (issue #109).

1. ``ppgrid.__all__`` is the explicit beta surface (Pipeline, main,
   version); the module docstring names what is internal.
2. Library use (``Pipeline`` with ``ui=None`` in a fresh interpreter)
   prints nothing directly to stdout/stderr, raises no SystemExit, and
   any stderr line is a ppgrid-logging record.
3. Every ``__all__`` name has a docstring; ``Pipeline``'s documents
   units, CRS expectations, side effects and exceptions.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import pytest

import ppgrid
from ppgrid import pipeline as pipeline_mod

REPO_ROOT = Path(__file__).resolve().parents[1]

# The probe installs this handler on the `ppgrid` logger: every line a
# logger record produces on stderr then carries the prefix, so any
# unprefixed line is a direct print (issue #109's "stderr only via the
# ppgrid logger" made checkable).
_LOG_PREFIX = "PPGLOG "


# ---------------------------------------------------------------------------
# Surface + docstrings (in-process)
# ---------------------------------------------------------------------------


def test_all_is_the_beta_surface() -> None:
    """__all__ is exactly the beta public surface; internals are named (#109)."""
    assert ppgrid.__all__ == ["Pipeline", "main", "version"]
    doc = ppgrid.__doc__ or ""
    assert "beta" in doc
    for mod in (
        "ppgrid.turbop",
        "ppgrid.ui",
        "ppgrid.zstdmt",
        "ppgrid.calibrate",
        "ppgrid.pullpush",
        "ppgrid._cinterp",
        "ppgrid.idwgrid",
    ):
        assert mod in doc, f"{mod} not named as internal in the package docstring"


def test_public_names_resolve_to_pipeline_objects() -> None:
    """The lazy public surface is the pipeline's objects (identity)."""
    assert ppgrid.Pipeline is pipeline_mod.Pipeline
    assert ppgrid.main is pipeline_mod.main
    assert ppgrid.TurboInfeasibleError is pipeline_mod.TurboInfeasibleError
    assert ppgrid.version() == ppgrid.__version__
    assert ppgrid.version()


def test_all_names_have_docstrings() -> None:
    """Every __all__ name is documented (#109 item 4)."""
    for name in ppgrid.__all__:
        obj = getattr(ppgrid, name)
        assert (obj.__doc__ or "").strip(), f"{name} has no docstring"


def test_pipeline_docstring_documents_the_contract() -> None:
    """Pipeline.__doc__ covers units, CRS, side effects, exceptions (#109 item 4)."""
    doc = pipeline_mod.Pipeline.__doc__ or ""
    # Units.
    assert "metres" in doc
    assert "kilometres" in doc
    # CRS expectations (the three default EPSG codes).
    assert "EPSG" in doc
    assert "4326" in doc
    assert "6933" in doc
    assert "3857" in doc
    # Side effects (the written files).
    assert "value.tif" in doc
    assert "support_km.tif" in doc
    assert "calibration.json" in doc
    # Exceptions.
    assert "ValueError" in doc
    assert "OSError" in doc
    assert "CRSError" in doc
    assert "TurboInfeasibleError" in doc


def test_ui_note_library_mode_logs_not_prints(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    """_ui_note in library mode routes to the ppgrid logger, not print (#109).

    CLI mode always has a UI attached (main()), so the ui=None branch is
    library-only; every fallback warning a library run can hit (zstd
    oracle, tile layout guards, stale output) travels the logger.
    """
    import logging

    # Simulate a fresh library process: drop any CLI handler a previous
    # main() left on the ppgrid logger (_configure_logging is process-global).
    root = logging.getLogger("ppgrid")
    for h in list(root.handlers):
        root.removeHandler(h)
    monkeypatch.setitem(pipeline_mod._CURRENT_UI, "ui", None)  # ruff: ignore[private-member-access]
    with caplog.at_level(logging.WARNING, logger="ppgrid.pipeline"):
        pipeline_mod._ui_note("warn", "some fallback happened")  # ruff: ignore[private-member-access]
        pipeline_mod._ui_note("error", "something failed")  # ruff: ignore[private-member-access]
    out, err = capsys.readouterr()
    assert not out
    assert not err
    msgs = [r.getMessage() for r in caplog.records]
    assert "[warn] some fallback happened" in msgs
    assert "error: something failed" in msgs


# ---------------------------------------------------------------------------
# Library-mode guard (fresh interpreter, #109 item 2)
# ---------------------------------------------------------------------------

_LIB_PROBE = r"""
import io
import json
import logging
import sys
from contextlib import redirect_stdout, redirect_stderr

csv = sys.argv[1]
out = sys.argv[2]
mode = sys.argv[3]
verdict_path = sys.argv[4]

# All ppgrid-logger stderr lines carry the prefix; direct prints don't.
_h = logging.StreamHandler(stream=sys.stderr)
_h.setFormatter(logging.Formatter("PPGLOG %(message)s"))
logging.getLogger("ppgrid").addHandler(_h)

if mode == "strict":
    # Full-AU A9 anchor geometry @ 16 GB: the shared path is infeasible
    # (per_box), the shape --turbo-strict rejects. resolve_cap + the
    # ceiling readers are stubbed so the shape holds on any machine.
    from ppgrid import pipeline as pm

    pm.turbop.resolve_cap = lambda preset, *a, **k: (preset, "explicit")
    pm.turbop._read_physical_gb = lambda: 128.0
    pm.turbop._read_cgroup_max_gb = lambda: None
    orig = pm.Pipeline._turbo_precheck

    def a9_hack(self):
        self.n, self.nx_padded, self.ny_padded, self.res, self.cap_km_val = (
            400_000, 41_472, 38_400, 100.0, 25
        )
        orig(self)

    pm.Pipeline._turbo_precheck = a9_hack
    kwargs = {"turbo": True, "turbo_cap_gb": 16.0, "turbo_strict": True}
else:
    kwargs = {}

import ppgrid  # noqa: E402

buf_out = io.StringIO()
buf_err = io.StringIO()
with redirect_stdout(buf_out), redirect_stderr(buf_err):
    p = ppgrid.Pipeline(
        csv, "value", "lng", "lat", out, res=5000.0, cap_km=10.0, workers=1, ui=None, **kwargs
    )
    try:
        p.run()
        result = {"status": "ok"}
    except SystemExit as e:
        result = {"status": "systemexit", "code": e.code}
    except BaseException as e:
        result = {"status": "exception", "type": type(e).__qualname__, "message": str(e)}

with open(verdict_path, "w") as f:
    json.dump({"result": result, "stdout": buf_out.getvalue(), "stderr": buf_err.getvalue()}, f)
"""


def _small_csv(path: Path) -> Path:
    """400 deterministic points around Melbourne (fast library run)."""
    rng = np.random.default_rng(20261014)
    n = 400
    lng = 144.9 + rng.random(n) * 0.1
    lat = -37.8 - rng.random(n) * 0.1
    value = rng.integers(100_000, 2_000_000, n).astype(float)
    pd.DataFrame({"value": value, "lng": lng, "lat": lat}).to_csv(path, index=False)
    return path


def _run_lib_guard(tmp_path: Path, mode: str) -> dict[str, object]:
    """Run the library probe in a fresh interpreter; check the no-print contract."""
    csv = _small_csv(tmp_path / f"pts_{mode}.csv")
    out = tmp_path / f"out_{mode}"
    verdict = tmp_path / f"verdict_{mode}.json"
    proc = subprocess.run(
        [sys.executable, "-c", _LIB_PROBE, str(csv), str(out), mode, str(verdict)],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=REPO_ROOT,
        check=False,
    )
    assert proc.returncode == 0, f"guard probe crashed:\n{proc.stdout}\n{proc.stderr}"
    # Process-level stdout: nothing, ever (the --json/--help contract).
    assert not proc.stdout, f"library run wrote to stdout: {proc.stdout!r}"
    # Process-level stderr: only prefixed ppgrid-logger lines.
    for line in proc.stderr.splitlines():
        assert line.startswith(_LOG_PREFIX), f"non-logger stderr in library mode: {line!r}"
    v = json.loads(verdict.read_text())
    # Python-level capture on the swapped streams: the same contract.
    assert not v["stdout"], f"direct print to stdout in library mode: {v['stdout']!r}"
    for line in v["stderr"].splitlines():
        assert line.startswith(_LOG_PREFIX), f"direct print to stderr in library mode: {line!r}"
    return v


def test_library_run_no_prints_no_systemexit(tmp_path: Path) -> None:
    """A library run (ui=None) completes silently: no prints, no SystemExit (#109)."""
    v = _run_lib_guard(tmp_path, "ok")
    assert v["result"] == {"status": "ok"}, v["result"]
    out = tmp_path / "out_ok"
    assert (out / "value.tif").is_file()
    assert (out / "support_km.tif").is_file()
    assert (out / "calibration.json").is_file()


def test_library_strict_infeasible_raises_not_exits(tmp_path: Path) -> None:
    """Strict-infeasible library run raises TurboInfeasibleError, not SystemExit (#109)."""
    v = _run_lib_guard(tmp_path, "strict")
    result = v["result"]
    assert isinstance(result, dict)
    assert result.get("status") == "exception", result
    assert result.get("type") == "TurboInfeasibleError", result
    message = result.get("message")
    assert isinstance(message, str)
    assert message.startswith("error: --turbo est peak "), message
    assert "shared path infeasible" in message, message
