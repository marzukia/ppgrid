"""CLI verification surface tests: compare, --check, completions (#66/#67/#68/#69).

Covers:
1. compare: identical / different / file form / usage errors / --max-diff.
2. --check: a clean run verifies; a zero-filled out-of-coverage corner and a
   missing band are detected with the exact field, exit 1, outputs kept.
3. completions: deterministic bash/zsh output from the live parser, no heavy
   imports (fresh-process check), usage error on a bad shell.
4. The run CLI stays byte-identical: two runs with the same argv produce the
   same output hashes (parser subcommand append changed nothing for runs).
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.windows import Window

from ppgrid.pipeline import (
    Pipeline,
    _build_parser,
    _make_pipeline,
    _parse_args_cli,
    _post_write_check,
    _resolve_sentinels,
    main,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

# 3-point diagonal (top-left -> bottom-right), so the top-right corner of the
# grid is out of fill-cap coverage: a clean run writes NODATA there, and
# zero-filling it is detectable by --check.
CSV_TEXT = "value,longitude,latitude\n1.0,149.000,-35.000\n2.0,149.033,-35.027\n3.0,149.066,-35.054\n"


def _pts_csv(tmp_path: Path) -> str:
    csv = tmp_path / "pts.csv"
    csv.write_text(CSV_TEXT)
    return str(csv)


def _run(tmp_path: Path, out: Path, *extra: str) -> int:
    """Run the pipeline in-process; return the exit code (0 = normal return)."""
    try:
        main([_pts_csv(tmp_path), "-o", str(out), "--cap-km", "2", "--res", "100", "--workers", "2", *extra])
    except SystemExit as exc:
        return int(exc.value.code or 0)
    return 0


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _outputs_sha(out: Path) -> dict[str, str]:
    return {name: _sha(out / name) for name in ("value.tif", "support_km.tif", "calibration.json")}


def _fresh_python(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, check=False)


# ---------------------------------------------------------------------------
# run CLI byte-identity (the subcommand append must not change run outputs)
# ---------------------------------------------------------------------------


def test_run_cli_byte_identical(tmp_path: Path) -> None:
    """Two runs with the same argv produce byte-identical outputs."""
    assert _run(tmp_path, tmp_path / "a") == 0
    assert _run(tmp_path, tmp_path / "b") == 0
    assert _outputs_sha(tmp_path / "a") == _outputs_sha(tmp_path / "b")


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------


def test_compare_identical_dir(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Self-compare of a run dir exits 0 and reports identical."""
    assert _run(tmp_path, tmp_path / "a") == 0
    with pytest.raises(SystemExit) as exc:
        main(["compare", str(tmp_path / "a"), str(tmp_path / "a")])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "compare: identical" in out
    assert "  cells:     " in out
    assert "0 mismatch" in out


def test_compare_different(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Different resolution -> exit 1 with the structural mismatch named."""
    assert _run(tmp_path, tmp_path / "a") == 0
    assert _run(tmp_path, tmp_path / "b", "--res", "200") == 0
    with pytest.raises(SystemExit) as exc:
        main(["compare", str(tmp_path / "a"), str(tmp_path / "b")])
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "compare: different (2 of 2 bands)" in out
    assert "transform: differ" in out


def test_compare_files(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Two GeoTIFF arguments compare the single named band."""
    assert _run(tmp_path, tmp_path / "a") == 0
    with pytest.raises(SystemExit) as exc:
        main(["compare", str(tmp_path / "a" / "value.tif"), str(tmp_path / "a" / "value.tif")])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "compare: value.tif (" in out
    assert "compare: identical" in out


def test_compare_mixed_types(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Dir vs file is a usage error (exit 2) on stderr."""
    assert _run(tmp_path, tmp_path / "a") == 0
    with pytest.raises(SystemExit) as exc:
        main(["compare", str(tmp_path / "a"), str(tmp_path / "a" / "value.tif")])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "cannot mix directory and file arguments" in err


def test_compare_not_found(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A missing path is a usage error (exit 2)."""
    with pytest.raises(SystemExit) as exc:
        main(["compare", str(tmp_path / "nope"), str(tmp_path / "nope")])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "not found: " in err


def test_compare_missing_arg(tmp_path: Path) -> None:
    """Compare with one path exits 2 (argparse error)."""
    with pytest.raises(SystemExit) as exc:
        main(["compare", str(tmp_path)])
    assert exc.value.code == 2


def test_compare_max_diff(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """--max-diff tolerates small DNs; beyond it the band differs; negative is a usage error."""
    assert _run(tmp_path, tmp_path / "a") == 0
    assert _run(tmp_path, tmp_path / "b") == 0
    # Bump 64 in-coverage pixels by 3 DN in the copy.
    with rasterio.open(tmp_path / "b" / "value.tif", "r+") as ds:
        arr = ds.read(1, window=Window(2, 2, 8, 8)).astype(np.int32) + 3
        ds.write(arr.astype(np.int16), 1, window=Window(2, 2, 8, 8))
    with pytest.raises(SystemExit) as exc:
        main(["compare", str(tmp_path / "a"), str(tmp_path / "b"), "--max-diff", "2"])
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "max diff:  3" in out
    assert "mismatch bbox:" in out
    with pytest.raises(SystemExit) as exc:
        main(["compare", str(tmp_path / "a"), str(tmp_path / "b"), "--max-diff", "3"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "within --max-diff" in out
    assert "64 cell(s) differ" in out
    with pytest.raises(SystemExit) as exc:
        main(["compare", str(tmp_path / "a"), str(tmp_path / "b"), "--max-diff", "-1"])
    assert exc.value.code == 2


# ---------------------------------------------------------------------------
# --check
# ---------------------------------------------------------------------------


def _pipeline_state(tmp_path: Path, out: Path, *extra: str) -> Pipeline:
    """Return a Pipeline with post-ingest state for _post_write_check (dry run)."""
    parser = _build_parser()
    args = _parse_args_cli(parser, [_pts_csv(tmp_path), "-o", str(out), "--cap-km", "2", "--res", "100", *extra])
    _resolve_sentinels(args)  # no-config path of main(): UNSET sentinels -> built-in defaults
    p = _make_pipeline(args, dry_run=True, ui=None)
    p.ingest()
    p.calibrate()
    p.grid()
    return p


def test_check_pass(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """A clean run with --check exits 0 and logs the verification line."""
    with caplog.at_level("INFO", logger="ppgrid"):
        assert _run(tmp_path, tmp_path / "a", "--check") == 0
    msgs = [r.getMessage() for r in caplog.records if r.name == "ppgrid.pipeline"]
    assert any("verified against the run plan" in m for m in msgs)


def test_check_detects_zero_corner(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Zero-filled out-of-coverage corner: exact field named, exit 1, outputs kept."""
    assert _run(tmp_path, tmp_path / "a") == 0
    out = tmp_path / "a"
    with rasterio.open(out / "value.tif", "r+") as ds:
        ds.write(np.zeros((16, 16), dtype=np.int16), 1, window=Window(ds.width - 16, 0, 16, 16))
    p = _pipeline_state(tmp_path, out)
    with pytest.raises(SystemExit) as exc:
        _post_write_check(p, out)
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "top-right corner" in err
    assert "zero pixel(s) with no input point within the 2 km fill-cap window" in err
    # verify, don't repair: outputs are kept
    assert (out / "value.tif").is_file()
    assert (out / "support_km.tif").is_file()


def test_check_missing_band(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A missing final raster: named, exit 1, no traceback."""
    assert _run(tmp_path, tmp_path / "a") == 0
    out = tmp_path / "a"
    (out / "support_km.tif").unlink()
    p = _pipeline_state(tmp_path, out)
    with pytest.raises(SystemExit) as exc:
        _post_write_check(p, out)
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "support_km.tif: file missing" in err


# ---------------------------------------------------------------------------
# completions
# ---------------------------------------------------------------------------


def test_completions_deterministic(capsys: pytest.CaptureFixture[str]) -> None:
    """bash/zsh output is stable across calls and carries the live flags."""
    with pytest.raises(SystemExit) as exc:
        main(["completions", "bash"])
    assert exc.value.code == 0
    bash1 = capsys.readouterr().out
    with pytest.raises(SystemExit) as exc:
        main(["completions", "bash"])
    assert exc.value.code == 0
    bash2 = capsys.readouterr().out
    assert bash1 == bash2
    for token in ("--check", "--cap-km", "--turbo", "compare", "completions"):
        assert token in bash1, f"{token!r} missing from bash completions"
    with pytest.raises(SystemExit) as exc:
        main(["completions", "zsh"])
    assert exc.value.code == 0
    zsh1 = capsys.readouterr().out
    with pytest.raises(SystemExit) as exc:
        main(["completions", "zsh"])
    assert exc.value.code == 0
    assert capsys.readouterr().out == zsh1
    for token in ("--check", "--cap-km", "compare", "completions"):
        assert token in zsh1, f"{token!r} missing from zsh completions"


def test_completions_bad_shell(capsys: pytest.CaptureFixture[str]) -> None:
    """An unknown shell is a usage error (exit 2)."""
    with pytest.raises(SystemExit) as exc:
        main(["completions", "fish"])
    assert exc.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_completions_defer_native_modules() -> None:
    """Completions exits 0 in a fresh process without loading the heavy stack."""
    code = (
        "import sys\n"
        "from ppgrid.pipeline import main\n"
        "try:\n"
        "    main(['completions', 'bash'])\n"
        "except SystemExit as e:\n"
        "    assert e.code in (0, None), f'completions exited {e.code}'\n"
        "for mod in ('rasterio', 'pyproj', 'pandas', 'rich', 'ppgrid.ui'):\n"
        "    assert mod not in sys.modules, f'{mod} loaded by completions'\n"
    )
    result = _fresh_python(code)
    assert result.returncode == 0, result.stderr


def test_compare_help_defers_native_modules() -> None:
    """Compare --help exits 0 in a fresh process without loading the heavy stack."""
    code = (
        "import sys\n"
        "from ppgrid.pipeline import main\n"
        "try:\n"
        "    main(['compare', '--help'])\n"
        "except SystemExit as e:\n"
        "    assert e.code in (0, None), f'compare --help exited {e.code}'\n"
        "for mod in ('rasterio', 'pyproj', 'pandas', 'rich', 'ppgrid.ui'):\n"
        "    assert mod not in sys.modules, f'{mod} loaded by compare --help'\n"
    )
    result = _fresh_python(code)
    assert result.returncode == 0, result.stderr
