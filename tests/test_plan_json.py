"""--plan --json machine-readable plan + schema_version (issue #105).

Covers: stdout purity (exactly one JSON object, nothing else), the plan
key sets (top level, grid/turbo/ingest/estimated/outputs mirroring
run_summary.json names), value consistency between the JSON and the
human --plan printer (single source of truth), the estimated block
(s30: every value under "estimated" is an estimate), overwrite
behaviour, the --turbo 16 delta, and the additive schema_version on
run_summary.json.
"""

import json
from pathlib import Path

import pytest

from ppgrid import pipeline as _pl

# Key sets: the plan JSON and run_summary.json share names where the same
# data exists (#105 acceptance box 2). grid mirrors run_summary's grid
# exactly; turbo mirrors run_summary's turbo (decision/plan asdict shapes).
_PLAN_TOP_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "ppgrid_version",
        "input",
        "seed",
        "transform",
        "cap_km",
        "grid",
        "ingest",
        "workers",
        "strategy",
        "turbo",
        "estimated",
        "outputs",
        "out_dir",
        "overwrite",
    }
)
_GRID_KEYS = frozenset(
    {
        "nx",
        "ny",
        "cells",
        "res_m",
        "x0_m",
        "y0_m",
        "nx_padded",
        "ny_padded",
        "levels",
        "step",
        "halo",
        "bsize",
        "nbx",
        "nby",
        "n_tasks",
        "n_empty_blocks",
        "src_crs",
        "work_crs",
        "out_crs",
        "saturation",
        "scale",
        "percentile_step",
        "block_size",
        "compress",
    }
)
_TURBO_KEYS = frozenset({"enabled", "decision", "plan", "budget_bytes"})
_DECISION_KEYS = frozenset({"path", "regime", "est_peak_gb", "budget_gb", "exit_code", "message", "summary"})
_PLAN_TURBO_KEYS = frozenset(
    {
        "cap_gb",
        "n_cells",
        "regime",
        "box_chunks",
        "descent_bands",
        "workers",
        "write_scratch_tiles",
        "val_sup_memmap",
        "zstd_ctx_bytes",
        "est_peak_bytes",
        "budget_bytes",
        "sizing_bytes",
        "wall_s",
        "wall_range_s",
    }
)
_ESTIMATED_KEYS = frozenset({"peak_ram_gb", "disk_bytes"})
_RUN_SUMMARY_TOP_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "ppgrid_version",
        "seed",
        "transform",
        "cap_km",
        "grid",
        "ingest",
        "calibration",
        "timings_ms",
        "volumetrics",
        "turbo",
        "outputs",
        "cli",
        "git_commit",
    }
)


def _argv(tmp_path: Path, out: Path, *extra: str) -> list[str]:
    """Tiny 2-point CSV (same pattern as test_cli_polish) + plan argv."""
    csv = tmp_path / "pts.csv"
    csv.write_text("value,lng,lat\n1.0,149.0,-35.0\n2.0,149.1,-35.1\n")
    return [
        str(csv),
        "-o",
        str(out),
        "--value-col",
        "value",
        "--lng-col",
        "lng",
        "--lat-col",
        "lat",
        "--res",
        "100",
        "--block",
        "2048",
        "--workers",
        "2",
        *extra,
    ]


def _plan_json(tmp_path: Path, out: Path, capsys: pytest.CaptureFixture[str], *extra: str) -> tuple[dict, str]:
    """Run --plan (+extra) and parse the plan doc from stdout."""
    with pytest.raises(SystemExit) as exc:
        _pl.main(_argv(tmp_path, out, "--plan", *extra))
    assert exc.value.code == 0
    captured = capsys.readouterr()
    return json.loads(captured.out), captured.out


# ---------------------------------------------------------------------------
# stdout purity + key sets
# ---------------------------------------------------------------------------


def test_plan_json_stdout_is_single_object(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Acceptance 1: stdout is exactly one parseable JSON object, nothing else.

    json.loads on the WHOLE captured stdout fails on any extra line
    (log line, summary block).
    """
    out, text = _plan_json(tmp_path, tmp_path / "out", capsys, "--json")
    assert not (tmp_path / "out").exists()  # plan mode creates nothing
    assert text.lstrip().startswith("{")
    assert text.endswith("}\n")
    assert out["schema_version"] == 1
    assert out["status"] == "plan"


def test_plan_json_full_key_set(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Acceptance 5: the plan doc's key sets (top level + sub-dicts)."""
    doc, _ = _plan_json(tmp_path, tmp_path / "out", capsys, "--json")
    assert set(doc) == _PLAN_TOP_KEYS
    assert set(doc["grid"]) == _GRID_KEYS
    assert set(doc["ingest"]) == {"rows_read", "points", "dropped_nan_inf"}
    assert set(doc["estimated"]) == _ESTIMATED_KEYS
    assert set(doc["outputs"]) == {"value", "support"}
    assert doc["turbo"] is None  # no --turbo: the key exists, value null


def test_plan_json_grid_crs_and_outputs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Acceptance 3: resolved src/work/out CRS, grid extent/dims/cells, outputs."""
    out, _ = _plan_json(tmp_path, tmp_path / "out", capsys, "--json")
    g = out["grid"]
    assert g["src_crs"] == 4326
    assert g["work_crs"] == 6933
    assert g["out_crs"] == 3857
    assert g["cells"] == g["nx"] * g["ny"]
    assert g["cells"] > 0
    assert g["res_m"] == 100.0
    assert g["nx_padded"] >= g["nx"]
    assert g["ny_padded"] >= g["ny"]
    assert out["input"].endswith("pts.csv")
    assert out["out_dir"] == str(tmp_path / "out")
    assert out["outputs"]["value"] == str(tmp_path / "out" / "value.tif")
    assert out["outputs"]["support"] == str(tmp_path / "out" / "support_km.tif")
    assert out["overwrite"] == "create"
    assert out["ingest"]["rows_read"] == 2
    assert out["ingest"]["points"] == 2


def test_plan_json_workers_strategy_estimated(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Acceptance 3: strategy + workers + estimated block (s30 labelled).

    Every value under "estimated" is an estimate, not a measurement:
    disk_bytes = raw int16 payload of both bands, peak_ram_gb > 0.
    """
    out, _ = _plan_json(tmp_path, tmp_path / "out", capsys, "--json")
    assert out["workers"] == 2
    assert out["strategy"] == "standard, workers 2"
    est = out["estimated"]
    assert est["disk_bytes"] == out["grid"]["cells"] * 2 * 2  # value + support, int16
    assert est["peak_ram_gb"] > 0


# ---------------------------------------------------------------------------
# --turbo 16 delta
# ---------------------------------------------------------------------------


def test_plan_json_turbo_key_sets_and_values(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The turbo plan mirrors run_summary's turbo shape; est peak consistent."""
    out, _ = _plan_json(tmp_path, tmp_path / "out", capsys, "--json", "--turbo", "16")
    assert set(out["turbo"]) == _TURBO_KEYS
    t = out["turbo"]
    assert t["enabled"] is True
    assert set(t["decision"]) == _DECISION_KEYS
    assert set(t["plan"]) == _PLAN_TURBO_KEYS
    # cap_gb / budget / est peak live under the run_summary names.
    assert t["plan"]["cap_gb"] > 0
    assert t["decision"]["budget_gb"] > 0
    assert t["decision"]["exit_code"] == 0
    assert t["decision"]["path"] in ("shared", "per_box")
    assert out["estimated"]["peak_ram_gb"] == t["decision"]["est_peak_gb"]
    assert out["strategy"].startswith(turbop_regime_prefix(t["decision"]["regime"]))
    assert out["strategy"].endswith(f"workers {t['plan']['workers']}")


def turbop_regime_prefix(regime: str) -> str:
    """Plain-language regime prefix as in the strategy line (mirrors #103)."""
    return {
        "A": "full in-RAM",
        "B": "input in-RAM, banded in-flight",
        "C": "file-backed (page-cache)",
        "per_box": "per-box low-RAM fallback",
    }[regime]


# ---------------------------------------------------------------------------
# single source of truth: human printer and JSON read the same dict
# ---------------------------------------------------------------------------


def test_plan_text_and_json_share_source(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Every value in the human --plan lines equals its JSON counterpart."""
    doc, _ = _plan_json(tmp_path, tmp_path / "out", capsys, "--json")
    with pytest.raises(SystemExit) as exc:
        _pl.main(_argv(tmp_path, tmp_path / "out", "--plan"))
    assert exc.value.code == 0
    text = capsys.readouterr().out
    g = doc["grid"]
    for line in (
        "ppgrid run plan (--plan: nothing written, exit 0)",
        f"  input:         {doc['input']}",
        f"  grid extent:   x0={g['x0_m']:.3f} m, y0={g['y0_m']:.3f} m (work CRS EPSG:{g['work_crs']}, metres)",
        (
            f"  grid cells:    {g['nx']} x {g['ny']} = {g['cells']:,} cells @ res={g['res_m']:g} m "
            f"(padded {g['nx_padded']} x {g['ny_padded']}, step {g['step']})"
        ),
        f"  levels:        {g['levels']} (halo {g['halo']}, cap {doc['cap_km']:g} km)",
        (
            f"  blocks:        {g['nbx']} x {g['nby']} of bsize={g['bsize']} "
            f"({g['n_tasks']} tasks, {g['n_empty_blocks']} empty)"
        ),
        f"  transform:     {doc['transform']}",
        f"  saturation:    {g['saturation']:g}",
        f"  scale:         {g['scale']:g} (DN = percentile * scale), percentile-step: off",
        f"  seed:          {doc['seed']}",
        f"  workers:       {doc['workers']}, max-band-parallel: default (--workers)",
        f"  compress:      {g['compress']}",
        f"  outputs:       value.tif (EPSG:{g['out_crs']}) + support_km.tif",
        "  turbo:         off",
    ):
        assert line in text, line
    assert f"ingest:        {doc['ingest']['rows_read']} rows -> {doc['ingest']['points']} points" in text


def test_plan_json_stdout_clean_of_human_printer(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """--plan --json: no human plan lines leak into the JSON stream."""
    _, text = _plan_json(tmp_path, tmp_path / "out", capsys, "--json")
    assert "run plan" not in text
    assert "grid cells" not in text


# ---------------------------------------------------------------------------
# overwrite behaviour (mirrors the #54 guard)
# ---------------------------------------------------------------------------


def test_plan_json_overwrite_behaviour(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Track the overwrite status as the out dir fills up (create -> force)."""
    out = tmp_path / "out"
    doc, _ = _plan_json(tmp_path, out, capsys, "--json")
    assert doc["overwrite"] == "create"

    out.mkdir()
    doc, _ = _plan_json(tmp_path, out, capsys, "--json")
    assert doc["overwrite"] == "existing"

    (out / "value.tif").write_bytes(b"x")
    doc, _ = _plan_json(tmp_path, out, capsys, "--json")
    assert doc["overwrite"] == "guarded"

    doc, _ = _plan_json(tmp_path, out, capsys, "--json", "--force")
    assert doc["overwrite"] == "force"
    # an unrelated file does not trip the guard
    (out / "value.tif").unlink()
    (out / "notes.txt").write_text("x")
    doc, _ = _plan_json(tmp_path, out, capsys, "--json")
    assert doc["overwrite"] == "existing"


# ---------------------------------------------------------------------------
# run_summary.json: additive schema_version
# ---------------------------------------------------------------------------


def test_run_summary_gains_schema_version(tmp_path: Path) -> None:
    """Acceptance 4+5: run_summary.json parses; key set = main + schema_version."""
    out = tmp_path / "out"
    _pl.main(_argv(tmp_path, out, "--json"))
    s = json.loads((out / "run_summary.json").read_text())
    assert s["schema_version"] == 1
    assert set(s) == _RUN_SUMMARY_TOP_KEYS
    assert set(s["grid"]) == _GRID_KEYS  # same grid key set as the plan doc
    assert s["status"] == "ok"


def test_run_summary_schema_version_additive(tmp_path: Path) -> None:
    """The schema_version delta is additive: every pre-#105 key is intact."""
    out = tmp_path / "out"
    _pl.main(_argv(tmp_path, out, "--json", "--seed", "7"))
    s = json.loads((out / "run_summary.json").read_text())
    pre = {
        "status",
        "ppgrid_version",
        "seed",
        "transform",
        "cap_km",
        "grid",
        "ingest",
        "calibration",
        "timings_ms",
        "volumetrics",
        "turbo",
        "outputs",
        "cli",
        "git_commit",
    }
    assert pre <= set(s)
    assert s["seed"] == 7
    assert s["ingest"] == {"rows_read": 2, "points": 2, "dropped_nan_inf": 0}
