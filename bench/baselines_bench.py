"""Config-driven baseline scale bench (issue #107).

Reads bench/config/baselines.json (workload: source points + query count;
limits: wall + RSS; the claimed-regime confirmation flag), builds a seeded
synthetic workload at that scale, runs the independent IDW + nearest
baselines, and reports wall/RSS against the configured limits. Limits are
reported, not enforced: a missed limit prints `limits met: false` and lands
in the JSON report (exit 0 either way).

Usage:
    uv run python bench/baselines_bench.py
    uv run python bench/baselines_bench.py --n-source 100000 --n-queries 10000 --tag sanity
    uv run python bench/baselines_bench.py --config bench/config/baselines.json --json /tmp/b.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ppgrid.baselines import idw, load_config, nearest, synthetic_workload

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "bench" / "config" / "baselines.json"


def main() -> None:
    ap = argparse.ArgumentParser(description="Config-driven IDW + nearest baseline bench (issue #107)")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG), help="baseline config JSON (default: the committed one)")
    ap.add_argument("--n-source", type=int, default=None, help="override the config's n_source")
    ap.add_argument("--n-queries", type=int, default=None, help="override the config's n_queries")
    ap.add_argument("--seed", type=int, default=0, help="workload RNG seed")
    ap.add_argument("--tag", default="regime", help="label for the report")
    ap.add_argument("--json", default=None, help="write the JSON report to this path")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.n_source is not None:
        cfg = dataclasses.replace(cfg, n_source=args.n_source)
    if args.n_queries is not None:
        cfg = dataclasses.replace(cfg, n_queries=args.n_queries)
    if cfg.needs_andryo_confirmation:
        print(f"[warn] claimed regime, not yet confirmed (needs_andryo_confirmation=true, config: {cfg.source})")
    print(f"config: {json.dumps(cfg.to_dict(), sort_keys=True)}")

    t0 = time.perf_counter()
    points, values, query = synthetic_workload(cfg.n_source, cfg.n_queries, seed=args.seed)
    t_build = time.perf_counter() - t0
    print(f"workload: {len(points)} points, {len(query)} queries built in {t_build:.1f}s")

    report = {"tag": args.tag, "config": cfg.to_dict(), "t_build_workload_s": round(t_build, 2), "results": []}
    for kind, run in (
        (
            "idw",
            lambda: idw(points, values, query, power=cfg.power, k=cfg.k, min_neighbours=cfg.min_neighbours, limits=cfg),
        ),
        ("nearest", lambda: nearest(points, query, values, limits=cfg)),
    ):
        r = run()
        report["results"].append(r.to_dict())
        lim = r.limits or {}
        wall_lim = f"{lim['wall_limit_s']:g}s" if "wall_limit_s" in lim else "-"
        rss_lim = f"{lim['rss_limit_gb']:g}GiB" if "rss_limit_gb" in lim else "-"
        met = "met" if r.wall_ok in (True, None) and r.rss_ok in (True, None) else "NOT MET"
        print(
            f"{kind}: wall {r.wall_s:.1f}s (limit {wall_lim}, ok={r.wall_ok}) "
            f"peakRSS {r.peak_rss_mib:.0f}MiB (limit {rss_lim}, ok={r.rss_ok}) [{met}]"
        )
        print(f"  {r.methods()}")

    if args.json:
        out = Path(args.json)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"report -> {out}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
