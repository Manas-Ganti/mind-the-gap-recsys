"""`recsys ingest | split | harness-check | embed | train | evaluate | serve | report | all | synth`."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_CONFIG = "configs/config.yaml"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="recsys", description=__doc__)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ingest", help="Spark: parse MIND into parquet + results/dataset_stats.json")
    sp = sub.add_parser("split", help="Spark: temporal / temporal_val / temporal_rtrain / random splits, negatives, audit")
    sp.add_argument("--only", nargs="*", choices=["temporal", "temporal_val", "temporal_rtrain", "random"])
    hc = sub.add_parser("harness-check", help="score dummy rankings (random/oracle) under both protocols")
    hc.add_argument("--split", default="temporal", choices=["temporal", "temporal_val", "temporal_rtrain", "random"])
    sub.add_parser("embed", help="encode item text once with the frozen sentence encoder (cached)")
    sub.add_parser("train", help="tune all retrievers on temporal_val, train rankers, refit on eval splits")
    sub.add_parser("evaluate", help="paired protocols, ceilings, end-to-end, popularity -> results/metrics.json")
    sub.add_parser("serve", help="FAISS ANN recall/latency over two-tower embeddings -> results/ann.json")
    sub.add_parser("report", help="results/report.md, figures/, gate verdicts in GATES.md")
    sub.add_parser("all", help="ingest, split, train, evaluate, serve, report")
    sy = sub.add_parser("synth", help="write tiny synthetic MIND-format files (tests / smoke runs only)")
    sy.add_argument("--out", required=True)
    sy.add_argument("--users", type=int, default=300)
    sy.add_argument("--items", type=int, default=600)
    a = p.parse_args(argv)

    if a.cmd == "synth":
        from recsys.synthetic import make_synthetic_mind
        print(make_synthetic_mind(Path(a.out), n_users=a.users, n_items=a.items))
        return 0

    from recsys.config import load_config
    cfg = load_config(a.config)
    cfg.ensure_dirs()
    steps = ["ingest", "split", "embed", "train", "evaluate", "serve", "report"] if a.cmd == "all" else [a.cmd]
    for step in steps:
        print(f"== {step} ==", flush=True)
        if step == "ingest":
            from recsys.data import ingest
            print(json.dumps({k: v for k, v in ingest(cfg).items() if k != "per_day"}, indent=2))
        elif step == "split":
            from recsys.splits import build_splits
            build_splits(cfg, only=getattr(a, "only", None))
        elif step == "harness-check":
            from recsys.eval.harness_check import format_table, run
            out = run(cfg, a.split)
            print(format_table(out, cfg.eval.headline_k))
            if not out["checks"]["all_pass"]:
                return 1
        elif step == "embed":
            from recsys.retrieval.text_embed import ensure_item_embeddings, embeddings_path
            print(ensure_item_embeddings(cfg))
            print(embeddings_path(cfg).with_suffix(".json").read_text())
        elif step == "train":
            from recsys.pipeline import train
            train(cfg)
        elif step == "evaluate":
            from recsys.pipeline import evaluate
            evaluate(cfg)
        elif step == "serve":
            # Separate process: faiss and torch both ship OpenMP; loading both can crash on macOS.
            import subprocess
            code = ("from recsys.config import load_config; from recsys.serve.ann import run; "
                    f"run(load_config({str(a.config)!r}))")
            r = subprocess.run([sys.executable, "-c", code])
            if r.returncode:
                return r.returncode
        elif step == "report":
            from recsys.report import report
            gates = report(cfg)
            print(json.dumps({g: v["pass"] for g, v in gates.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
