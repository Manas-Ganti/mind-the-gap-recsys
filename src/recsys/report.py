"""Turn results/*.json (written by real runs) into results/report.md, figures/, and the gate
verdicts in GATES.md. Reads numbers; never invents them. If an input is missing, the report
says so instead of filling a gap."""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from recsys.config import Config
from recsys.retrieval import METHOD_ORDER

RANKERS = ["gbdt", "neural"]
# Reference categorical palette (dataviz skill), fixed method -> slot so color follows the
# entity across every figure.
COLORS = {"popular": "#2a78d6", "itemknn": "#eb6834", "graph": "#1baf7a", "two_tower": "#eda100",
          "gbdt": "#e87ba4", "neural": "#008300"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def _load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


# --------------------------------------------------------------------------- gates

def evaluate_gates(cfg: Config, metrics: dict, tuning: dict | None) -> dict:
    k = cfg.eval.headline_k
    gates = {}
    sp = metrics.get("splits", {})

    # A
    detail, ok = {}, True
    if "temporal" in sp and "random" in sp:
        for m in METHOD_ORDER:
            t, r = sp["temporal"]["retrieval"][m]["full"]["user"], sp["random"]["retrieval"][m]["full"]["user"]
            passed = t[f"recall@{k}"] < r[f"recall@{k}"] and t[f"ndcg@{k}"] < r[f"ndcg@{k}"]
            detail[m] = {"temporal": {f"recall@{k}": t[f"recall@{k}"], f"ndcg@{k}": t[f"ndcg@{k}"]},
                         "random": {f"recall@{k}": r[f"recall@{k}"], f"ndcg@{k}": r[f"ndcg@{k}"]}, "pass": passed}
            ok &= passed
    else:
        ok, detail = False, {"error": "temporal and random splits are not both evaluated"}
    gates["A"] = {"pass": ok, "detail": detail}

    # B
    missing = []
    for split, s in sp.items():
        for m in METHOD_ORDER:
            r = s["retrieval"].get(m)
            if r is None or not np.isfinite(r["full"]["user"].get(f"ndcg@{k}", np.nan)) \
                    or not np.isfinite(r["sampled"]["per_positive"].get(f"hit@{k}", np.nan)):
                missing.append(f"{split}/{m}")
        if split == "temporal":
            for rk in RANKERS:
                e = s.get("end_to_end", {}).get(rk)
                if e is None or "sampled" not in e:
                    missing.append(f"{split}/{rk}")
    gates["B"] = {"pass": not missing and bool(sp), "missing": missing}

    # C
    if tuning and "itemknn" in tuning.get("methods", {}) and "two_tower" in tuning["methods"]:
        ic = cfg.retrieval.itemknn
        declared = len(ic.neighbors) * len(ic.shrink) * len(ic.alpha)
        knn, tt = tuning["methods"]["itemknn"], tuning["methods"]["two_tower"]
        ordered = knn["completed_at"] < tt["started_at"]
        gates["C"] = {"pass": knn["grid_size"] == declared and ordered, "declared_grid": declared,
                      "evaluated_grid": knn["grid_size"], "itemknn_completed_at": knn["completed_at"],
                      "two_tower_started_at": tt["started_at"]}
    else:
        gates["C"] = {"pass": False, "error": "results/tuning.json missing itemknn or two_tower"}

    # D
    problems = []
    for split, s in sp.items():
        ceil_t = [c["computed_at"] for c in s.get("ceiling", {}).values()]
        e2e_t = [e["computed_at"] for e in s.get("end_to_end", {}).values()]
        if not ceil_t:
            problems.append(f"{split}: no ceiling")
        if e2e_t and max(ceil_t) >= min(e2e_t):
            problems.append(f"{split}: an end-to-end number was computed before a ceiling")
        for m, c in s.get("ceiling", {}).items():
            for metric, e in c.get("end_to_end", {}).items():
                if e > c["ceiling"][metric] + 1e-9:
                    problems.append(f"{split}/{m}: end-to-end {metric} exceeds ceiling")
        for rk, e in s.get("end_to_end", {}).items():
            for metric, v in e["end_to_end"].items():
                if v > e["ceiling"][metric] + 1e-9:
                    problems.append(f"{split}/{rk}: end-to-end {metric} exceeds ceiling")
    gates["D"] = {"pass": not problems and bool(sp), "problems": problems}
    return gates


def write_gates_block(gates: dict, results_dir: Path) -> None:
    """Fill the verdict block of the GATES.md that sits next to `results_dir` (the repo's own
    for real runs; smoke runs have none, so the real file is never touched by synthetic data)."""
    path = Path(results_dir).parent / "GATES.md"
    if not path.exists():
        return
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    lines = [f"Evaluated {day} by `recsys report` from `results/metrics.json`.", "",
             "| Gate | Result | Notes |", "|---|---|---|"]
    notes = {
        "A": "; ".join(f"{m}: {'ok' if d['pass'] else 'FAIL'}" for m, d in gates["A"]["detail"].items()
                       if isinstance(d, dict)) or gates["A"]["detail"].get("error", ""),
        "B": "missing: " + ", ".join(gates["B"]["missing"]) if gates["B"]["missing"] else "all present",
        "C": (f"{gates['C'].get('evaluated_grid')}/{gates['C'].get('declared_grid')} grid points; "
              f"itemknn done {gates['C'].get('itemknn_completed_at')}, two-tower started "
              f"{gates['C'].get('two_tower_started_at')}") if "error" not in gates["C"] else gates["C"]["error"],
        "D": "; ".join(gates["D"]["problems"]) or "ceilings first; no end-to-end exceeds its ceiling",
    }
    for g in "ABCD":
        lines.append(f"| {g} | {'PASS' if gates[g]['pass'] else 'FAIL'} | {notes[g]} |")
    block = "\n".join(lines)
    text = path.read_text()
    text = re.sub(r"(<!-- GATES:BEGIN.*?-->\n).*?(\n<!-- GATES:END -->)", lambda m: m.group(1) + block + m.group(2),
                  text, flags=re.S)
    path.write_text(text)


# --------------------------------------------------------------------------- tables

def _f(x, d=4) -> str:
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"


def _pm(split: dict, method: str, key: str, value: float, d: int = 4) -> str:
    """value, plus '± sd' across seeds when the method is stochastic."""
    sd = split.get("seed_variation", {}).get("sd", {}).get(method, {}).get(key)
    return _f(value, d) + (f" ± {_f(sd, d)}" if sd is not None and len(split["seed_variation"]["seeds"]) > 1 else "")


def seed_section(split: dict, name: str, k: int) -> list[str]:
    sv = split.get("seed_variation")
    if not sv or len(sv["seeds"]) < 2:
        return []
    lines = [f"### {name}: variation across seeds {sv['seeds']}", "",
             "Two-tower and the rankers are trained once per seed. Tables above show mean ± sd; "
             "the baselines are deterministic, so they have no seed spread.", "",
             f"| model | " + " | ".join(f"seed {x}: full NDCG@{k} / sampled HR@{k}" for x in sv["seeds"]) + " |",
             "|---" * (1 + len(sv["seeds"])) + "|"]
    for m, rows in sv["per_seed"].items():
        lines.append(f"| {m} | " + " | ".join(f"{_f(r[f'full_user_ndcg@{k}'])} / {_f(r[f'sampled_hit@{k}'])}" for r in rows) + " |")
    lines += ["", "Method ordering per seed (retrieval, per-positive HR@%d, full vs sampled):" % k, ""]
    for x, o in zip(sv["seeds"], sv["orderings_per_seed"]):
        lines.append(f"- seed {x}: full {o['retrieval_full_hit_pp']} vs sampled {o['retrieval_sampled_hit_pp']}, "
                     f"τ = {o['kendall_tau']:.2f}" + (f"; system τ = {o['system']['kendall_tau']:.2f}" if "system" in o else ""))
    return lines + [""]


def paired_table(split: dict, k: int) -> str:
    rows = ["| method | full Recall@{k} | full NDCG@{k} | full HR@{k} (per positive) | sampled HR@{k} | "
            "full NDCG@{k} (pp) | sampled NDCG@{k} |".format(k=k), "|---" * 7 + "|"]
    for m in METHOD_ORDER:
        s = split["retrieval"][m]
        rows.append(f"| {m} | {_pm(split, m, f'full_user_recall@{k}', s['full']['user'][f'recall@{k}'])} | "
                    f"{_pm(split, m, f'full_user_ndcg@{k}', s['full']['user'][f'ndcg@{k}'])} | "
                    f"{_f(s['full']['per_positive'][f'hit@{k}'])} | {_pm(split, m, f'sampled_hit@{k}', s['sampled']['per_positive'][f'hit@{k}'])} | "
                    f"{_f(s['full']['per_positive'][f'ndcg@{k}'])} | {_f(s['sampled']['per_positive'][f'ndcg@{k}'])} |")
    for rk, e in split.get("end_to_end", {}).items():
        rows.append(f"| {rk} (end-to-end) | {_pm(split, rk, f'full_user_recall@{k}', e['end_to_end'][f'recall@{k}'])} | "
                    f"{_pm(split, rk, f'full_user_ndcg@{k}', e['end_to_end'][f'ndcg@{k}'])} | "
                    f"- | {_pm(split, rk, f'sampled_hit@{k}', e['sampled']['per_positive'][f'hit@{k}'])} | - | "
                    f"{_f(e['sampled']['per_positive'][f'ndcg@{k}'])} |")
    return "\n".join(rows)


def ceiling_table(split: dict, k: int) -> str:
    rows = [f"| pipeline | candidates | retrieval recall@K | ceiling Recall@{k} | end-to-end Recall@{k} | "
            f"fraction of ceiling | retrieval loss | ranking loss |", "|---" * 8 + "|"]
    for m in METHOD_ORDER:
        c = split["ceiling"][m]
        d = c["decomposition"][f"recall@{k}"]
        rows.append(f"| {m} (own order) | {round(c['candidate_set_size'])} | {_f(c['ceiling']['retrieval_recall'])} | "
                    f"{_f(d['ceiling'])} | {_f(d['end_to_end'])} | {_f(d['fraction_of_ceiling'], 3)} | "
                    f"{_f(d['retrieval_loss'])} | {_f(d['ranking_loss'])} |")
    for rk, e in split.get("end_to_end", {}).items():
        d = e["decomposition"][f"recall@{k}"]
        rows.append(f"| union -> {rk} | {round(e['candidate_set_size'])} | {_f(e['ceiling']['retrieval_recall'])} | "
                    f"{_f(d['ceiling'])} | {_f(d['end_to_end'])} | {_f(d['fraction_of_ceiling'], 3)} | "
                    f"{_f(d['retrieval_loss'])} | {_f(d['ranking_loss'])} |")
    return "\n".join(rows)


def popularity_table(split: dict, k: int) -> str:
    rows = [f"| method | HR@{k} | IPS HR@{k} | share of hits a popularity ranker also gets | "
            f"HR@{k} cold | HR@{k} tail | HR@{k} torso | HR@{k} head |", "|---" * 8 + "|"]
    for m in METHOD_ORDER:
        p = split["popularity"][m]
        b = p["by_bucket"]
        rows.append(f"| {m} | {_f(p[f'hit@{k}'])} | {_f(p[f'ips_hit@{k}'])} | {_f(p.get('share_of_hits_explained'), 3)} | "
                    + " | ".join(_f(b[x][f"hit@{k}"]) for x in ("cold", "tail", "torso", "head")) + " |")
    shares = split["popularity"][METHOD_ORDER[0]]["by_bucket"]
    rows.append("| *share of positives* | | | | " + " | ".join(_f(shares[x]["share_of_positives"], 3)
                                                         for x in ("cold", "tail", "torso", "head")) + " |")
    return "\n".join(rows)


def split_gap_table(metrics: dict, k: int) -> str:
    sp = metrics["splits"]
    rows = [f"| method | temporal NDCG@{k} | random NDCG@{k} | random / temporal | temporal Recall@{k} | random Recall@{k} |",
            "|---" * 6 + "|"]
    for m in METHOD_ORDER:
        t, r = sp["temporal"]["retrieval"][m]["full"]["user"], sp["random"]["retrieval"][m]["full"]["user"]
        ratio = r[f"ndcg@{k}"] / t[f"ndcg@{k}"] if t[f"ndcg@{k}"] > 0 else float("nan")
        rows.append(f"| {m} | {_f(t[f'ndcg@{k}'])} | {_f(r[f'ndcg@{k}'])} | {_f(ratio, 2)}x | "
                    f"{_f(t[f'recall@{k}'])} | {_f(r[f'recall@{k}'])} |")
    return "\n".join(rows)


# --------------------------------------------------------------------------- figures

def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fig_ordering(split: dict, k: int, path: Path) -> None:
    """Slopegraph: rank of each method under sampled HR@k (left) vs full-catalog NDCG@k (right)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    o = split["orderings"].get("system")
    left, right = (o["sampled_hit"], o["full_user_ndcg"]) if o else \
        (split["orderings"]["retrieval_sampled_hit_pp"], split["orderings"]["retrieval_full_hit_pp"])
    lv = o["sampled_values"] if o else {m: split["retrieval"][m]["sampled"]["per_positive"][f"hit@{k}"] for m in left}
    rv = o["full_values"] if o else {m: split["retrieval"][m]["full"]["per_positive"][f"hit@{k}"] for m in right}
    n = len(left)
    fig, ax = plt.subplots(figsize=(7.5, 0.55 * n + 1.8), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for m in left:
        y0, y1 = left.index(m), right.index(m)
        ax.plot([0, 1], [y0, y1], color=COLORS[m], linewidth=2, solid_capstyle="round")
        ax.scatter([0, 1], [y0, y1], s=64, color=COLORS[m], edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.text(-0.04, y0, f"{m}  {lv[m]:.3f}", ha="right", va="center", fontsize=9, color=INK)
        ax.text(1.04, y1, f"{rv[m]:.4f}  {m}", ha="left", va="center", fontsize=9, color=INK)
    ax.set_xlim(-0.75, 1.75)
    ax.set_ylim(n - 0.5, -0.9)
    ax.axis("off")
    ax.text(0, -0.75, f"Sampled HR@{k}", ha="center", fontsize=10, color=INK2, weight="bold")
    ax.text(1, -0.75, f"Full-catalog NDCG@{k}", ha="center", fontsize=10, color=INK2, weight="bold")
    tau = (o or split["orderings"])["kendall_tau"]
    fig.suptitle(f"Method ordering: sampled vs full-catalog (Kendall τ = {tau:.2f})", fontsize=11, color=INK, x=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def fig_split_gap(metrics: dict, k: int, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sp = metrics["splits"]
    ms = METHOD_ORDER
    t = [sp["temporal"]["retrieval"][m]["full"]["user"][f"ndcg@{k}"] for m in ms]
    r = [sp["random"]["retrieval"][m]["full"]["user"][f"ndcg@{k}"] for m in ms]
    y = np.arange(len(ms))
    fig, ax = plt.subplots(figsize=(7.5, 3.2), facecolor=SURFACE)
    _style(ax)
    h = 0.36
    ax.barh(y - h / 2 - 0.01, t, height=h, color="#2a78d6", label="temporal (honest)")
    ax.barh(y + h / 2 + 0.01, r, height=h, color="#eb6834", label="random (leaks future)")
    for yi, a, b in zip(y, t, r):
        ax.text(a, yi - h / 2, f" {a:.4f}", va="center", fontsize=8, color=INK2)
        ax.text(b, yi + h / 2, f" {b:.4f}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(y, ms)
    ax.invert_yaxis()
    ax.set_xlabel(f"Full-catalog NDCG@{k}", color=INK2, fontsize=9)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.set_title("Same models, same data: temporal vs random split", fontsize=11, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def fig_ceiling(split: dict, k: int, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [(f"{m}", split["ceiling"][m]["decomposition"][f"recall@{k}"]) for m in METHOD_ORDER]
    rows += [(f"union → {rk}", e["decomposition"][f"recall@{k}"]) for rk, e in split.get("end_to_end", {}).items()]
    labels = [r[0] for r in rows]
    e2e = np.array([r[1]["end_to_end"] for r in rows])
    rank_loss = np.array([r[1]["ranking_loss"] for r in rows])
    ret_loss = np.array([r[1]["retrieval_loss"] for r in rows])
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(7.5, 0.5 * len(rows) + 1.6), facecolor=SURFACE)
    _style(ax)
    gap = 0.004
    ax.barh(y, e2e, color="#2a78d6", height=0.6, label=f"achieved Recall@{k}")
    ax.barh(y, np.maximum(rank_loss - gap, 0), left=e2e + gap, color="#eb6834", height=0.6, label="lost to ranking")
    ax.barh(y, np.maximum(ret_loss - gap, 0), left=e2e + rank_loss + gap, color=GRID, height=0.6,
            hatch="////", edgecolor="#b5b3ad", linewidth=0, label="lost to retrieval (ceiling)")
    for yi, a in zip(y, e2e):
        ax.text(0.005, yi, f"{a:.3f}", va="center", fontsize=8, color="white" if a > 0.06 else INK)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel(f"Share of test positives (Recall@{k})", color=INK2, fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3)
    ax.set_title("Where recall is lost: retrieval ceiling vs ranking", fontsize=11, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


BUCKET_COLORS = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]   # ordinal blue ramp, validated


def fig_popularity(split: dict, path: Path) -> None:
    """100% stacked bars: share of top-k slots by item-popularity bucket, per method, with the
    distribution of actual clicks as the reference row."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from recsys.eval.popularity import BUCKETS

    conc = split["popularity_concentration"]
    rows = ["clicks"] + [m for m in METHOD_ORDER + RANKERS if m in conc]
    labels = ["actual clicks"] + rows[1:]
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(7.5, 0.5 * len(rows) + 1.8), facecolor=SURFACE)
    _style(ax)
    left = np.zeros(len(rows))
    gap = 0.004
    for j, bkt in enumerate(BUCKETS):
        vals = np.array([conc[r][bkt] for r in rows])
        ax.barh(y, np.maximum(vals - gap, 0), left=left, height=0.62, color=BUCKET_COLORS[j], label=bkt)
        for yi, l, v in zip(y, left, vals):
            if v >= 0.07:
                ax.text(l + v / 2, yi, f"{v:.0%}", ha="center", va="center", fontsize=8,
                        color=INK if j == 0 else "white")
        left += vals
    ax.axhline(0.5, color=INK2, linewidth=0.8)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel(f"Share of top-{conc['k']} recommendation slots (item popularity bucket)", color=INK2, fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=4)
    ax.set_title("Popularity concentration: what each method shows vs what users clicked",
                 fontsize=11, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def fig_ann(ann: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 3.6), facecolor=SURFACE)
    _style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    for kind, key, color in (("hnsw", "ef_search", "#2a78d6"), ("ivf", "nprobe", "#eb6834")):
        pts = ann[kind]
        x, y = [p["p95_ms"] for p in pts], [p["recall"] for p in pts]
        ax.plot(x, y, color=color, linewidth=2, marker="o", markersize=7, markeredgecolor=SURFACE,
                markeredgewidth=2, label=kind.upper())
        ax.text(x[-1], y[-1], f"  {kind.upper()}", color=INK, fontsize=9, va="center")
        for p in pts:
            ax.annotate(f"{key}={p[key]}", (p["p95_ms"], p["recall"]), fontsize=7, color=INK2,
                        xytext=(4, -10), textcoords="offset points")
    ex = ann["exact"]
    ax.scatter([ex["p95_ms"]], [1.0], s=64, color=INK2, zorder=3)
    ax.text(ex["p95_ms"], 1.0, "  exact", fontsize=9, color=INK, va="bottom")
    ax.set_xscale("log")
    ax.set_xlabel("p95 single-query latency (ms, log)", color=INK2, fontsize=9)
    ax.set_ylabel(f"Recall@{ann['k']} vs exact", color=INK2, fontsize=9)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.set_title(f"ANN over {ann['n_items']:,} item embeddings", fontsize=11, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


# --------------------------------------------------------------------------- entry point

def report(cfg: Config) -> dict:
    res, figs = Path(cfg.paths.results_dir), Path(cfg.paths.figures_dir)
    figs.mkdir(parents=True, exist_ok=True)
    metrics = _load(res / "metrics.json")
    if metrics is None:
        raise FileNotFoundError("results/metrics.json not found: run `recsys evaluate` first")
    tuning = _load(res / "tuning.json")
    audit = _load(res / "split_audit.json")
    stats = _load(res / "dataset_stats.json")
    ann = _load(res / "ann.json")
    k = cfg.eval.headline_k

    gates = evaluate_gates(cfg, metrics, tuning)
    (res / "gates.json").write_text(json.dumps(gates, indent=2))
    write_gates_block(gates, res)

    md = [f"# Results (generated {datetime.now(timezone.utc).isoformat()} from real runs)", "",
          f"Seed {metrics['meta']['seed']}, run date {metrics['meta']['date']}, "
          f"config {metrics['meta']['config_fingerprint']}.", ""]
    if stats:
        md += ["## Dataset", "", f"MIND-{stats['variant']}: {stats['users']:,} users, {stats['items']:,} items, "
               f"{stats['impressions']:,} impressions, {stats['clicks']:,} clicks, CTR {stats['ctr']:.4f}, "
               f"sparsity {stats['sparsity']:.6f}.", ""]
    if audit:
        md += ["## Split audit", "", "| split | test pairs | eval users | catalog | cold users | cold items | "
               "cold positives | cold catalog |", "|---" * 8 + "|"]
        for n, a in audit.items():
            m, au = a.get("meta") or {}, a["audit"]
            md.append(f"| {n} | {m.get('test_pairs', 'n/a')} | {m.get('eval_users', 'n/a')} | {m.get('catalog_size', 'n/a')} | "
                      f"{_f(au['cold_user_rate'], 3)} | {_f(au['cold_item_rate'], 3)} | "
                      f"{_f(au['cold_positive_rate'], 3)} | {_f(au['cold_catalog_rate'], 3)} |")
        md.append("")
    md += ["## Gates", "", "| gate | result |", "|---|---|"] + \
          [f"| {g} | {'PASS' if v['pass'] else 'FAIL'} |" for g, v in gates.items()] + [""]

    sp = metrics["splits"]
    for name, split in sp.items():
        md += [f"## {name}: paired protocols", "", paired_table(split, k), "",
               f"Orderings (retrieval, per-positive HR@{k}): full {split['orderings']['retrieval_full_hit_pp']} vs "
               f"sampled {split['orderings']['retrieval_sampled_hit_pp']}, Kendall τ = "
               f"{split['orderings']['kendall_tau']:.2f}", ""]
        if "system" in split["orderings"]:
            o = split["orderings"]["system"]
            md += [f"Orderings (system): full NDCG@{k} {o['full_user_ndcg']} vs sampled HR@{k} {o['sampled_hit']}, "
                   f"Kendall τ = {o['kendall_tau']:.2f}", ""]
        md += seed_section(split, name, k)
        md += [f"### {name}: retrieval ceiling", "", ceiling_table(split, k), "",
               f"### {name}: popularity decomposition", "", popularity_table(split, k), ""]
    if "temporal" in sp and "random" in sp:
        md += ["## Temporal vs random", "", split_gap_table(metrics, k), ""]
        fig_split_gap(metrics, k, figs / "temporal_vs_random.png")
    if "temporal" in sp:
        fig_ordering(sp["temporal"], k, figs / "method_ordering.png")
        fig_ceiling(sp["temporal"], k, figs / "ceiling.png")
        fig_popularity(sp["temporal"], figs / "popularity_concentration.png")
    if tuning:
        md += ["## Tuning (temporal_val)", ""] + [
            f"- **{m}**: {t['grid_size']} grid points, best {t['best']} -> {tuning['metric']} = {_f(t['best_value'])}"
            + (f" — **WARNING: best on a grid edge: {', '.join(t['at_grid_edge'])}**" if t.get("at_grid_edge") else "")
            for m, t in tuning["methods"].items()] + [""]
    if ann:
        md += ["## ANN serving", "", "| index | setting | recall | p50 ms | p95 ms | p99 ms |", "|---" * 6 + "|",
               f"| exact | - | 1.000 | {_f(ann['exact']['p50_ms'], 3)} | {_f(ann['exact']['p95_ms'], 3)} | "
               f"{_f(ann['exact']['p99_ms'], 3)} |"]
        for kind, key in (("hnsw", "ef_search"), ("ivf", "nprobe")):
            for p in ann[kind]:
                md.append(f"| {kind} | {key}={p[key]} | {_f(p['recall'], 3)} | {_f(p['p50_ms'], 3)} | "
                          f"{_f(p['p95_ms'], 3)} | {_f(p['p99_ms'], 3)} |")
        md.append("")
        fig_ann(ann, figs / "ann_tradeoff.png")
    (res / "report.md").write_text("\n".join(md))
    return gates
