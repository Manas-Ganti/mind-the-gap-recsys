"""Build-order checkpoint (CLAUDE.md decision 6): run the harness on real split data with
dummy rankings whose correct scores are known in closed form, before any model exists.

  random       i.i.d. uniform scores. A positive competing with n negatives lands at a
               uniform rank in 1..n+1, so
                 E[HR@k]   = min(k, n+1) / (n+1)
                 E[NDCG@k] = sum_{r=1..min(k,n+1)} 1/log2(r+1) / (n+1)
                 E[MRR]    = H_{n+1} / (n+1)
               evaluated with each positive's own n (full: rankable catalog items;
               sampled: drawn negatives). Observed must match within sampling error.
  oracle       reads the test set (a deliberate cheat): every positive outscores everything.
               Must score exactly 1.0 on every metric under both protocols.
  anti_oracle  positives score lowest. HR@k must be exactly 0 wherever n >= k.

The gap between random-under-sampled and random-under-full is the protocol's built-in
inflation before any model is involved: ~k/(N+1) vs ~k/|catalog|.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from recsys.eval.metrics import mean_metrics, single_positive_metrics


def expected_random(n: np.ndarray, k: int) -> dict[str, float]:
    """Closed-form expectations for a uniformly random ranking, averaged over positives."""
    n = np.asarray(n, dtype=np.int64)
    m = n + 1
    disc = 1.0 / np.log2(np.arange(2, max(k, 1) + 2))
    cum = np.concatenate([[0.0], np.cumsum(disc)])
    harm = np.concatenate([[0.0], np.cumsum(1.0 / np.arange(1, int(m.max()) + 1))]) if len(m) else np.zeros(1)
    return {f"hit@{k}": float(np.mean(np.minimum(k, m) / m)),
            f"ndcg@{k}": float(np.mean(cum[np.minimum(k, m)] / m)),
            "mrr": float(np.mean(harm[m] / m))}


def dummy_scorers(data, seed: int) -> dict:
    C = len(data.catalog)
    rng = np.random.default_rng(seed)
    P = data.test_pos.tocsr()
    ccol = data.catalog_col

    def random(users):
        return rng.random((len(users), C)).astype(np.float32)

    def _pos_mask(users, value):
        S = np.zeros((len(users), C), np.float32)
        Pb = P[users]
        rows = np.repeat(np.arange(len(users)), np.diff(Pb.indptr))
        S[rows, ccol[Pb.indices]] = value
        return S

    return {"random": random,
            "oracle": lambda users: _pos_mask(users, 1.0),
            "anti_oracle": lambda users: _pos_mask(users, -1.0)}


def run(cfg, split: str = "temporal") -> dict:
    from recsys.dataset import load_split
    from recsys.eval.protocol import run_protocol

    data = load_split(cfg, split)
    ks = cfg.eval.ks
    runs = run_protocol(data, dummy_scorers(data, cfg.seed), ks=ks, n_negatives=cfg.eval.n_negatives,
                        candidates_k=cfg.retrieval.candidates_k, user_batch=cfg.eval.user_batch,
                        tie_noise=cfg.eval.tie_noise, seed=cfg.seed)
    out = {"split": split, "date": datetime.now(timezone.utc).isoformat(), "seed": cfg.seed,
           "variant": cfg.data.variant, "n_eval_users": int(len(data.eval_users)),
           "catalog_union": int(len(data.catalog)), "n_negatives": cfg.eval.n_negatives, "methods": {}}
    for name, r in runs.items():
        s = r.summary(ks)
        s["full_candidates_per_positive"] = {"mean": float(r.full_n.mean()), "min": int(r.full_n.min()),
                                             "max": int(r.full_n.max())}
        if name == "random":
            s["expected"] = {"full": {}, "sampled": {}}
            for k in ks:
                s["expected"]["full"].update(expected_random(r.full_n, k))
                s["expected"]["sampled"].update(expected_random(r.sampled_n, k))
        out["methods"][name] = s
    out["checks"] = checks(runs, out, ks)
    path = Path(cfg.paths.results_dir) / "harness_check.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    return out


def checks(runs: dict, out: dict, ks: list[int]) -> dict:
    res = {}
    o = out["methods"]["oracle"]
    res["oracle_perfect_full"] = all(v == 1.0 for v in o["full"]["per_positive"].values()) and \
        all(o["full"]["user"][f"ndcg@{k}"] == 1.0 for k in ks)
    res["oracle_perfect_sampled"] = all(v == 1.0 for v in o["sampled"]["per_positive"].values())
    a = runs["anti_oracle"]
    for proto, ranks, n in (("full", a.full_rank, a.full_n), ("sampled", a.sampled_rank, a.sampled_n)):
        res[f"anti_oracle_last_{proto}"] = bool(np.all(ranks == n + 1))
    r = out["methods"]["random"]
    for proto, n_units in (("full", len(runs["random"].full_rank)), ("sampled", len(runs["random"].sampled_rank))):
        obs = r[proto]["per_positive"]
        exp = r["expected"][proto]
        z = {}
        for m in exp:
            # binomial-style standard error with a floor; observed must be within 4 SE
            p = exp[m]
            se = max(np.sqrt(max(p * (1 - p), 1e-12) / max(n_units, 1)), 1e-4)
            z[m] = (obs[m] - p) / se
        res[f"random_matches_closed_form_{proto}"] = bool(all(abs(v) < 4 for v in z.values()))
        res[f"random_z_{proto}"] = {m: round(v, 2) for m, v in z.items()}
    res["all_pass"] = all(v for kk, v in res.items() if isinstance(v, bool))
    return res


def format_table(out: dict, k: int) -> str:
    lines = [f"split={out['split']}  users={out['n_eval_users']}  catalog(union)={out['catalog_union']}  "
             f"N={out['n_negatives']}",
             f"{'scorer':<12} {'protocol':<9} {'HR@'+str(k):>9} {'NDCG@'+str(k):>9} {'MRR':>9}   "
             f"{'expected HR / NDCG / MRR':>28}"]
    for name, s in out["methods"].items():
        for proto in ("full", "sampled"):
            pp = s[proto]["per_positive"]
            exp = s.get("expected", {}).get(proto)
            e = f"{exp[f'hit@{k}']:.4f} / {exp[f'ndcg@{k}']:.4f} / {exp['mrr']:.4f}" if exp else ""
            lines.append(f"{name:<12} {proto:<9} {pp[f'hit@{k}']:>9.4f} {pp[f'ndcg@{k}']:>9.4f} {pp['mrr']:>9.4f}   {e:>28}")
    lines.append("checks: " + json.dumps({kk: v for kk, v in out["checks"].items() if isinstance(v, bool)}))
    return "\n".join(lines)
