"""Why does GBDT end-to-end score below popularity? Seed 42, MIND-small, laptop.

Builds the ranker training block exactly as `train` does (val retrievers, val users) and the
test block exactly as `evaluate` does (retrievers refit on temporal train), then reports:
  1. NDCG@10 within the union candidates, per group with a positive, for: popular order,
     GBDT, neural - on held-out val groups AND on test groups.
  2. Feature drift val -> test (medians, KS).
  3. GBDT ablations retrained on the same val block.
"""
import json
import sys
import time

import numpy as np
from scipy.stats import ks_2samp

from recsys.config import load_config
from recsys.dataset import load_split, subset_eval_users
from recsys.eval.metrics import ndcg_at_k
from recsys.pipeline import (_collect_blocks, _sample_users, make_retriever, BASELINES)
from recsys.ranking.gbdt import GBDTRanker
from recsys.ranking.neural import NeuralRanker

cfg = load_config("configs/mind_small.yaml")
SEED = 42
tun = json.load(open("results/mind_small/tuning.json"))
best = {m: tun["methods"][m]["best"] for m in ["popular", "itemknn", "graph", "two_tower"]}
print("best params", best, flush=True)


def fit_all(data):
    r = {m: make_retriever(m, best[m], cfg).fit(data) for m in BASELINES}
    t = time.time()
    r["two_tower"] = make_retriever("two_tower", best["two_tower"], cfg, seed=SEED).fit(data)
    print(f"  two_tower fit {time.time() - t:.0f}s", flush=True)
    return r


def group_ndcg(block, scores, k=10):
    out = []
    for g in range(len(block.users)):
        a, b = block.ptr[g], block.ptr[g + 1]
        y = block.y[a:b]
        if not y.any():
            continue
        order = np.argsort(-scores[a:b], kind="stable")
        out.append(ndcg_at_k(list(order), set(np.flatnonzero(y)), k))
    return float(np.mean(out)), len(out)


# ---- val (ranker training data), as in train()
val = load_split(cfg, "temporal_val")
rv = fit_all(val)
rk = cfg.ranking
sub = subset_eval_users(val, _sample_users(val.eval_users, rk.max_train_users, cfg.seed + 1))
_, vblock, _, fb = _collect_blocks(sub, rv, cfg, with_labels=True, sampled=False)
has_pos = np.add.reduceat(vblock.y, vblock.ptr[:-1]) > 0
is_valid = np.random.default_rng(cfg.seed + 2).random(len(vblock.users)) < rk.valid_fraction
tr, va = vblock.subset_groups(~is_valid & has_pos), vblock.subset_groups(is_valid & has_pos)
del val, sub, rv

# ---- test, as in evaluate(), but with labels so we can score within-candidate NDCG
test = load_split(cfg, "temporal")
rt = fit_all(test)
_, tblock, _, _ = _collect_blocks(test, rt, cfg, with_labels=True, sampled=False)
names = fb.names
ip = names.index("score_popular")


def report(tag, model_scores):
    for split, blk in (("val-heldout", va), ("test", tblock)):
        v, n = group_ndcg(blk, model_scores(blk))
        print(f"  {tag:<34s} {split:<12s} NDCG@10 within union = {v:.4f}  (groups={n})", flush=True)


print("\n== 1. within-union NDCG@10 ==", flush=True)
report("popular score order", lambda b: b.X[:, ip])
g = GBDTRanker(rk.gbdt.params, rk.gbdt.num_boost_round, rk.gbdt.early_stopping_rounds, SEED).fit(tr, va, names)
print(f"  gbdt best_iteration={g.best_iteration_}", flush=True)
report("gbdt (as shipped)", g.predict)
nc = rk.neural
n = NeuralRanker(nc.hidden, nc.dropout, nc.lr, nc.epochs, nc.batch_users, nc.patience, nc.device, SEED, nc.threads).fit(tr, va, names)
report("neural (as shipped)", n.predict)

print("\n== 2. feature drift val(train groups) -> test ==", flush=True)
rows = []
for j, f in enumerate(names):
    a, b = tr.X[:, j], tblock.X[:, j]
    ks = ks_2samp(a[::7], b[::7]).statistic
    rows.append((ks, f, np.median(a), np.median(b), np.percentile(a, 99), np.percentile(b, 99)))
for ks, f, ma, mb, pa, pb in sorted(rows, reverse=True):
    print(f"  {f:<24s} KS={ks:.3f}  median {ma:10.4g} -> {mb:10.4g}   p99 {pa:10.4g} -> {pb:10.4g}", flush=True)

print("\n== 3. GBDT ablations ==", flush=True)


def ablate(tag, keep):
    idx = [i for i, f in enumerate(names) if keep(f)]
    from recsys.ranking.features import CandidateBlock as CB
    sl = lambda b: CB(b.users, b.ptr, b.items, np.ascontiguousarray(b.X[:, idx]), b.y)
    m = GBDTRanker(rk.gbdt.params, rk.gbdt.num_boost_round, rk.gbdt.early_stopping_rounds, SEED).fit(
        sl(tr), sl(va), [names[i] for i in idx])
    report(f"gbdt {tag} (it={m.best_iteration_})", lambda b: m.predict(sl(b)))


ablate("no item_age_h", lambda f: f != "item_age_h")
ablate("no age/ctr/recent", lambda f: not (f in ("item_age_h", "item_ctr") or f.startswith("item_recent")))
ablate("no retrieval scores", lambda f: not f.startswith("score_"))
ablate("only retrieval scores+in_top", lambda f: f.startswith("score_") or f.startswith("in_top_"))
