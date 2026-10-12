"""Orchestration: tune -> fit -> evaluate. `cli.py` is a thin wrapper around these.

Order is enforced by construction:
  * every baseline (popular, itemknn, graph) finishes its grid before two-tower training
    starts (Gate C; timestamps recorded in results/tuning.json)
  * tuning and ranker training only ever touch temporal_val
  * ceilings are computed and timestamped before any end-to-end number (Gate D)
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from recsys.config import Config
from recsys.dataset import SplitData, load_split, subset_eval_users
from recsys.eval.ceiling import ceiling_per_user, ceiling_report
from recsys.eval.metrics import bootstrap_ci, kendall_tau, mean_metrics, single_positive_metrics, user_metrics_from_positions
from recsys.eval.popularity import BUCKETS, assign_buckets, bucket_thresholds, concentration, decompose
from recsys.eval.protocol import MethodRun, method_order, run_protocol
from recsys.ranking.features import CandidateBlock, FeatureBuilder, rerank
from recsys.retrieval import METHOD_ORDER, get_retriever_class
from recsys.retrieval.base import Retriever

EVAL_SPLITS = ("temporal", "random")
RANKERS = ("gbdt", "neural")


# --------------------------------------------------------------------------- helpers

def _now() -> float:
    return time.time()


def _iso(t: float) -> str:
    return datetime.fromtimestamp(t, timezone.utc).isoformat()


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_default))


def _protocol_kw(cfg: Config) -> dict:
    return dict(ks=cfg.eval.ks, n_negatives=cfg.eval.n_negatives, candidates_k=cfg.retrieval.candidates_k,
                user_batch=cfg.eval.user_batch, tie_noise=cfg.eval.tie_noise, seed=cfg.seed)


def two_tower_kwargs(cfg: Config) -> dict:
    from recsys.retrieval.text_embed import embeddings_path

    kw = cfg.retrieval.two_tower.model_dump()
    for k in ("temperature", "text_encoder", "encode_batch_size", "seeds"):
        kw.pop(k)
    kw["seed"] = cfg.seed
    kw["embeddings_path"] = str(embeddings_path(cfg))
    return kw


def make_retriever(name: str, params: dict, cfg: Config, seed: int | None = None) -> Retriever:
    cls = get_retriever_class(name)
    if name == "two_tower":
        kw = two_tower_kwargs(cfg)
        if seed is not None:
            kw["seed"] = seed
        return cls(**params, **kw)
    return cls(**params)


def seeds_of(cfg: Config) -> list[int]:
    return list(cfg.retrieval.two_tower.seeds) or [cfg.seed]


BASELINES = [m for m in METHOD_ORDER if m != "two_tower"]


def tt_name(seed: int) -> str:
    return f"two_tower_seed{seed}"


def model_path(cfg: Config, split: str, name: str) -> Path:
    return Path(cfg.paths.models_dir) / split / f"{name}.pkl"


def _sample_users(users: np.ndarray, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(users, min(n, len(users)), replace=False))


# --------------------------------------------------------------------------- tuning

# Boundaries that are natural limits of a parameter, not arbitrary grid ends. A best value on
# any OTHER boundary means the optimum may lie outside the grid, i.e. the method is not yet
# tuned seriously; it is flagged in tuning.json and the report.
NATURAL_BOUNDS = {
    "popular": {"window_hours": (None, "all-time")},
    "itemknn": {"shrink": (0.0, None), "alpha": (0.0, 1.0)},
    "graph": {"beta": (0.0, None)},
}


def grid_edges(method: str, best: dict, grid: list[dict]) -> list[str]:
    """Parameters whose best value sits on a non-natural edge of the evaluated grid."""
    flagged = []
    natural = NATURAL_BOUNDS.get(method, {})
    for key, val in best.items():
        values = {g["params"][key] for g in grid}
        numeric = sorted(v for v in values if v is not None)
        if len(values) < 2 or val is None:
            continue
        lo_nat, hi_nat = natural.get(key, (None, None))
        hi = None if None in values else numeric[-1]      # None (all-time) is the top of the range
        if val == numeric[0] and val != lo_nat:
            flagged.append(f"{key}={val} (lowest tried)")
        elif hi is not None and val == hi and val != hi_nat:
            flagged.append(f"{key}={val} (highest tried)")
    return flagged


def tune(cfg: Config, val: SplitData) -> tuple[dict, dict[str, Retriever]]:
    """Grid-search every retriever on temporal_val. Baselines first, neural last."""
    from recsys.retrieval import graph, itemknn
    from recsys.retrieval.popular import PopularRetriever

    sub = subset_eval_users(val, _sample_users(val.eval_users, cfg.retrieval.tuning_users, cfg.seed))
    metric, k = cfg.retrieval.tuning_metric, cfg.eval.headline_k
    kw = _protocol_kw(cfg)
    path = Path(cfg.paths.results_dir) / "tuning.json"
    out = {"split": "temporal_val", "metric": f"full-catalog user {metric}", "n_users": len(sub.eval_users),
           "seed": cfg.seed, "methods": {}}
    best: dict[str, Retriever] = {}

    def search(name: str, grid_iter, started: float) -> None:
        rows, best_val, best_r = [], -np.inf, None
        for params, r in grid_iter:
            s = run_protocol(sub, {name: r.score}, **kw)[name].summary(cfg.eval.ks)
            v = s["full"]["user"][metric]
            rows.append({"params": params, metric: v, f"sampled_hit@{k}": s["sampled"]["per_positive"][f"hit@{k}"]})
            print(f"  [tune] {name} {params} {metric}={v:.5f}", flush=True)
            if v > best_val:
                best_val, best_r = v, r
        best[name] = best_r
        best_params = rows[int(np.argmax([x[metric] for x in rows]))]["params"]
        edges = grid_edges(name, best_params, rows)
        if edges:
            print(f"  [tune] WARNING {name}: best is on a grid edge: {edges}. Widen the grid in configs/.", flush=True)
        out["methods"][name] = {"grid_size": len(rows), "grid": rows, "best": best_params,
                                "best_value": best_val, "at_grid_edge": edges,
                                "started_at": _iso(started), "completed_at": _iso(_now())}
        write_json(path, out)

    pc, ic, gc, tc = cfg.retrieval.popular, cfg.retrieval.itemknn, cfg.retrieval.graph, cfg.retrieval.two_tower
    t = _now()
    search("popular", (({"window_hours": w}, PopularRetriever(w).fit(val)) for w in pc.window_hours), t)
    t = _now()
    search("itemknn", ((r.params, r) for r in itemknn.fit_grid(val, ic.neighbors, ic.shrink, ic.alpha)), t)
    t = _now()
    search("graph", ((r.params, r) for r in graph.fit_grid(val, gc.neighbors, gc.alpha, gc.beta)), t)
    t = _now()
    rows, best_mean, best_models = [], -np.inf, None
    for tau in tc.temperature:
        per, models = {}, {}
        for sd in seeds_of(cfg):
            r = make_retriever("two_tower", {"temperature": tau}, cfg, seed=sd).fit(val)
            per[sd] = run_protocol(sub, {"m": r.score}, **kw)["m"].summary(cfg.eval.ks)["full"]["user"][metric]
            models[sd] = r
            print(f"  [tune] two_tower {{'temperature': {tau}}} seed={sd} {metric}={per[sd]:.5f}", flush=True)
        vals = np.array(list(per.values()))
        rows.append({"params": {"temperature": tau}, metric: float(vals.mean()), "sd": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                     "per_seed": {str(k_): v for k_, v in per.items()}})
        print(f"  [tune] two_tower {{'temperature': {tau}}} mean {metric}={vals.mean():.5f} sd={rows[-1]['sd']:.5f}", flush=True)
        if vals.mean() > best_mean:
            best_mean, best_models = float(vals.mean()), models
    best["two_tower"] = best_models                     # {seed: fitted model}
    best_params = rows[int(np.argmax([x[metric] for x in rows]))]["params"]
    edges = grid_edges("two_tower", best_params, rows)
    if edges:
        print(f"  [tune] WARNING two_tower: best is on a grid edge: {edges}. Widen the grid in configs/.", flush=True)
    out["methods"]["two_tower"] = {"grid_size": len(rows), "grid": rows, "best": best_params, "best_value": best_mean,
                                   "seeds": seeds_of(cfg), "selection": "mean over seeds", "at_grid_edge": edges,
                                   "started_at": _iso(t), "completed_at": _iso(_now())}
    write_json(path, out)
    return out, best


# --------------------------------------------------------------------------- training

def _collect_blocks(data: SplitData, retrievers: dict[str, Retriever], cfg: Config, *, with_labels: bool,
                    sampled: bool) -> tuple[dict[str, MethodRun], CandidateBlock, CandidateBlock | None, FeatureBuilder]:
    fb = FeatureBuilder(data, METHOD_ORDER)
    blocks, sblocks = [], []

    def hook(users, scores, cand_cols, sampled=None, **_):
        blocks.append(fb.build(users, scores, cand_cols, with_labels=with_labels))
        if sampled is not None and do_sampled:
            sblocks.append(fb.build_sampled(users, scores, cand_cols, *sampled))

    do_sampled = sampled
    runs = run_protocol(data, {m: retrievers[m].score for m in METHOD_ORDER}, on_batch=hook, **_protocol_kw(cfg))
    return runs, CandidateBlock.concat(blocks), (CandidateBlock.concat(sblocks) if sblocks else None), fb


def _with_seed(retr: dict, two_tower) -> dict:
    return {**{m: retr[m] for m in BASELINES}, "two_tower": two_tower}


def train(cfg: Config) -> dict:
    from recsys.ranking.gbdt import GBDTRanker
    from recsys.ranking.neural import NeuralRanker

    from recsys.retrieval.text_embed import ensure_item_embeddings

    cfg.ensure_dirs()
    ensure_item_embeddings(cfg)            # no-op when the `embed` stage already cached them
    val = load_split(cfg, "temporal_val")
    tuning, best_val = tune(cfg, val)

    # Stage-2 training data: candidates from the tuned val retrievers on held-in val users, once
    # per seed (each seed's two-tower produces different candidates and score features).
    rk = cfg.ranking
    sub = subset_eval_users(val, _sample_users(val.eval_users, rk.max_train_users, cfg.seed + 1))
    tuning["rankers"] = {"seeds": seeds_of(cfg), "per_seed": {}}
    nc = rk.neural
    for sd in seeds_of(cfg):
        print(f"  [rankers] seed {sd}: building candidate features", flush=True)
        _, block, _, fb = _collect_blocks(sub, _with_seed(best_val, best_val["two_tower"][sd]), cfg,
                                          with_labels=True, sampled=False)
        has_pos = np.add.reduceat(block.y, block.ptr[:-1]) > 0 if len(block.y) else np.zeros(0, bool)
        is_valid = np.random.default_rng(cfg.seed + 2).random(len(block.users)) < rk.valid_fraction
        tr, va = block.subset_groups(~is_valid & has_pos), block.subset_groups(is_valid & has_pos)
        info = {"n_groups": int(len(block.users)), "groups_with_positive_in_candidates": float(has_pos.mean()),
                "n_train_groups": int(len(tr.users)), "n_valid_groups": int(len(va.users)),
                "mean_candidates": float(np.diff(block.ptr).mean()), "features": fb.names}
        gbdt = GBDTRanker(rk.gbdt.params, rk.gbdt.num_boost_round, rk.gbdt.early_stopping_rounds, sd).fit(tr, va, fb.names)
        gbdt.save(model_path(cfg, "rankers", f"gbdt_seed{sd}"))
        info["gbdt"] = {"best_iteration": gbdt.best_iteration_, "feature_importance_gain": gbdt.feature_importance_}
        neural = NeuralRanker(nc.hidden, nc.dropout, nc.lr, nc.epochs, nc.batch_users, nc.patience, nc.device,
                              sd, nc.threads).fit(tr, va, fb.names)
        neural.save(model_path(cfg, "rankers", f"neural_seed{sd}"))
        info["neural"] = {"valid_ndcg@10_curve": neural.valid_curve_, "best_valid_ndcg@10": neural.best_valid_ndcg_,
                          "epochs_run": len(neural.valid_curve_)}
        tuning["rankers"]["per_seed"][str(sd)] = info
        write_json(Path(cfg.paths.results_dir) / "tuning.json", tuning)
        del block, tr, va
    del val, best_val, sub

    # Final retrievers: best params, refit on each evaluation split's train set. Baselines once,
    # two-tower once per seed.
    best_params = {m: tuning["methods"][m]["best"] for m in METHOD_ORDER}
    for split in EVAL_SPLITS:
        data = load_split(cfg, split)
        for m in BASELINES:
            print(f"  [fit] {split}/{m} {best_params[m]}", flush=True)
            make_retriever(m, best_params[m], cfg).fit(data).save(model_path(cfg, split, m))
        for i, sd in enumerate(seeds_of(cfg)):
            print(f"  [fit] {split}/two_tower seed={sd} {best_params['two_tower']}", flush=True)
            r = make_retriever("two_tower", best_params["two_tower"], cfg, seed=sd).fit(data)
            r.save(model_path(cfg, split, tt_name(sd)))
            if split == "temporal" and i == 0:
                emb = Path(cfg.paths.models_dir) / "embeddings"
                emb.mkdir(parents=True, exist_ok=True)
                np.save(emb / "items.npy", r.item_vecs_)
                q = _sample_users(data.eval_users, cfg.serve.n_queries, cfg.seed + 3)
                np.save(emb / "queries.npy", r.user_vectors(q))
    return tuning


# --------------------------------------------------------------------------- evaluation

def _ci(values: np.ndarray, cfg: Config, salt: int) -> list[float]:
    return list(bootstrap_ci(values, cfg.eval.bootstrap, cfg.seed + salt))


def _sampled_ranks(scores: np.ndarray, block: CandidateBlock, seed: int, tie_noise: float) -> np.ndarray:
    """Rank of each group's first row (the positive) among the group's other rows."""
    s = scores.astype(np.float64)
    s = s + np.random.default_rng(seed).random(len(s)) * tie_noise * (1.0 + np.abs(s))
    ranks = np.empty(len(block.users), dtype=np.int64)
    for g in range(len(block.users)):
        a, b = block.ptr[g], block.ptr[g + 1]
        ranks[g] = 1 + int((s[a + 1:b] > s[a]).sum())
    return ranks


def _evaluate_split(cfg: Config, data: SplitData, retr: dict[str, Retriever], rankers: dict | None,
                    raw_dir: Path, raw_names: dict[str, str]) -> dict:
    """Everything for one split with one set of models (one seed of the stochastic ones)."""
    ks, k = cfg.eval.ks, cfg.eval.headline_k
    with_rankers = rankers is not None
    runs, block, sblock, _ = _collect_blocks(data, retr, cfg, with_labels=False, sampled=with_rankers)
    for m, run in runs.items():
        if m in raw_names:
            run.save(raw_dir / f"{raw_names[m]}.npz")

    out = {"n_eval_users": int(len(data.eval_users)), "catalog_size": int(len(data.catalog)),
           "candidates_k": cfg.retrieval.candidates_k, "retrieval": {}}
    for m, run in runs.items():
        s = run.summary(ks)
        user = user_metrics_from_positions(run.position, run.user_ptr, [k])
        samp = single_positive_metrics(run.sampled_rank, [k])
        s["ci95"] = {f"full_user_recall@{k}": _ci(user[f"recall@{k}"], cfg, 1),
                     f"full_user_ndcg@{k}": _ci(user[f"ndcg@{k}"], cfg, 2),
                     f"sampled_hit@{k}": _ci(samp[f"hit@{k}"], cfg, 3)}
        s["params"] = retr[m].params
        out["retrieval"][m] = s

    # Gate D: ceilings first.
    out["ceiling"] = {}
    for m, run in runs.items():
        out["ceiling"][m] = {**ceiling_report(run.candidates, None, data.eval_users, data.test_pos, ks),
                             "computed_at": _iso(_now())}
    if with_rankers:
        union = rerank(block, np.zeros(len(block.items)))
        out["ceiling"]["union"] = {"ceiling": mean_metrics(ceiling_per_user(union, block.users, data.test_pos, ks)),
                                   "candidate_set_size": float(np.diff(block.ptr).mean()),
                                   "computed_at": _iso(_now())}
        out["end_to_end"] = {}
        for name, rk_model in rankers.items():
            ranked = rerank(block, rk_model.predict(block))
            rep = ceiling_report(union, ranked, block.users, data.test_pos, ks)
            sranks = _sampled_ranks(rk_model.predict(sblock), sblock, cfg.seed + 7, cfg.eval.tie_noise)
            rep["sampled"] = {"per_positive": mean_metrics(single_positive_metrics(sranks, ks)),
                              "n_pairs": int(len(sranks))}
            rep["computed_at"] = _iso(_now())
            out["end_to_end"][name] = rep

    pop = np.bincount(data.train["i"].to_numpy(), minlength=data.n_items)
    base = runs["popular"].full_rank
    out["popularity"] = {m: decompose(run.full_rank, pop[run.pos_item], k, cfg.eval.popularity_gamma,
                                      baseline_ranks=None if m == "popular" else base)
                         for m, run in runs.items()}
    all_pos_pop = pop[runs["popular"].pos_item]
    q1, q2 = bucket_thresholds(all_pos_pop)
    b = assign_buckets(all_pos_pop, q1, q2)
    conc = {"thresholds": [q1, q2], "k": k,
            "clicks": {lab: float((b == j).mean()) for j, lab in enumerate(BUCKETS)}}
    for m, run in runs.items():
        conc[m] = concentration(run.candidates[:, :k], pop, q1, q2)
    if with_rankers:
        for name, rk_model in rankers.items():
            conc[name] = concentration(rerank(block, rk_model.predict(block), max_len=k), pop, q1, q2)
    out["popularity_concentration"] = conc
    out["orderings"] = _orderings(out, k, with_rankers)
    return out


def _orderings(out: dict, k: int, with_rankers: bool) -> dict:
    summ = out["retrieval"]
    o_full = method_order(summ, "full", f"hit@{k}")
    o_samp = method_order(summ, "sampled", f"hit@{k}")
    res = {"retrieval_full_hit_pp": o_full, "retrieval_sampled_hit_pp": o_samp,
           "kendall_tau": kendall_tau(o_full, o_samp), "metric": f"hit@{k} per positive"}
    if with_rankers:
        full_sys = {m: summ[m]["full"]["user"][f"ndcg@{k}"] for m in summ}
        full_sys.update({r: out["end_to_end"][r]["end_to_end"][f"ndcg@{k}"] for r in RANKERS})
        samp_sys = {m: summ[m]["sampled"]["per_positive"][f"hit@{k}"] for m in summ}
        samp_sys.update({r: out["end_to_end"][r]["sampled"]["per_positive"][f"hit@{k}"] for r in RANKERS})
        a = sorted(full_sys, key=lambda x: -full_sys[x])
        b = sorted(samp_sys, key=lambda x: -samp_sys[x])
        res["system"] = {"full_user_ndcg": a, "sampled_hit": b, "kendall_tau": kendall_tau(a, b),
                         "full_values": full_sys, "sampled_values": samp_sys}
    return res


def _deep_stat(objs: list, fn):
    """Apply fn elementwise across structurally identical results; non-numeric leaves come from
    the first seed (they are identical, e.g. params, or ordering-related and recomputed)."""
    first = objs[0]
    if isinstance(first, dict):
        return {key: _deep_stat([o[key] for o in objs], fn) for key in first if all(key in o for o in objs)}
    if isinstance(first, (int, float)) and not isinstance(first, bool) and all(isinstance(o, (int, float)) for o in objs):
        return float(fn(np.asarray(objs, dtype=float)))
    if isinstance(first, list) and first and all(isinstance(x, (int, float)) for x in first) \
            and all(isinstance(o, list) and len(o) == len(first) for o in objs):
        return [float(fn(np.asarray([o[i] for o in objs], dtype=float))) for i in range(len(first))]
    return first


def _aggregate_seeds(outs: list[dict], seeds: list[int], k: int, with_rankers: bool) -> dict:
    """Mean over seeds (identical for the deterministic baselines), sd for the stochastic models,
    orderings recomputed from the means and also reported per seed."""
    agg = _deep_stat(outs, np.mean)
    agg["orderings"] = _orderings(agg, k, with_rankers)
    sd = (lambda a: a.std(ddof=1)) if len(outs) > 1 else (lambda a: 0.0)
    head = {}
    for o in outs:
        for m in ("two_tower",):
            r = o["retrieval"][m]
            head.setdefault(m, []).append({f"full_user_ndcg@{k}": r["full"]["user"][f"ndcg@{k}"],
                                           f"full_user_recall@{k}": r["full"]["user"][f"recall@{k}"],
                                           f"sampled_hit@{k}": r["sampled"]["per_positive"][f"hit@{k}"]})
        for m in (RANKERS if with_rankers else ()):
            e = o["end_to_end"][m]
            head.setdefault(m, []).append({f"full_user_ndcg@{k}": e["end_to_end"][f"ndcg@{k}"],
                                           f"full_user_recall@{k}": e["end_to_end"][f"recall@{k}"],
                                           f"sampled_hit@{k}": e["sampled"]["per_positive"][f"hit@{k}"]})
    agg["seed_variation"] = {
        "seeds": seeds,
        "sd": {m: _deep_stat(v, sd) for m, v in head.items()},
        "per_seed": {m: v for m, v in head.items()},
        "orderings_per_seed": [o["orderings"] for o in outs],
        "kendall_tau_per_seed": [o["orderings"]["kendall_tau"] for o in outs],
    }
    if with_rankers:
        agg["seed_variation"]["system_kendall_tau_per_seed"] = [o["orderings"]["system"]["kendall_tau"] for o in outs]
    return agg


def evaluate(cfg: Config) -> dict:
    from recsys.ranking.gbdt import GBDTRanker
    from recsys.ranking.neural import NeuralRanker

    k = cfg.eval.headline_k
    res = Path(cfg.paths.results_dir)
    tuning_path = res / "tuning.json"
    seeds = seeds_of(cfg)
    metrics = {"meta": {"seed": cfg.seed, "stochastic_seeds": seeds, "date": _iso(_now()),
                        "config_fingerprint": cfg.fingerprint(), "config": cfg.model_dump(mode="json"),
                        "headline_k": k, "n_negatives": cfg.eval.n_negatives},
               "tuning": json.loads(tuning_path.read_text()) if tuning_path.exists() else None,
               "splits": {}}

    for split in EVAL_SPLITS:
        data = load_split(cfg, split)
        base = {m: Retriever.load(model_path(cfg, split, m)).attach(data) for m in BASELINES}
        with_rankers = split == "temporal"
        outs = []
        for i, sd in enumerate(seeds):
            print(f"[evaluate] {split} seed {sd}", flush=True)
            retr = _with_seed(base, Retriever.load(model_path(cfg, split, tt_name(sd))).attach(data))
            rankers = ({"gbdt": GBDTRanker.load(model_path(cfg, "rankers", f"gbdt_seed{sd}")),
                        "neural": NeuralRanker.load(model_path(cfg, "rankers", f"neural_seed{sd}"))}
                       if with_rankers else None)
            raw_names = {"two_tower": tt_name(sd), **({m: m for m in BASELINES} if i == 0 else {})}
            outs.append(_evaluate_split(cfg, data, retr, rankers, res / "raw" / split, raw_names))
        metrics["splits"][split] = _aggregate_seeds(outs, seeds, k, with_rankers)
        write_json(res / "metrics.json", metrics)
    return metrics
