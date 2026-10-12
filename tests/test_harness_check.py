"""The dummy-ranking checkpoint on a tiny split: closed forms and exact oracle behaviour."""
import numpy as np
import pytest

from conftest import make_split
from recsys.eval.harness_check import checks, dummy_scorers, expected_random
from recsys.eval.metrics import mean_metrics, single_positive_metrics
from recsys.eval.protocol import run_protocol


def test_expected_random_closed_form_by_enumeration():
    for n in (0, 3, 9, 50):
        ranks = np.arange(1, n + 2)                       # uniform over 1..n+1
        exact = mean_metrics(single_positive_metrics(ranks, [10]))
        e = expected_random(np.array([n]), 10)
        for m in ("hit@10", "ndcg@10", "mrr"):
            assert e[m] == pytest.approx(exact[m])


def test_dummies_on_tiny_split():
    rng = np.random.default_rng(0)
    n_users, n_items = 300, 60
    test = [(u, int(i)) for u in range(n_users) for i in rng.choice(n_items, 2, replace=False)]
    pos = {u: {i for uu, i in test if uu == u} for u in range(n_users)}
    negs = {(u, i): [int(j) for j in rng.permutation([j for j in range(n_items) if j not in pos[u]])[:20]]
            for u, i in test}
    d = make_split([], test, n_users, n_items, negatives=negs)
    runs = run_protocol(d, dummy_scorers(d, 0), ks=[1, 10], n_negatives=20, candidates_k=10,
                        user_batch=64, tie_noise=1e-9, seed=0)
    out = {"methods": {}}
    for name, r in runs.items():
        s = r.summary([1, 10])
        if name == "random":
            s["expected"] = {"full": {**expected_random(r.full_n, 1), **expected_random(r.full_n, 10)},
                             "sampled": {**expected_random(r.sampled_n, 1), **expected_random(r.sampled_n, 10)}}
        out["methods"][name] = s
    c = checks(runs, out, [1, 10])
    assert c["all_pass"], c
    # sampled inflates a random ranking: ~10/21 vs ~10/59
    assert out["methods"]["random"]["sampled"]["per_positive"]["hit@10"] > \
        2 * out["methods"]["random"]["full"]["per_positive"]["hit@10"]


def test_config_extends_inherits_protocol_and_overrides_machine():
    from pathlib import Path
    from recsys.config import load_config

    root = Path(__file__).resolve().parents[1] / "configs"
    base, arc = load_config(root / "mind_small.yaml"), load_config(root / "arc_mind_small.yaml")
    assert arc.spark.master == "local[96]" and arc.spark.driver_memory == "48g"
    assert arc.split == base.split and arc.eval == base.eval and arc.retrieval == base.retrieval
    assert arc.paths == base.paths and arc.seed == base.seed


def test_grid_edge_detection():
    from recsys.pipeline import grid_edges

    grid = [{"params": {"neighbors": n, "shrink": s, "alpha": a}} for n in (50, 500) for s in (0.0, 10.0) for a in (0.3, 0.7)]
    # neighbors=500 and alpha=0.7 are arbitrary top edges; shrink=0 is a natural floor
    assert grid_edges("itemknn", {"neighbors": 500, "shrink": 0.0, "alpha": 0.7}, grid) == \
        ["neighbors=500 (highest tried)", "alpha=0.7 (highest tried)"]
    full = grid + [{"params": {"neighbors": 50, "shrink": 0.0, "alpha": 1.0}}]
    # alpha=1.0 is the natural top of [0, 1], so only neighbors and shrink are flagged
    assert grid_edges("itemknn", {"neighbors": 50, "shrink": 10.0, "alpha": 1.0}, full) == \
        ["neighbors=50 (lowest tried)", "shrink=10.0 (highest tried)"]
    w = [{"params": {"window_hours": h}} for h in (3, 6, None)]
    assert grid_edges("popular", {"window_hours": 3}, w) == ["window_hours=3 (lowest tried)"]
    assert grid_edges("popular", {"window_hours": None}, w) == []        # all-time is the natural top
    assert grid_edges("popular", {"window_hours": 6}, w) == []


def test_seed_aggregation_means_numbers_keeps_first_for_the_rest():
    import numpy as np
    from recsys.pipeline import _deep_stat

    a = {"m": {"x": 1.0, "ci": [0.0, 2.0], "params": {"t": 0.1}, "order": ["a", "b"], "n": 10}}
    b = {"m": {"x": 3.0, "ci": [2.0, 4.0], "params": {"t": 0.1}, "order": ["b", "a"], "n": 10}}
    mean = _deep_stat([a, b], np.mean)
    assert mean["m"]["x"] == 2.0 and mean["m"]["ci"] == [1.0, 3.0] and mean["m"]["n"] == 10.0
    assert mean["m"]["params"] == {"t": 0.1} and mean["m"]["order"] == ["a", "b"]
    sd = _deep_stat([a, b], lambda v: v.std(ddof=1))
    assert abs(sd["m"]["x"] - np.std([1.0, 3.0], ddof=1)) < 1e-12
