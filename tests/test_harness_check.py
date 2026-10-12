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
