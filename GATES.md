# Pre-stated gates

Written **2026-10-12**, before any data was downloaded or any model was run. The gates are
fixed; only the "Result" column below gets filled in, by `recsys report`, after real runs.
If a gate fails, the failure is reported as-is. A gate is never reworded after the numbers
are seen.

## Gate A: temporal < random

For **every retrieval method** (popular, itemknn, graph/RP3beta, two_tower), full-catalog
user-level **Recall@10 and NDCG@10 are strictly lower on the temporal split than on the
random split**.

If this fails, the split logic is assumed to be wrong. It is not treated as a finding.

Scope: the two rankers are evaluated on the temporal split only. They are trained on a
time-ordered validation window that the random split cannot provide, so Gate A does not
cover them.

## Gate B: both protocols, no exceptions

Every method has numbers under both the full-catalog and the sampled-negative
(N = `eval.n_negatives`, default 100) protocol in `results/metrics.json`. Baselines are
included. "Every method" means every retrieval method on every evaluated split, plus both
rankers on the temporal split. For a ranker:

- **full** means end-to-end over the stage-1 candidates. A positive that retrieval missed
  counts as a miss.
- **sampled** means the ranker scores the positive against the same N sampled negatives.
  This bypasses retrieval, which is exactly what a sampled evaluation of a ranker
  implicitly does.

## Gate C: item-kNN tuned before any neural comparison

Item-kNN is tuned over the grid declared in `configs/config.yaml`
(`retrieval.itemknn`: neighbors × shrink × alpha) on `temporal_val`, and every grid point
is evaluated. The tuning finishes (`completed_at` in `results/tuning.json`) before
two-tower training starts (`started_at`). The popularity window and the RP3beta grid are
tuned in the same way.

## Gate D: ceiling before end-to-end

On every split, the retrieval recall ceiling for each candidate source is computed and
written before any end-to-end number is reported (`computed_at` ordering in
`results/metrics.json`). As a sanity check, no end-to-end metric may exceed its ceiling.

---

## Result

<!-- GATES:BEGIN (written by `recsys report`; do not edit by hand) -->
Not run yet. No gate has been evaluated.
<!-- GATES:END -->
