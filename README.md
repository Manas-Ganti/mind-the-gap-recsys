# Mind the Gap: a Two-Stage Recommender with an Honest Evaluation Protocol

A retrieval-plus-ranking news recommender on MIND impression logs. **The deliverable is the
evaluation protocol.** The models are what gets measured.

## Why this exists

Most recommender work reports a number from a random split, scored against sampled
negatives, versus an untuned baseline. All three choices inflate the result, and the third
can invert the conclusion entirely. This project builds a working two-stage recommender and
then measures what each of those three shortcuts costs, on the same data, with the same
models.

> **Status: run end to end on MIND-small.** Metrics are in `results/mind_small/metrics.json`
> and the tables generated from them in `results/mind_small/report.md` (seed 42; two-tower and
> both rankers also trained at seeds 43 and 44). All four gates pass
> (`results/mind_small/gates.json`), and the harness checks (oracle, anti-oracle, random scorer
> against its closed form) report `all_pass: true` on both splits
> (`results/mind_small/harness_check_temporal.json`, `results/mind_small/harness_check_random.json`).
> Every number below is copied from `results/mind_small/report.md` unless another file is named.
> The outputs in `smoke/` come from tiny synthetic data; they are pipeline checks, not results.

## The rules

1. **No model is evaluated on an interaction it could not have seen at serving time.** The
   primary split is temporal. A random split is computed only as a comparison artifact, and
   the gap between the two is reported as a finding.
2. **A sampled-negative metric is never reported without the full-catalog number next to
   it.** The harness scores every method under both protocols in a single pass.
3. **Baselines are tuned before any neural model is compared to them.** Item-kNN gets a
   declared grid, evaluated in full, before two-tower training starts.
4. **Every gate is written down before the numbers are seen.** See [`GATES.md`](GATES.md).

## Architecture

```
MIND behaviors.tsv / news.tsv
  │
  ├─[1] INGEST (Spark)      parse impressions (shown + clicked), histories, news text → parquet
  ├─[2] SPLIT + AUDIT       temporal (primary) · temporal_val (tuning, ranker selection) · temporal_rtrain (ranker training) · random (comparison)
  │                         Spark-sampled negatives · leakage + cold-start audit
  ├─[3] EVALUATION HARNESS  full catalog vs sampled, paired · retrieval ceiling · popularity
  ├─[4] RETRIEVAL           popular (tuned window) · item-kNN (tuned) · RP3beta graph · two-tower
  ├─[5] RANKING             LightGBM lambdarank · MLP listwise, over the union of stage-1 candidates
  ├─[6] SERVING             FAISS HNSW / IVF vs exact: recall and p50/p95/p99 latency
  └─[7] REPORT              results/report.md · figures/ · gate verdicts in GATES.md
```

### Design decisions that matter for honesty

- **Availability is the same in every split.** An item can be ranked for a test click only
  if it was shown on that click's day. Each eval user is evaluated on one day's request. In
  the temporal split that is the single test day. In the random split it is a seeded choice
  among that user's test days. Without this, the random split would be ranked against a
  7-day catalog and would look *harder*, for reasons that have nothing to do with leakage.
  (The synthetic smoke run caught this; see `src/recsys/splits.py`.)
- **MIND's `history` is used as a pre-window snapshot, and the split proves that is safe.**
  On MIND-small, each of the 91,935 users with history has exactly one history version
  across all their impressions. This includes 5,943 users active both before and after the
  cutoff. Also, 42,467 of the 44,908 history items never appear in the 6-day log. So the
  history predates the window and never contains a test-day click. An earlier,
  over-conservative rule (use history only if it was attached to a pre-cutoff impression)
  made 88% of temporal test users look cold. The split now asserts the one-version property
  on every run and fails if it breaks (`split.history_policy`).
- **Ties are broken at random.** Scores get a seeded jitter shared by all methods. Without
  it, a popularity model that scores every fresh item 0 would be ranked by item index.
- **Negatives are paired.** They are drawn once in Spark with `xxhash64(u, pos, j, seed)`,
  so the draw is deterministic regardless of partitioning, and every method sees the same
  negatives.
- **Rankers only see stage-1 candidates.** That is how the system runs. A ranker's
  "sampled" number scores the positive against the sampled negatives, which bypasses
  retrieval. That is exactly what a sampled evaluation of a ranker implicitly does, and it
  is reported as such.
- **Rankers are selected on a later day, not on held-out users from the same day.** They
  train on 2019-11-13 candidates (retrievers fitted before that day), are early-stopped and
  pick one of three feature sets declared in `configs/` (`all`, `time_invariant`,
  `retrieval_only`) by their score on 2019-11-14, and are then refit on 2019-11-14 for the
  chosen number of rounds. A same-day user holdout cannot see day-to-day shift. On
  MIND-small it rated the GBDT far above what it scored on test
  (`results/mind_small/diagnostics/`).
- **Graph = RP3beta, not LightGCN.** It is the spec's "co-occurrence graph with
  propagation" option. It is a strong and cheap baseline, and choosing it keeps the budget
  on the evaluation.

## The four findings

Each is a question with a pre-stated measurement. Answers go here **only** from a real run.

1. **Temporal vs random split.** How much does a random split inflate each method? This
   also reports cold-start rates per split, which explain much of the gap.
   *MIND-small:* the random split inflates full-catalog NDCG@10 for every method: popular
   2.11x, item-kNN 4.54x, graph 7.44x, two-tower 36.42x (two-tower 0.0025 temporal vs 0.0923
   random). The cold-positive rate is 0.490 on the temporal split and 0.011 on the random split
   (split audit table; `results/mind_small/split_audit.json`).
2. **Sampled vs full-catalog ordering.** Does the ranking of methods change between the
   two protocols? Measured as Kendall τ between the orderings, over the retrieval methods
   and over the whole system including rankers.
   *MIND-small:* on the temporal split the retrieval ordering changes, Kendall τ = 0.00
   (full catalog: popular, graph, itemknn, two_tower; sampled: popular, two_tower, itemknn,
   graph). Per seed (42, 43, 44) it is τ = 0.00, 0.00, 0.33. Over the whole system
   τ = 0.47. On the random split the retrieval τ = 0.67.
3. **Tuned item-kNN vs neural.** Does the two-tower model beat a seriously tuned item-kNN
   on full-catalog metrics? If it doesn't, that is the result.
   *MIND-small:* it doesn't, on the temporal split. Item-kNN (120 grid points, all evaluated
   before two-tower training started; `results/mind_small/gates.json`, gate C) scores full
   Recall@10 0.0045 / NDCG@10 0.0027; two-tower scores 0.0044 ± 0.0005 / 0.0025 ± 0.0004
   (mean ± sd over seeds). Under sampled negatives the two-tower looks better (HR@10
   0.2702 ± 0.0268 vs 0.1523). On the random split the two-tower leads (NDCG@10 0.0923 vs
   0.0123).
4. **Retrieval ceiling.** Is the system limited by retrieval or by ranking? *MIND-small:*
   by retrieval. On the temporal split the union of candidates fed to the rankers has a
   Recall@10 ceiling of 0.4254, so retrieval loss is 0.5746; ranking loss is 0.3291 for the
   neural ranker and 0.3808 for the GBDT (see the table below).

## The retrieval ceiling

A ranker cannot recover an item retrieval never returned. For candidate set C and test
positives P with h = |C ∩ P|, the best any ranker can do is Recall@k = min(h, k)/|P|, with
NDCG@k for h hits packed at the top. Every pipeline is reported as:

```
1.0 ─┬─ retrieval loss = 1 − ceiling          (positive never retrieved)
     ├─ ranking loss   = ceiling − end-to-end  (retrieved, ranked too low)
     └─ end-to-end                              (what the user sees)
```

This is computed for each retriever's own ordering and for the union of candidates fed to
each ranker. Gate D requires every ceiling to be computed before any end-to-end number, and
no end-to-end number may exceed its ceiling. Tests check the bound by construction, with
random reorderings, and check that the ideal reordering attains it exactly.

*Results, MIND-small temporal split* (gate D passes, `results/mind_small/gates.json`):

| pipeline | candidates | retrieval recall@K | ceiling Recall@10 | end-to-end Recall@10 | fraction of ceiling | retrieval loss | ranking loss |
|---|---|---|---|---|---|---|---|
| popular (own order) | 200 | 0.2982 | 0.2982 | 0.0740 | 0.248 | 0.7018 | 0.2242 |
| itemknn (own order) | 200 | 0.0735 | 0.0735 | 0.0045 | 0.061 | 0.9265 | 0.0690 |
| graph (own order) | 200 | 0.0746 | 0.0746 | 0.0060 | 0.081 | 0.9254 | 0.0686 |
| two_tower (own order) | 200 | 0.1119 | 0.1119 | 0.0044 | 0.040 | 0.8881 | 0.1075 |
| union -> gbdt | 620 | 0.4254 | 0.4254 | 0.0446 | 0.104 | 0.5746 | 0.3808 |
| union -> neural | 620 | 0.4254 | 0.4254 | 0.0963 | 0.226 | 0.5746 | 0.3291 |

The random-split table is in `results/mind_small/report.md`.

## Popularity decomposition

For each method:

- the share of its hits that a pure popularity ranker also gets
- hit rate in the cold / tail / torso / head buckets
- IPS-weighted metrics with weight 1/(pop+1)^γ, where γ is a modelling choice set in the
  config

*Results, MIND-small temporal split:* 0.487 of test positives fall in the cold bucket.
Popular, item-kNN and graph all score HR@10 0.0000 on cold items; two-tower scores 0.0021. The
share of hits a popularity ranker also gets is 0.057 for item-kNN, 0.079 for graph and 0.041 for
two-tower. On the random split it is 0.600 for two-tower. The full tables, including the IPS
and tail / torso / head columns, are in `results/mind_small/report.md`.

## Running it

```bash
# needs Java 8/11/17 for Spark 3.5 (e.g. brew install openjdk@17; export JAVA_HOME=...)
uv venv && uv pip install -e ".[dev]"
pytest                                    # metric verification, protocol, ceiling, splits (Spark if a JVM is present)

# see data/download.md, then:
recsys ingest      # Spark → data/processed/ingest, results/dataset_stats.json
recsys split       # Spark → temporal / temporal_val / temporal_rtrain / random + negatives, results/split_audit.json
recsys train       # tune on temporal_val (baselines first), train rankers on temporal_rtrain, select + refit on temporal_val, refit retrievers on eval splits
recsys evaluate    # → results/metrics.json (seed + date recorded), results/raw/*.npz
recsys serve       # FAISS → results/ann.json
recsys report      # → results/report.md, figures/, gate verdicts in GATES.md
```

On VT ARC (one OWL CPU node), see [`docs/arc.md`](docs/arc.md):
`arc/setup_env.sh` once, then `arc/submit.sh ... -- --config configs/arc_mind_small.yaml ingest split harness-check`.

Pipeline smoke test on synthetic MIND-format data. It writes only to `smoke/`:

```bash
recsys synth --out smoke/raw && recsys --config configs/smoke.yaml all
```

## Layout

```
configs/config.yaml   split dates, k values, N negatives, every grid, seeds
src/recsys/
  data.py             Spark ingestion            splits.py   Spark splits + negatives + audit
  dataset.py          in-memory split view       pipeline.py tune → train → evaluate
  eval/               metrics · protocol · ceiling · popularity
  retrieval/          popular · itemknn · graph (RP3beta) · two_tower
  ranking/            features · gbdt (isolated LightGBM worker) · neural
  serve/ann.py        FAISS (isolated process)   report.py   tables, figures, gates
tests/                every metric vs hand-worked / sklearn references; leakage; ceiling bound
```

## Limitations

- **Popularity and all other retrieval state are frozen at the cutoff.** A production news
  system updates popularity within minutes, using test-period clicks that happened *before*
  each request. That would be legitimate, but it is not done here. This penalises the
  popularity baseline most, and it makes the absolute numbers pessimistic.
- **One day of test data.** The temporal test window is MIND's single dev day, so results
  carry day-specific effects. The bootstrap CIs cover user sampling, not day-to-day
  variation.
- **The test set is click-defined and exposure-biased.** Positives are clicks on items MSN's
  production system chose to show. The IPS view is a sensitivity analysis under an assumed
  propensity, not a debiased estimate.
- **Tuning budgets are unequal on purpose.** Baselines get larger grids than the two-tower
  model, which gets a temperature sweep. This biases *against* the neural model, which is
  the conservative direction for the claim being tested. If the neural model wins anyway,
  the win is understated rather than overstated.
- **The two-tower item text encoder is frozen** (all-MiniLM-L6-v2, per the project
  decisions in `CLAUDE.md`). Only a projection and the user tower are trained. A fine-tuned
  encoder would probably do better on news-specific semantics. Freezing it keeps the
  neural model cheap and leaves the comparison about retrieval design, not language-model
  fine-tuning.
- **The rankers see one day at a time.** They are selected on 2019-11-14 and refit on it,
  with candidates from retrievers fitted on less data than the test-time retrievers, so
  some train/serve feature skew remains. The selection day also comes from MIND's `train`
  file, while the test day is its `dev` file, which samples users differently. A
  next-day check inside `train` may therefore under-state the shift to test.
- **The ANN benchmark measures single-thread, in-process latency** on the local machine. It
  is a tradeoff curve, not a capacity number.
- **Engineering note:** on macOS arm64, torch and LightGBM each bundle libomp and segfault
  when both are loaded in one process. LightGBM and FAISS therefore run in isolated worker
  processes.
