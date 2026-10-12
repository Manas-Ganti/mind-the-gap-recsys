# Mind the Gap: a Two-Stage Recommender with an Honest Evaluation Protocol

A retrieval-plus-ranking news recommender on MIND impression logs. **The deliverable is the
evaluation protocol.** The models are what gets measured.

## Why this exists

Most recommender work reports a number from a random split, scored against sampled
negatives, versus an untuned baseline. All three choices inflate the result, and the third
can invert the conclusion entirely. This project builds a working two-stage recommender and
then measures what each of those three shortcuts costs, on the same data, with the same
models.

> **Status: code-complete, not yet run on MIND.** Every results section below is empty on
> purpose. Numbers will appear only after a real run produces `results/metrics.json`. The
> pipeline has been exercised end to end only on tiny synthetic data, and those outputs live
> in `smoke/`. They are pipeline checks, not results.

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
  ├─[2] SPLIT + AUDIT       temporal (primary) · temporal_val (tuning) · random (comparison)
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
- **Graph = RP3beta, not LightGCN.** It is the spec's "co-occurrence graph with
  propagation" option. It is a strong and cheap baseline, and choosing it keeps the budget
  on the evaluation.

## The four findings

Each is a question with a pre-stated measurement. Answers go here **only** from a real run.

1. **Temporal vs random split.** How much does a random split inflate each method? This
   also reports cold-start rates per split, which explain much of the gap.
   *Not yet run.*
2. **Sampled vs full-catalog ordering.** Does the ranking of methods change between the
   two protocols? Measured as Kendall τ between the orderings, over the retrieval methods
   and over the whole system including rankers.
   *Not yet run.*
3. **Tuned item-kNN vs neural.** Does the two-tower model beat a seriously tuned item-kNN
   on full-catalog metrics? If it doesn't, that is the result.
   *Not yet run.*
4. **Retrieval ceiling.** Is the system limited by retrieval or by ranking? *Not yet run.*

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

*Results: not yet run.*

## Popularity decomposition

For each method:

- the share of its hits that a pure popularity ranker also gets
- hit rate in the cold / tail / torso / head buckets
- IPS-weighted metrics with weight 1/(pop+1)^γ, where γ is a modelling choice set in the
  config

*Results: not yet run.*

## Running it

```bash
# needs Java 8/11/17 for Spark 3.5 (e.g. brew install openjdk@17; export JAVA_HOME=...)
uv venv && uv pip install -e ".[dev]"
pytest                                    # metric verification, protocol, ceiling, splits (Spark if a JVM is present)

# see data/download.md, then:
recsys ingest      # Spark → data/processed/ingest, results/dataset_stats.json
recsys split       # Spark → temporal / temporal_val / random + negatives, results/split_audit.json
recsys train       # tune on temporal_val (baselines first), train rankers, refit on eval splits
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
- **The two-tower text encoder is trained from scratch** (bag of words + category). A
  pretrained text encoder would likely help cold items, but it would add a dependency
  outside the agreed set.
- **The rankers are trained on one validation day** of candidates from retrievers fitted on
  less data than the test-time retrievers, so there is a small train/serve feature skew.
- **The ANN benchmark measures single-thread, in-process latency** on the local machine. It
  is a tradeoff curve, not a capacity number.
- **Engineering note:** on macOS arm64, torch and LightGBM each bundle libomp and segfault
  when both are loaded in one process. LightGBM and FAISS therefore run in isolated worker
  processes.
