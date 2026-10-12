# Running on VT ARC

The whole pipeline runs on **one OWL CPU node** (AMD Genoa, 96 cores). No GPU is needed:
Spark, the item-kNN and graph grids, evaluation and LightGBM are CPU/RAM work, and the
item tower uses frozen MiniLM embeddings (CLAUDE.md decision 1). If embedding the corpus
on CPU turns out slow, that one step can move to a short GPU job. Measure first.

Lessons are inherited from earlier ARC projects. Each rule below exists because ignoring
it once wasted an allocation.

## One-time setup (OWL login node, `owl1.arc.vt.edu`)

```bash
bind 'set enable-bracketed-paste off'          # VS Code terminals corrupt pastes otherwise
git clone <repo-url> ~/ondemand/data/mind-the-gap-recsys && cd ~/ondemand/data/mind-the-gap-recsys
bash arc/setup_env.sh                          # conda env at ~/miniconda3/envs/recsys (python 3.12, openjdk 17, CPU torch)
sacctmgr show assoc user=$USER format=account%30,qos%80   # which allocation is live?
quota                                          # ~20 GB needed for MIND-large + outputs
```

Data is never in git (MIND licence). Copy it from the laptop, then let the job verify it
against the checked-in manifest:

```bash
# on the laptop
rsync -av data/raw/MINDsmall_train data/raw/MINDsmall_dev <pid>@owl1.arc.vt.edu:~/ondemand/data/mind-the-gap-recsys/data/raw/
```

## Submitting

Always from the repo root on an OWL login node, **after `git pull --ff-only`**. The
`.slurm` file is snapshotted at submit time, but Python is read at run time.

```bash
ACCOUNT=<alloc> MAIL=<pid>@vt.edu arc/submit.sh --mem 64G --time 02:00:00 -- --config configs/arc_mind_small.yaml ingest split harness-check
```

- `DRY_RUN=1` prints the `sbatch` line without submitting.
- QOS defaults to `owl_normal_base` (7-day cap, 1× billing). `QOS=owl_normal_short`
  gives priority but has a 1-day cap and 2× billing.
- `--mem` is required, and `0` is refused: `--mem=0` asks for a whole idle node and
  never backfills.
- Settings travel as arguments to `pipeline.slurm`, never as `VAR=x sbatch`. Unknown
  variables are dropped silently, while unknown arguments fail loudly.

## What the job does before spending the allocation

1. Calls the env's Python by absolute path and checks that it prints a version. This
   catches the 0-byte-interpreter failure, where a job "completes" in 2 s.
2. Asserts the interpreter is inside `$CONDA_ENV` and every required import exists
   (`arc_require`).
3. Checks Java from the env (`$CONDA_ENV/lib/jvm`). It needs no ARC module.
4. Prints the config fingerprint, Spark settings, stages and git commit. It warns on
   uncommitted changes.
5. For `ingest`, runs `sha256sum -c data/MANIFEST-mind<variant>.sha256`.

## Order of runs

1. **MIND-small**:
   `ingest split harness-check`
   `train evaluate serve report` (once models are approved)
   Then read `sacct -j <id> --format=JobID,State,Elapsed,MaxRSS,AllocCPUS` to size step 2.
2. **MIND-large**: `configs/arc_mind_large.yaml`. Set its `spark.driver_memory` and the
   job's `--mem` from step 1's MaxRSS. Add `data/MANIFEST-mindlarge.sha256` first.
3. Copy back: `rsync -av <pid>@owl1.arc.vt.edu:~/ondemand/data/mind-the-gap-recsys/{results,figures} .`

## Watching

```bash
squeue -u $USER -o "%.10i %.14j %.9T %.11M %.11L %.22R %N"
tail -f logs/slurm/recsys-<jobid>.out
seff <jobid>
```

Telegram notifications work automatically if `~/.config/vrr/secrets.env` sets
`TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`. Otherwise they are a no-op.

## Unverified on ARC (confirm on first run, then record in the call centre)

- conda-forge `openjdk=17` running Spark on a compute node
- `SPARK_LOCAL_DIRS=$TMPDIR` having enough space for MIND-large shuffles
- OWL Genoa node memory (`scontrol show node <node> | grep RealMemory`)
- `/home` being shared between OWL and Tinkercliffs
