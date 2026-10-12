#!/usr/bin/env bash
# Run recsys stages on one VT ARC CPU node, in order, stopping at the first failure.
# Submit through arc/submit.sh (it supplies account, partition, QOS, --mem and mail).
#
#   arc/pipeline.slurm --config configs/arc_mind_small.yaml ingest split harness-check
#
# Stages: ingest split harness-check embed train evaluate serve report
#SBATCH --job-name=recsys
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80
# No --mem / --account / --partition / --mail-user here on purpose: #SBATCH lines cannot
# read env vars, and a wrong hardcoded value silently wins. submit.sh passes them.

set -euo pipefail
# SLURM runs a spool copy of this file, so resolve the repo from the submit dir.
source "${SLURM_SUBMIT_DIR:-$PWD}/arc/arc_env.sh"
arc_require recsys pyspark pandas pyarrow scipy sklearn lightgbm torch faiss matplotlib yaml sentence_transformers

CONFIG=""
STAGES=()
while [ $# -gt 0 ]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    ingest|split|harness-check|embed|train|evaluate|serve|report) STAGES+=("$1"); shift ;;
    *) echo "[pipeline] FATAL: unknown argument '$1'" >&2; exit 2 ;;
  esac
done
[ -n "$CONFIG" ] && [ -f "$CONFIG" ] || { echo "[pipeline] FATAL: --config <existing file> required" >&2; exit 2; }
[ ${#STAGES[@]} -gt 0 ] || { echo "[pipeline] FATAL: no stages given" >&2; exit 2; }

scontrol update jobid="${SLURM_JOB_ID:-0}" name="recsys-${STAGES[0]}" >/dev/null 2>&1 || true
trap arc_notify_finish EXIT
arc_notify "START recsys ${STAGES[*]} ${SLURM_JOB_ID:-?} on ${SLURM_JOB_NODELIST:-?} ($CONFIG)"

# Echo every effective knob before spending the allocation.
"$PY" -c "
from recsys.config import load_config
c = load_config('$CONFIG')
print('[pipeline] config=$CONFIG fingerprint=' + c.fingerprint())
print('[pipeline] variant=%s raw=%s results=%s' % (c.data.variant, c.paths.raw_dir, c.paths.results_dir))
print('[pipeline] spark master=%s driver_memory=%s' % (c.spark.master, c.spark.driver_memory))
print('[pipeline] stages: ${STAGES[*]}')
"

# Data must match the checked-in manifest before ingest reads it.
if printf '%s\n' "${STAGES[@]}" | grep -qx ingest; then
  variant="$("$PY" -c "from recsys.config import load_config; print(load_config('$CONFIG').data.variant)")"
  manifest="data/MANIFEST-mind${variant}.sha256"
  [ -f "$manifest" ] || { echo "[pipeline] FATAL: $manifest missing" >&2; exit 3; }
  sha256sum -c --quiet "$manifest" || { echo "[pipeline] FATAL: data does not match $manifest" >&2; exit 3; }
  echo "[pipeline] data verified against $manifest"
fi

for stage in "${STAGES[@]}"; do
  echo "[pipeline] ==== $stage  $(date -u '+%F %T')Z"
  start=$SECONDS
  "$PY" -m recsys.cli --config "$CONFIG" "$stage"
  echo "[pipeline] ==== $stage done in $((SECONDS - start)) s"
done
echo "[pipeline] all stages OK. Resource use: sacct -j ${SLURM_JOB_ID:-?} --format=JobID,State,Elapsed,MaxRSS,AllocCPUS"
