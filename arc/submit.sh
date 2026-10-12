#!/usr/bin/env bash
# Submit arc/pipeline.slurm to the OWL CPU cluster with the known-good flags.
# Run from the repo root on an OWL login node (owl1.arc.vt.edu).
#
#   ACCOUNT=<alloc> MAIL=<pid>@vt.edu arc/submit.sh --mem 64G --time 02:00:00 -- \
#       --config configs/arc_mind_small.yaml ingest split harness-check
#
# DRY_RUN=1 prints the sbatch line without submitting.
# QOS: owl_normal_base (7 d, 1x billing) by default; QOS=owl_normal_short for priority
# (1 d cap, 2x billing).
set -euo pipefail

MEM="" ; TIME="04:00:00"
while [ $# -gt 0 ]; do
  case "$1" in
    --mem) MEM="$2"; shift 2 ;;
    --time) TIME="$2"; shift 2 ;;
    --) shift; break ;;
    *) echo "usage: ACCOUNT=.. MAIL=.. $0 --mem 64G [--time HH:MM:SS] -- <pipeline args>" >&2; exit 2 ;;
  esac
done
: "${ACCOUNT:?set ACCOUNT=<slurm account> (check: sacctmgr show assoc user=\$USER format=account%30,qos%80)}"
: "${MAIL:?set MAIL=<pid>@vt.edu}"
[ -n "$MEM" ] || { echo "[submit] --mem is required (never --mem=0: it blocks backfill)" >&2; exit 2; }
[ "$MEM" != "0" ] || { echo "[submit] refusing --mem 0" >&2; exit 2; }
[ $# -gt 0 ] || { echo "[submit] no pipeline args after --" >&2; exit 2; }

cd "$(git rev-parse --show-toplevel)"
if [ -n "$(git status --porcelain -- src configs arc)" ]; then
  echo "[submit] WARNING: uncommitted changes in src/ configs/ arc/ (Python is read at RUN time)" >&2
fi
if git fetch -q 2>/dev/null && upstream="$(git rev-parse -q --verify '@{u}' 2>/dev/null)" \
   && [ "$(git rev-parse HEAD)" != "$upstream" ]; then
  echo "[submit] WARNING: checkout differs from its upstream; run git pull --ff-only first?" >&2
fi
mkdir -p logs/slurm

cmd=(sbatch --account="$ACCOUNT" --partition=normal_q --qos="${QOS:-owl_normal_base}" --constraint=genoa
     --nodes=1 --ntasks=1 --cpus-per-task="${CPUS:-96}" --mem="$MEM" --time="$TIME"
     --mail-user="$MAIL" --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80
     arc/pipeline.slurm "$@")
echo "[submit] commit $(git rev-parse --short HEAD)"
echo "[submit] ${cmd[*]}"
[ "${DRY_RUN:-0}" = "1" ] && exit 0
"${cmd[@]}"
