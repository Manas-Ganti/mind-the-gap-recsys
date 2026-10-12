#!/usr/bin/env bash
# Shared environment for this project's VT ARC jobs (sourced, not executed).
# Patterned on the VLM project's arc_env.sh; every check here exists because its absence
# once cost an allocation (see docs/arc.md).
#
# No `set -e` here: the .slurm file sets it for the run.

# ---- interpreter: absolute path, never `conda activate` --------------------- #
# `conda activate` in a batch shell can report success without switching interpreters,
# and a bare `python` once resolved to a 0-byte binary that "completed" jobs in 2 s.
# RECSYS_ENV, not CONDA_ENV: other projects export CONDA_ENV from ~/.bashrc.
export CONDA_ENV="${RECSYS_ENV:-$HOME/miniconda3/envs/recsys}"
if [ ! -f "$CONDA_ENV/.recsys-env" ]; then
  echo "[arc_env] FATAL: $CONDA_ENV is not this project's env (no .recsys-env marker); run arc/setup_env.sh" >&2
  return 1 2>/dev/null || exit 1
fi
case "$CONDA_ENV" in /*) ;; *) echo "[arc_env] FATAL: CONDA_ENV must be an absolute path" >&2; return 1 2>/dev/null || exit 1 ;; esac
export PY="$CONDA_ENV/bin/python"
export CONDA_PREFIX="$CONDA_ENV"
export PATH="$CONDA_ENV/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

if ! "$PY" -V 2>&1 | grep -q '^Python 3\.'; then
  echo "[arc_env] FATAL: $PY did not print a Python version (missing or 0-byte binary?)" >&2
  echo "[arc_env]        check: ls -la $PY; find $CONDA_ENV/bin -type f -size 0" >&2
  return 1 2>/dev/null || exit 1
fi
"$PY" -c '
import os, sys
env = os.path.realpath(os.environ["CONDA_ENV"])
print("[arc_env] python=" + sys.executable + " " + sys.version.split()[0], flush=True)
if not os.path.realpath(sys.executable).startswith(env):
    sys.exit("[arc_env] FATAL: interpreter is not inside " + env)
' || { return 1 2>/dev/null || exit 1; }

# arc_require <module>... : fail in seconds, not an hour in, if this env lacks a stage's imports.
arc_require() {
  [ $# -gt 0 ] || return 0
  REQ="$*" "$PY" -c '
import importlib.util, os, sys
missing = [m for m in os.environ["REQ"].split() if importlib.util.find_spec(m) is None]
if missing:
    sys.exit("[arc_env] FATAL: missing %s in %s (run arc/setup_env.sh)" % (", ".join(missing), sys.executable))
print("[arc_env] requires " + os.environ["REQ"] + " - present", flush=True)
' || { return 1 2>/dev/null || exit 1; }
}

# ---- Java for Spark: from the env (conda-forge openjdk), not an ARC module --------- #
if [ -x "$CONDA_ENV/lib/jvm/bin/java" ]; then export JAVA_HOME="$CONDA_ENV/lib/jvm"; else export JAVA_HOME="$CONDA_ENV"; fi
if ! "$JAVA_HOME/bin/java" -version >/dev/null 2>&1; then
  echo "[arc_env] FATAL: no working java under $JAVA_HOME (setup_env.sh installs openjdk=17)" >&2
  return 1 2>/dev/null || exit 1
fi
echo "[arc_env] java=$("$JAVA_HOME/bin/java" -version 2>&1 | head -1)"
# Spark shuffle/spill: the job's private tmp if SLURM provides one.
export SPARK_LOCAL_DIRS="${SPARK_LOCAL_DIRS:-${TMPDIR:-/tmp}}"

# ---- threads: one process uses the whole allocation --------------------------- #
# (Unlike a fork pool, our numpy/scipy/torch/LightGBM run in one process, so threads =
# allocated cores rather than 1.)
_n="${SLURM_CPUS_PER_TASK:-8}"
export OMP_NUM_THREADS="$_n" OPENBLAS_NUM_THREADS="$_n" MKL_NUM_THREADS="$_n"
unset _n
export MPLBACKEND=Agg
export TOKENIZERS_PARALLELISM=false

# ---- caches: /home, not /projects (/projects/$USER is not writable here) ---------- #
export HF_HOME="${HF_HOME:-/home/$USER/hf_cache}"

# ---- optional Telegram notifications ------------------------------------------- #
# Reuses the VLM project's secrets file if present (TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID).
# Every function no-ops when unset and never fails the job.
[ -f "$HOME/.config/vrr/secrets.env" ] && source "$HOME/.config/vrr/secrets.env"
_arc_tg_ready() { [ -n "${TELEGRAM_BOT_TOKEN:-}" ] && [ -n "${TELEGRAM_CHAT_ID:-}" ]; }
arc_notify() {
  _arc_tg_ready || return 0
  curl -s -m 15 -o /dev/null -d chat_id="$TELEGRAM_CHAT_ID" --data-urlencode text="$1" \
    "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" 2>/dev/null || true
}
arc_notify_finish() {
  local rc=$? log
  log="$(scontrol show job "${SLURM_JOB_ID:-0}" 2>/dev/null | tr ' ' '\n' | sed -n 's/^StdOut=//p' | head -1)"
  arc_notify "$([ "$rc" -eq 0 ] && echo OK || echo "FAILED rc=$rc") ${SLURM_JOB_NAME:-job} ${SLURM_JOB_ID:-?}
$( [ -f "$log" ] && tail -n 25 "$log" | cut -c1-180 | tail -c 3000)"
  return "$rc"
}

# ---- provenance ------------------------------------------------------------------ #
export PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$PROJECT_DIR"
_commit="$(git rev-parse --short HEAD 2>/dev/null || echo no-git)"
if [ -n "$(git status --porcelain -- src configs arc 2>/dev/null)" ]; then
  echo "[arc_env] WARNING: uncommitted changes in src/ configs/ arc/ at commit $_commit"
fi
echo "[arc_env] node=$(hostname) cpus=${SLURM_CPUS_PER_TASK:-?} mem=${SLURM_MEM_PER_NODE:-?}MB commit=$_commit hf_home=$HF_HOME spark_local=$SPARK_LOCAL_DIRS"
unset _commit
