#!/usr/bin/env bash
# One-time environment build on an ARC login node (OWL or Tinkercliffs; /home is shared).
#
#   bash arc/setup_env.sh                 # -> $HOME/miniconda3/envs/recsys
#
# CPU-only torch: the pipeline runs on OWL CPU nodes, and an unpinned torch now pulls a
# CUDA-13 build that does not match ARC's CUDA 12.6 module anyway.
# Java comes from conda-forge (openjdk 17, supported by Spark 3.5) so no ARC module is needed.
set -euo pipefail
ENV="${CONDA_ENV:-$HOME/miniconda3/envs/recsys}"
CONDA="${CONDA_EXE:-$HOME/miniconda3/bin/conda}"
cd "$(git rev-parse --show-toplevel)"

[ -x "$CONDA" ] || { echo "conda not found at $CONDA (set CONDA_EXE)" >&2; exit 1; }
if [ ! -x "$ENV/bin/python" ]; then
  "$CONDA" create -y -p "$ENV" -c conda-forge --override-channels python=3.12 openjdk=17
fi
PY="$ENV/bin/python"
"$PY" -V
"$PY" -m pip install --upgrade pip
"$PY" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
"$PY" -m pip install -e ".[dev]"

# Verify, the same way every job will.
"$PY" -c "import recsys, pyspark, lightgbm, torch, faiss, sklearn, yaml; print('imports OK', torch.__version__, pyspark.__version__)"
JAVA_HOME="$ENV/lib/jvm"; [ -x "$JAVA_HOME/bin/java" ] || JAVA_HOME="$ENV"
"$JAVA_HOME/bin/java" -version
echo "find zero-byte binaries (must print nothing):"; find "$ENV/bin" -type f -size 0
echo "Now run the tests on a compute node, or here if the login node allows a short run:"
echo "  JAVA_HOME=$JAVA_HOME $PY -m pytest -q"
