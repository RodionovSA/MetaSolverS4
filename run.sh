#!/usr/bin/env bash
set -euo pipefail

# (optional) activate your env
# source ~/miniconda3/bin/activate s4_env

# Keep math libs from oversubscribing when using many MPI ranks
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}

# You can override these when running: NP=64 SCRIPT=foo.py ./run.sh
NP=${NP:-52}
SCRIPT=${SCRIPT:-run_s4.py}

mpirun -n "$NP" \
  -x OMP_NUM_THREADS \
  -x MKL_NUM_THREADS \
  -x OPENBLAS_NUM_THREADS \
  -x NUMEXPR_NUM_THREADS \
  python "$SCRIPT"
