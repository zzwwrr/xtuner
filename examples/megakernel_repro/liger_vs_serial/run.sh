#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ARM=${1:-}
if [[ "$ARM" != serial && "$ARM" != liger ]]; then
  echo 'Usage: bash run.sh serial|liger [benchmark arguments]' >&2
  exit 2
fi
shift
if [[ ${GPU_AUTHORIZATION:-} != GRANTED_BY_USER ]]; then
  echo 'Set GPU_AUTHORIZATION=GRANTED_BY_USER after obtaining GPU execution authorization.' >&2
  exit 2
fi
PY=${PYTHON:-python}
mkdir -p "$HERE/runs"
RUN=$(mktemp -d "$HERE/runs/$ARM-XXXXXX")
mkdir -p "$RUN/ranks" "$RUN/cache"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
export XDG_CACHE_HOME="$RUN/cache/xdg" TRITON_CACHE_DIR="$RUN/cache/triton"
export CUDA_CACHE_PATH="$RUN/cache/cuda" HF_HOME="$RUN/cache/hf"
# This historical isolated comparison uses upstream automatic templates and the RDC path.
unset LIGER_MOE_FORCE_CONFIG LIGER_MOE_BWD_FORCE_CONFIG LIGER_DIAG_NO_K_SPLIT
export LIGER_MOE_SM90_NONRDC=0
COMMAND=("$PY" -m torch.distributed.run --standalone --nproc_per_node=8
  --log-dir "$RUN/ranks" --redirects 3 --tee 3 "$HERE/compare.py"
  --backend "$ARM" --tokens-per-rank 8192 --hidden 2048 --intermediate 768
  --experts 128 --topk 8 --atol 0.03 --rtol 0.03
  --seed 42 --warmup 5 --iters 20 --max-relative-l2 0.02 "$@")
printf '%q ' "${COMMAND[@]}" > "$RUN/command.txt"
echo "RUN=$RUN"
set +e
timeout --signal=TERM --kill-after=30s "${RUN_TIMEOUT:-10m}" "${COMMAND[@]}" > "$RUN/launch.log" 2>&1
STATUS=$?
set -e
printf '%s\n' "$STATUS" > "$RUN/exit-code.txt"
tail -c 1800 "$RUN/launch.log"
exit "$STATUS"
