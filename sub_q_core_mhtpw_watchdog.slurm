#!/bin/bash
#SBATCH -J mhtpw_watch
#SBATCH -p kshcexclu04
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH -t 240:00:00
#SBATCH -D /public/home/putianshu/vis_mlp/ifs_baseline
#SBATCH -o /public/home/putianshu/vis_mlp/ifs_baseline/logs/%j_%x.out
#SBATCH -e /public/home/putianshu/vis_mlp/ifs_baseline/logs/%j_%x.err

set -euo pipefail

BASELINE_DIR="${BASELINE_DIR:-/public/home/putianshu/vis_mlp/ifs_baseline}"
RUN_TAG="${RUN_TAG:?RUN_TAG is required}"
WATCH_PYTHON="${WATCH_PYTHON:-/public/home/jarvis226/miniconda3/envs/torch/bin/python}"

cd "${BASELINE_DIR}"
mkdir -p logs

if [[ ! -x "${WATCH_PYTHON}" ]]; then
  WATCH_PYTHON="$(command -v python3 || command -v python)"
fi
"${WATCH_PYTHON}" -c 'import sys; assert sys.version_info >= (3, 10), sys.version'

echo "MHTPW watchdog started at: $(date)"
echo "RUN_TAG=${RUN_TAG}"
echo "WATCH_AUTO_RETRY=${WATCH_AUTO_RETRY:-0}"
echo "WATCH_ADOPT_MISSING=${WATCH_ADOPT_MISSING:-0}"
echo "WATCH_POLL_SECONDS=${WATCH_POLL_SECONDS:-180}"
echo "WATCH_STALE_MINUTES=startup:${WATCH_STARTUP_STALE_MINUTES:-45},data:${WATCH_DATA_STALE_MINUTES:-60},training:${WATCH_TRAIN_STALE_MINUTES:-45},validation:${WATCH_VALIDATION_STALE_MINUTES:-120}"
echo "WATCH_CONFIRMATIONS=${WATCH_CONFIRMATIONS:-2}"
echo "WATCH_MAX_RETRIES=${WATCH_MAX_RETRIES:-2}"
echo "WATCH_EXCLUDE_FAILED_NODES=${WATCH_EXCLUDE_FAILED_NODES:-1}"

export PYTHONUNBUFFERED=1
exec "${WATCH_PYTHON}" "${BASELINE_DIR}/watch_q_core_mhtpw_chain.py" --run-tag "${RUN_TAG}"
