#!/bin/bash
#SBATCH -J qcore_ale
#SBATCH -p kshdexclu01
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=dcu:4
#SBATCH --mem=0
#SBATCH -t 24:00:00
#SBATCH -D /public/home/putianshu/vis_mlp/ifs_baseline
#SBATCH -o /public/home/putianshu/vis_mlp/ifs_baseline/logs/%j_%x.out
#SBATCH -e /public/home/putianshu/vis_mlp/ifs_baseline/logs/%j_%x.err
#SBATCH --exclusive

set -euo pipefail

module purge
module load compiler/devtoolset/7.3.1
module load mpi/hpcx/2.11.0/gcc-7.3.1
source /public/home/xichen/ncydata/dtk/dtk-24.04.1/env.sh

BASE="${BASE:-/public/home/putianshu/vis_mlp}"
BASELINE_DIR="${BASELINE_DIR:-${BASE}/ifs_baseline}"
TRAIN_DIR="${TRAIN_DIR:-${BASE}/train}"
CKPT_DIR="${CKPT_DIR:-${BASELINE_DIR}/checkpoints}"
RUN_TAG="${RUN_TAG:?RUN_TAG is required}"
SEED="${SEED:?SEED is required}"
HYBRID_DATA_ROOT="${HYBRID_DATA_ROOT:-${BASELINE_DIR}/q_core_hybrid_datasets/${RUN_TAG}}"
EVAL_ROOT="${EVAL_ROOT:-${BASE}/paper_eval_results_pm10_pm25_journal/q_core_hybrid_factorial/${RUN_TAG}}"
SAMPLE_SIZE="${SAMPLE_SIZE:-50000}"
MIN_LOW_VIS="${MIN_LOW_VIS:-200}"
BOOTSTRAP_ITERS="${BOOTSTRAP_ITERS:-1000}"
LIMIT_ROWS="${LIMIT_ROWS:-0}"

mkdir -p "${BASELINE_DIR}/logs" "${EVAL_ROOT}/ale/seed_${SEED}"
source "${BASELINE_DIR}/activate_eval_torch_runtime.sh"
export PYTHONUNBUFFERED=1

pangu_run="exp_qcore_hybrid_${RUN_TAG}_mtw000_seed${SEED}_pm10_pm25"
tianji_run="exp_qcore_hybrid_${RUN_TAG}_mtw111_seed${SEED}_pm10_pm25"
sources="pangu=${HYBRID_DATA_ROOT}/mtw_000|${CKPT_DIR}/${pangu_run}_S2_PhaseB_best_score.pt|AUTO|Pangu q-core;tianji=${HYBRID_DATA_ROOT}/mtw_111|${CKPT_DIR}/${tianji_run}_S2_PhaseB_best_score.pt|AUTO|Tianji q-core"

python "${BASELINE_DIR}/analyze_q_core_ale.py" \
  --sources "${sources}" \
  --reference-source pangu \
  --out-dir "${EVAL_ROOT}/ale/seed_${SEED}" \
  --static-rnn-train-dir "${TRAIN_DIR}" \
  --sample-size "${SAMPLE_SIZE}" \
  --min-low-vis "${MIN_LOW_VIS}" \
  --bootstrap-iters "${BOOTSTRAP_ITERS}" \
  --limit-rows "${LIMIT_ROWS}" \
  --seed "$((20260702 + SEED))"
