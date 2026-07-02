#!/bin/bash
#SBATCH -J overlap_single_eval
#SBATCH -p kshcexclu04
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH -t 24:00:00
#SBATCH -D /public/home/putianshu/vis_mlp/ifs_baseline
#SBATCH -o /public/home/putianshu/vis_mlp/ifs_baseline/logs/%j_%x.out
#SBATCH -e /public/home/putianshu/vis_mlp/ifs_baseline/logs/%j_%x.err
#SBATCH --exclusive

set -euo pipefail

echo "Overlap single-source Static-RNN evaluation started at: $(date)"
echo "Node: $(hostname)"

module purge
module load compiler/devtoolset/7.3.1
module load mpi/hpcx/2.11.0/gcc-7.3.1
source /public/home/xichen/ncydata/dtk/dtk-24.04.1/env.sh
export LD_LIBRARY_PATH=/public/home/xichen/ncydata/dtk/dtk-24.04.1/.hyhal/lib:$LD_LIBRARY_PATH
export CONDA_NO_PLUGINS=true
export CONDA_SOLVER=classic
source /public/home/putianshu/vis_mlp/ifs_baseline/activate_eval_torch_runtime.sh

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"

BASE="${BASE:-/public/home/putianshu/vis_mlp}"
BASELINE_DIR="${BASELINE_DIR:-${BASE}/ifs_baseline}"
EVAL_SCRIPT="${EVAL_SCRIPT:-${BASE}/paper_eval/run_static_rnn_lowvis_eval_journal.py}"
TRAIN_DIR="${TRAIN_DIR:-${BASE}/train}"
SOURCE_TAG="${SOURCE_TAG:-tianji}"
FEATURE_SET="${FEATURE_SET:-common_core}"
if [[ -z "${DATA_DIR:-}" ]]; then
    if [[ "${SOURCE_TAG}" == "T2ND_rh2m" ]]; then
        DATA_DIR="${BASELINE_DIR}/ml_dataset_overlap_tianji_12h_pm10_pm25_T2ND_rh2m_${FEATURE_SET}"
    else
        DATA_DIR="${BASELINE_DIR}/ml_dataset_overlap_${SOURCE_TAG}_12h_pm10_pm25_${FEATURE_SET}"
    fi
fi
CKPT_DIR="${CKPT_DIR:-${BASELINE_DIR}/checkpoints}"
STAGE_TAG="${STAGE_TAG:-S2_PhaseB}"
RUN_SOURCE_TAG="${RUN_SOURCE_TAG:-${SOURCE_TAG}_${FEATURE_SET}}"
RUN_ID="${RUN_ID:-exp_overlap_static_rnn_s2_${RUN_SOURCE_TAG}_pm10_pm25}"
CKPT_PATH="${CKPT_PATH:-${CKPT_DIR}/${RUN_ID}_${STAGE_TAG}_best_score.pt}"
OUT_DIR="${OUT_DIR:-${BASE}/paper_eval_results_pm10_pm25_journal/overlap_source_single/${RUN_SOURCE_TAG}}"
DEVICE="${DEVICE:-cpu}"
LIMIT_SAMPLES="${LIMIT_SAMPLES:-}"
PLOTS="${PLOTS:-core}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p "${BASELINE_DIR}/logs"

CMD=(
    python "${EVAL_SCRIPT}"
    --mode single
    --base "${BASE}"
    --train_dir "${TRAIN_DIR}"
    --data_dir "${DATA_DIR}"
    --ckpt_dir "${CKPT_DIR}"
    --ckpt_path "${CKPT_PATH}"
    --run_id "${RUN_ID}"
    --stage_tag "${STAGE_TAG}"
    --out_dir "${OUT_DIR}"
    --device "${DEVICE}"
    --plots "${PLOTS}"
)
if [[ -n "${LIMIT_SAMPLES}" ]]; then
    CMD+=(--limit_samples "${LIMIT_SAMPLES}")
fi
if [[ -n "${EXTRA_ARGS}" ]]; then
    # shellcheck disable=SC2206
    EXTRA_ARGV=(${EXTRA_ARGS})
    CMD+=("${EXTRA_ARGV[@]}")
fi

echo "Command: ${CMD[*]}"
"${CMD[@]}"

echo "Overlap single-source Static-RNN evaluation finished at: $(date)"
