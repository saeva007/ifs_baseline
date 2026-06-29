#!/bin/bash
#SBATCH -J qcore_fair_eval
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

module purge
module load compiler/devtoolset/7.3.1
module load mpi/hpcx/2.11.0/gcc-7.3.1
source /public/home/xichen/ncydata/dtk/dtk-24.04.1/env.sh
export LD_LIBRARY_PATH=/public/home/xichen/ncydata/dtk/dtk-24.04.1/.hyhal/lib:${LD_LIBRARY_PATH:-}
export CONDA_NO_PLUGINS=true
export CONDA_SOLVER=classic
source /public/home/putianshu/miniconda3/bin/activate
conda activate torch

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"

BASE="${BASE:-/public/home/putianshu/vis_mlp}"
BASELINE_DIR="${BASELINE_DIR:-${BASE}/ifs_baseline}"
TRAIN_DIR="${TRAIN_DIR:-${BASE}/train}"
CKPT_DIR="${CKPT_DIR:-${BASELINE_DIR}/checkpoints}"
RUN_TAG="${RUN_TAG:?RUN_TAG is required}"
OUT_DIR="${OUT_DIR:-${BASE}/paper_eval_results_pm10_pm25_journal/q_core_fair_pangu2025/${RUN_TAG}/argmax_paired}"

S1_RUN_ID="${S1_RUN_ID:-exp_overlap_static_rnn_s1_q_core_no_rh2m_${RUN_TAG}_pm10_pm25}"
TIANJI_RUN_ID="${TIANJI_RUN_ID:-exp_overlap_static_rnn_s2_tianji_q_core_no_rh2m_${RUN_TAG}_pm10_pm25}"
IFS_RUN_ID="${IFS_RUN_ID:-exp_overlap_static_rnn_s2_ifs_q_core_no_rh2m_${RUN_TAG}_pm10_pm25}"
PANGU2025_RUN_ID="${PANGU2025_RUN_ID:-exp_overlap_static_rnn_s2_pangu2025_q_core_no_rh2m_${RUN_TAG}_pm10_pm25}"
ERA5_2025_RUN_ID="${ERA5_2025_RUN_ID:-exp_overlap_static_rnn_s2_era5_2025_q_core_no_rh2m_${RUN_TAG}_pm10_pm25}"

TIANJI_DATA_DIR="${TIANJI_DATA_DIR:-${BASELINE_DIR}/ml_dataset_overlap_tianji_12h_pm10_pm25_q_core_no_rh2m}"
IFS_DATA_DIR="${IFS_DATA_DIR:-${BASELINE_DIR}/ml_dataset_overlap_ifs_12h_pm10_pm25_q_core_no_rh2m}"
PANGU2025_DATA_DIR="${PANGU2025_DATA_DIR:-${BASELINE_DIR}/ml_dataset_overlap_pangu2025_12h_pm10_pm25_q_core_no_rh2m}"
ERA5_2025_DATA_DIR="${ERA5_2025_DATA_DIR:-${BASELINE_DIR}/ml_dataset_overlap_era5_2025_12h_pm10_pm25_q_core_no_rh2m}"

TIANJI_CKPT="${TIANJI_CKPT:-${CKPT_DIR}/${TIANJI_RUN_ID}_S2_PhaseB_best_score.pt}"
IFS_CKPT="${IFS_CKPT:-${CKPT_DIR}/${IFS_RUN_ID}_S2_PhaseB_best_score.pt}"
PANGU2025_CKPT="${PANGU2025_CKPT:-${CKPT_DIR}/${PANGU2025_RUN_ID}_S2_PhaseB_best_score.pt}"
ERA5_2025_CKPT="${ERA5_2025_CKPT:-${CKPT_DIR}/${ERA5_2025_RUN_ID}_S2_PhaseB_best_score.pt}"

DEVICE="${DEVICE:-cpu}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
NUM_WORKERS="${NUM_WORKERS:-0}"
LIMIT_SAMPLES="${LIMIT_SAMPLES:-}"
BOOTSTRAP_ITERS="${BOOTSTRAP_ITERS:-1000}"
BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-20250629}"

mkdir -p "${BASELINE_DIR}/logs" "${OUT_DIR}"
cd "${BASELINE_DIR}"

PANGU_SPEC="pangu2025_q_core_no_rh2m=${PANGU2025_DATA_DIR}|${PANGU2025_CKPT}|AUTO|Pangu-2025 q-core"
ERA5_SPEC="era5_2025_q_core_no_rh2m=${ERA5_2025_DATA_DIR}|${ERA5_2025_CKPT}|AUTO|ERA5-2025 q-core reference analysis"
CMD=(
    python "${BASELINE_DIR}/test_PMST_overlap_forecast_source_s2.py"
    --model_arch static_rnn
    --static_rnn_train_dir "${TRAIN_DIR}"
    --independent_sources
    --threshold_mode argmax
    --skip_validation_inference
    --skip_bootstrap
    --skip_ifs_forecast_baseline
    --no_figures
    --tianji_data_dir "${TIANJI_DATA_DIR}"
    --tianji_ckpt "${TIANJI_CKPT}"
    --tianji_scaler AUTO
    --ifs_data_dir "${IFS_DATA_DIR}"
    --ifs_ckpt "${IFS_CKPT}"
    --ifs_scaler AUTO
    --extra_sources "${PANGU_SPEC};${ERA5_SPEC}"
    --source_subset "tianji,ifs,pangu2025_q_core_no_rh2m,era5_2025_q_core_no_rh2m"
    --out_dir "${OUT_DIR}"
    --device "${DEVICE}"
    --batch_size "${BATCH_SIZE}"
    --num_workers "${NUM_WORKERS}"
)
if [[ -n "${LIMIT_SAMPLES}" ]]; then
    CMD+=(--limit_samples "${LIMIT_SAMPLES}")
fi
echo "Evaluation command: ${CMD[*]}"
"${CMD[@]}"

SOURCES="tianji=Tianji;ifs=IFS;pangu2025_q_core_no_rh2m=Pangu-2025;era5_2025_q_core_no_rh2m=ERA5 reference analysis"
SUMMARY_CMD=(
    python "${BASELINE_DIR}/summarize_q_core_paired_eval.py"
    --eval-dir "${OUT_DIR}"
    --sources "${SOURCES}"
    --reference pangu2025_q_core_no_rh2m
    --bootstrap-iters "${BOOTSTRAP_ITERS}"
    --bootstrap-seed "${BOOTSTRAP_SEED}"
)
echo "Paired-summary command: ${SUMMARY_CMD[*]}"
"${SUMMARY_CMD[@]}"

