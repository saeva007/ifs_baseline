#!/bin/bash
#SBATCH -J s1_zero_transfer_eval
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

echo "S1 zero-transfer source evaluation started at: $(date)"
echo "Node: $(hostname)"

module purge
module load compiler/devtoolset/7.3.1
module load mpi/hpcx/2.11.0/gcc-7.3.1
source /public/home/xichen/ncydata/dtk/dtk-24.04.1/env.sh
export LD_LIBRARY_PATH=/public/home/xichen/ncydata/dtk/dtk-24.04.1/.hyhal/lib:$LD_LIBRARY_PATH
export CONDA_NO_PLUGINS=true
export CONDA_SOLVER=classic
source /public/home/putianshu/miniconda3/bin/activate
conda activate torch
export LD_LIBRARY_PATH=/public/home/xichen/.conda/envs/py310_ppy/openssl/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/public/home/xichen/ncydata/panpy_test_liud/hipnn/lib/release/:$LD_LIBRARY_PATH

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"

BASE="${BASE:-/public/home/putianshu/vis_mlp}"
BASELINE_DIR="${BASELINE_DIR:-${BASE}/ifs_baseline}"
EVAL_SCRIPT="${EVAL_SCRIPT:-${BASELINE_DIR}/test_PMST_overlap_forecast_source_s2.py}"
MERGE_SCRIPT="${MERGE_SCRIPT:-${BASELINE_DIR}/merge_overlap_source_eval_metrics.py}"
TRAIN_DIR="${TRAIN_DIR:-${BASE}/train}"
CKPT_DIR="${CKPT_DIR:-${BASELINE_DIR}/checkpoints}"
S1_RUN_ID="${S1_RUN_ID:-exp_overlap_static_rnn_s1_common_core_pm10_pm25}"
S1_CKPT="${S1_CKPT:-${CKPT_DIR}/${S1_RUN_ID}_S1_best_score.pt}"
S1_SCALER="${S1_SCALER:-${CKPT_DIR}/robust_scaler_${S1_RUN_ID}_s1_w12_dyn27_pm.pkl}"
OUT_ROOT="${OUT_ROOT:-${BASE}/paper_eval_results_pm10_pm25_journal/zero_transfer_s1_forecast_sources}"
DEVICE="${DEVICE:-cpu}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
NUM_WORKERS="${NUM_WORKERS:-0}"
LIMIT_SAMPLES="${LIMIT_SAMPLES:-}"
SOURCE_GROUP="${SOURCE_GROUP:-all}"
MERGE_ONLY="${MERGE_ONLY:-0}"
THRESHOLD_MODE="${THRESHOLD_MODE:-argmax}"
THRESHOLD_SEARCH_POLICY="${THRESHOLD_SEARCH_POLICY:-response}"
SKIP_VALIDATION_INFERENCE="${SKIP_VALIDATION_INFERENCE:-1}"
SKIP_IFS_FORECAST_BASELINE="${SKIP_IFS_FORECAST_BASELINE:-1}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

TIANJI_DATA_DIR="${TIANJI_DATA_DIR:-${BASELINE_DIR}/ml_dataset_overlap_tianji_12h_pm10_pm25_common_core}"
IFS_DATA_DIR="${IFS_DATA_DIR:-${BASELINE_DIR}/ml_dataset_overlap_ifs_12h_pm10_pm25_common_core}"
T2ND_DATA_DIR="${T2ND_DATA_DIR:-${BASELINE_DIR}/ml_dataset_overlap_tianji_12h_pm10_pm25_T2ND_rh2m_common_core}"
PANGU_DATA_DIR="${PANGU_DATA_DIR:-${BASELINE_DIR}/ml_dataset_overlap_pangu2021_12h_pm10_pm25_common_core}"
ERA5_DATA_DIR="${ERA5_DATA_DIR:-${BASELINE_DIR}/ml_dataset_overlap_era5_2025_12h_pm10_pm25_common_core}"

EXTRA_SOURCES="${EXTRA_SOURCES:-T2ND_rh2m_common_core=${T2ND_DATA_DIR}|${S1_CKPT}|${S1_SCALER}|S1 zero-transfer / T2ND RH2M; pangu2021_common_core=${PANGU_DATA_DIR}|${S1_CKPT}|${S1_SCALER}|S1 zero-transfer / Pangu-2021; era5_2025_common_core=${ERA5_DATA_DIR}|${S1_CKPT}|${S1_SCALER}|S1 zero-transfer / ERA5-2025}"

mkdir -p "${BASELINE_DIR}/logs" "${OUT_ROOT}"

if [[ "${MERGE_ONLY}" == "1" || "${SOURCE_GROUP}" == "merge" ]]; then
    OUT_DIR="${OUT_DIR:-${OUT_ROOT}/merged}"
    CMD=(python "${MERGE_SCRIPT}" --root_dir "${OUT_ROOT}" --out_dir "${OUT_DIR}")
    echo "Command: ${CMD[*]}"
    "${CMD[@]}"
    echo "S1 zero-transfer source merge finished at: $(date)"
    exit 0
fi

case "${SOURCE_GROUP}" in
    all)
        SOURCE_SUBSET="tianji,ifs,T2ND_rh2m_common_core,pangu2021_common_core,era5_2025_common_core"
        RUN_TAG="all"
        ;;
    tianji|tianji_common_core)
        SOURCE_SUBSET="tianji"
        RUN_TAG="tianji_common_core"
        ;;
    ifs|ifs_common_core)
        SOURCE_SUBSET="ifs"
        RUN_TAG="ifs_common_core"
        ;;
    T2ND_rh2m|T2ND_rh2m_common_core|t2nd|t2nd_common_core)
        SOURCE_SUBSET="T2ND_rh2m_common_core"
        RUN_TAG="T2ND_rh2m_common_core"
        ;;
    pangu|pangu2021|pangu2021_common_core)
        SOURCE_SUBSET="pangu2021_common_core"
        RUN_TAG="pangu2021_common_core"
        ;;
    era5|era5_2025|era5_2025_common_core)
        SOURCE_SUBSET="era5_2025_common_core"
        RUN_TAG="era5_2025_common_core"
        ;;
    *)
        SOURCE_SUBSET="${SOURCE_GROUP}"
        RUN_TAG="${SOURCE_GROUP//,/__}"
        ;;
esac

OUT_DIR="${OUT_DIR:-${OUT_ROOT}/${RUN_TAG}}"

CMD=(
    python "${EVAL_SCRIPT}"
    --model_arch static_rnn
    --static_rnn_train_dir "${TRAIN_DIR}"
    --independent_sources
    --zero_transfer_s1
    --shared_ckpt "${S1_CKPT}"
    --shared_scaler "${S1_SCALER}"
    --source_subset "${SOURCE_SUBSET}"
    --tianji_data_dir "${TIANJI_DATA_DIR}"
    --ifs_data_dir "${IFS_DATA_DIR}"
    --extra_sources "${EXTRA_SOURCES}"
    --out_dir "${OUT_DIR}"
    --device "${DEVICE}"
    --batch_size "${BATCH_SIZE}"
    --num_workers "${NUM_WORKERS}"
    --threshold_mode "${THRESHOLD_MODE}"
    --threshold_search_policy "${THRESHOLD_SEARCH_POLICY}"
    --skip_bootstrap
    --no_per_sample_csv
)

if [[ "${SKIP_VALIDATION_INFERENCE}" == "1" && "${THRESHOLD_MODE}" != "val_search" ]]; then
    CMD+=(--skip_validation_inference)
fi
if [[ "${SKIP_IFS_FORECAST_BASELINE}" == "1" ]]; then
    CMD+=(--skip_ifs_forecast_baseline)
fi
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

echo "S1 zero-transfer source evaluation finished at: $(date)"
