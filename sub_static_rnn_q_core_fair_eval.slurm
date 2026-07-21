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
source /public/home/putianshu/vis_mlp/ifs_baseline/activate_eval_torch_runtime.sh

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
MODE="${MODE:-evaluate}"
FEATURE_SET="${FEATURE_SET:-q_core_no_rh2m}"
SOURCE_SCOPE="${SOURCE_SCOPE:-all}"
SEED="${SEED:-42}"
SEEDS="${SEEDS:-42:2025:20260702}"
EVAL_ROOT="${EVAL_ROOT:-${BASE}/paper_eval_results_pm10_pm25_journal/q_core_fair_pangu2025/${RUN_TAG}}"
OUT_DIR="${OUT_DIR:-${EVAL_ROOT}/argmax_paired}"
ANALYSIS_DIR="${ANALYSIS_DIR:-${EVAL_ROOT}/analysis}"

S1_RUN_ID="${S1_RUN_ID:-exp_overlap_static_rnn_s1_${FEATURE_SET}_${RUN_TAG}_pm10_pm25}"
TIANJI_RUN_ID="${TIANJI_RUN_ID:-exp_overlap_static_rnn_s2_tianji_${FEATURE_SET}_${RUN_TAG}_pm10_pm25}"
IFS_RUN_ID="${IFS_RUN_ID:-exp_overlap_static_rnn_s2_ifs_${FEATURE_SET}_${RUN_TAG}_pm10_pm25}"
PANGU2025_RUN_ID="${PANGU2025_RUN_ID:-exp_overlap_static_rnn_s2_pangu2025_${FEATURE_SET}_${RUN_TAG}_pm10_pm25}"
ERA5_2025_RUN_ID="${ERA5_2025_RUN_ID:-exp_overlap_static_rnn_s2_era5_2025_${FEATURE_SET}_${RUN_TAG}_pm10_pm25}"

TIANJI_DATA_DIR="${TIANJI_DATA_DIR:-${BASELINE_DIR}/ml_dataset_overlap_tianji_12h_pm10_pm25_${FEATURE_SET}}"
IFS_DATA_DIR="${IFS_DATA_DIR:-${BASELINE_DIR}/ml_dataset_overlap_ifs_12h_pm10_pm25_${FEATURE_SET}}"
PANGU2025_DATA_DIR="${PANGU2025_DATA_DIR:-${BASELINE_DIR}/ml_dataset_overlap_pangu2025_12h_pm10_pm25_${FEATURE_SET}}"
ERA5_2025_DATA_DIR="${ERA5_2025_DATA_DIR:-${BASELINE_DIR}/ml_dataset_overlap_era5_2025_12h_pm10_pm25_${FEATURE_SET}}"

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
BOOTSTRAP_MAX_ROWS="${BOOTSTRAP_MAX_ROWS:-0}"
AP_HIST_INITIAL_BINS="${AP_HIST_INITIAL_BINS:-4096}"
AP_HIST_MAX_BINS="${AP_HIST_MAX_BINS:-65536}"
AP_HIST_MAX_ERROR="${AP_HIST_MAX_ERROR:-0.0005}"
ECE_BINS="${ECE_BINS:-15}"
FORMATS="${FORMATS:-svg:pdf:png:tiff}"
DPI="${DPI:-600}"
SKIP_VALIDATION_INFERENCE="${SKIP_VALIDATION_INFERENCE:-1}"
NO_TEMP_SCALING="${NO_TEMP_SCALING:-0}"
RUN_PAIRED_SUMMARY="${RUN_PAIRED_SUMMARY:-1}"
TIANJI_SOURCE_TAG="${TIANJI_SOURCE_TAG:-tianji}"
PANGU_SOURCE_TAG="${PANGU_SOURCE_TAG:-pangu2025_${FEATURE_SET}}"

mkdir -p "${BASELINE_DIR}/logs" "${OUT_DIR}" "${ANALYSIS_DIR}"
cd "${BASELINE_DIR}"

if [[ "${MODE}" == "analyze" ]]; then
    CMD=(
        python "${BASELINE_DIR}/analyze_q_core_t925_fair.py"
        --eval-root "${EVAL_ROOT}"
        --out-dir "${ANALYSIS_DIR}"
        --seeds "${SEEDS}"
        --tianji-tag "${TIANJI_SOURCE_TAG}"
        --pangu-tag "${PANGU_SOURCE_TAG}"
        --bootstrap-iters "${BOOTSTRAP_ITERS}"
        --bootstrap-seed "${BOOTSTRAP_SEED}"
        --bootstrap-max-rows "${BOOTSTRAP_MAX_ROWS}"
        --ap-hist-initial-bins "${AP_HIST_INITIAL_BINS}"
        --ap-hist-max-bins "${AP_HIST_MAX_BINS}"
        --ap-hist-max-error "${AP_HIST_MAX_ERROR}"
        --ece-bins "${ECE_BINS}"
        --formats "${FORMATS}"
        --dpi "${DPI}"
    )
    echo "Joint analysis command: ${CMD[*]}"
    "${CMD[@]}"
    exit 0
fi
if [[ "${MODE}" != "evaluate" ]]; then
    echo "ERROR: MODE must be evaluate or analyze, got ${MODE}" >&2
    exit 2
fi
if [[ "${SOURCE_SCOPE}" == "pangu_tianji" && "${FEATURE_SET}" != "q_core_t925_no_rh2m" ]]; then
    echo "[WARN] two-source evaluation is using FEATURE_SET=${FEATURE_SET}, not q_core_t925_no_rh2m" >&2
elif [[ "${SOURCE_SCOPE}" != "pangu_tianji" && "${SOURCE_SCOPE}" != "all" ]]; then
    echo "ERROR: SOURCE_SCOPE must be all or pangu_tianji" >&2
    exit 2
fi
if [[ "${SOURCE_SCOPE}" == "all" && "${FEATURE_SET}" == "q_core_t925_no_rh2m" ]]; then
    echo "ERROR: q_core_t925_no_rh2m is a Pangu--Tianji overlap; use SOURCE_SCOPE=pangu_tianji." >&2
    exit 2
fi

FEATURE_LABEL="q-core"
if [[ "${FEATURE_SET}" == "q_core_t925_no_rh2m" ]]; then
    FEATURE_LABEL="q-core + T925"
fi
PANGU_SPEC="${PANGU_SOURCE_TAG}=${PANGU2025_DATA_DIR}|${PANGU2025_CKPT}|AUTO|Pangu-2025 ${FEATURE_LABEL}"
ERA5_SOURCE_TAG="era5_2025_${FEATURE_SET}"
ERA5_SPEC="${ERA5_SOURCE_TAG}=${ERA5_2025_DATA_DIR}|${ERA5_2025_CKPT}|AUTO|ERA5-2025 ${FEATURE_LABEL} reference analysis"
CMD=(
    python "${BASELINE_DIR}/test_PMST_overlap_forecast_source_s2.py"
    --model_arch static_rnn
    --static_rnn_train_dir "${TRAIN_DIR}"
    --independent_sources
    --threshold_mode argmax
    --skip_bootstrap
    --skip_ifs_forecast_baseline
    --no_figures
    --tianji_data_dir "${TIANJI_DATA_DIR}"
    --tianji_ckpt "${TIANJI_CKPT}"
    --tianji_scaler AUTO
    --out_dir "${OUT_DIR}"
    --device "${DEVICE}"
    --batch_size "${BATCH_SIZE}"
    --num_workers "${NUM_WORKERS}"
)
if [[ "${SKIP_VALIDATION_INFERENCE}" == "1" ]]; then
    CMD+=(--skip_validation_inference)
fi
if [[ "${NO_TEMP_SCALING}" == "1" ]]; then
    CMD+=(--no_temp_scaling)
fi
if [[ "${SOURCE_SCOPE}" == "pangu_tianji" ]]; then
    CMD+=(
        --extra_sources "${PANGU_SPEC}"
        --source_subset "${TIANJI_SOURCE_TAG},${PANGU_SOURCE_TAG}"
    )
    SOURCES="${TIANJI_SOURCE_TAG}=Tianji;${PANGU_SOURCE_TAG}=Pangu-2025"
else
    CMD+=(
        --ifs_data_dir "${IFS_DATA_DIR}"
        --ifs_ckpt "${IFS_CKPT}"
        --ifs_scaler AUTO
        --extra_sources "${PANGU_SPEC};${ERA5_SPEC}"
        --source_subset "${TIANJI_SOURCE_TAG},ifs,${PANGU_SOURCE_TAG},${ERA5_SOURCE_TAG}"
    )
    SOURCES="${TIANJI_SOURCE_TAG}=Tianji;ifs=IFS;${PANGU_SOURCE_TAG}=Pangu-2025;${ERA5_SOURCE_TAG}=ERA5 reference analysis"
fi
if [[ -n "${LIMIT_SAMPLES}" && "${LIMIT_SAMPLES}" != "0" ]]; then
    CMD+=(--limit_samples "${LIMIT_SAMPLES}")
fi
echo "Evaluation command: ${CMD[*]}"
"${CMD[@]}"

if [[ "${RUN_PAIRED_SUMMARY}" != "1" ]]; then
    exit 0
fi
SUMMARY_CMD=(
    python "${BASELINE_DIR}/summarize_q_core_paired_eval.py"
    --eval-dir "${OUT_DIR}"
    --sources "${SOURCES}"
    --reference "${PANGU_SOURCE_TAG}"
    --bootstrap-iters "${BOOTSTRAP_ITERS}"
    --bootstrap-seed "${BOOTSTRAP_SEED}"
)
echo "Paired-summary command: ${SUMMARY_CMD[*]}"
"${SUMMARY_CMD[@]}"
