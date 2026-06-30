#!/bin/bash
#SBATCH -J q1000_mech
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

echo "Q1000 mechanism analysis started at: $(date)"
echo "Node: $(hostname)"

module purge
module load compiler/devtoolset/7.3.1
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
SCRIPT_PATH="${SCRIPT_PATH:-${BASE}/paper_eval/analyze_q1000_mechanism.py}"
FEATURE_SET="${FEATURE_SET:-source_full}"
FEATURES="${FEATURES:-Q_1000,DP_1000,Q_925,Q1000_MINUS_Q925}"
OUT_DIR="${OUT_DIR:-${BASE}/paper_eval_results_pm10_pm25_journal/q1000_mechanism_${FEATURE_SET}}"
EVAL_DIR="${EVAL_DIR:-${BASE}/paper_eval_results_pm10_pm25_journal/best_effort_source_full_argmax/figure1_all_sources}"
REFERENCE_SOURCE="${REFERENCE_SOURCE:-era5_2025_source_full}"
PANGU_TAG="${PANGU_TAG:-pangu2025_source_full}"
NUMERICAL_TAGS="${NUMERICAL_TAGS:-tianji,ifs,era5_2025_source_full}"
BOOTSTRAP_ITERS="${BOOTSTRAP_ITERS:-1000}"
LIMIT_SAMPLES="${LIMIT_SAMPLES:-}"
MIN_PAIRS="${MIN_PAIRS:-100}"
REQUIRE_CASE_CONTROL="${REQUIRE_CASE_CONTROL:-1}"
DISTRIBUTION_ONLY="${DISTRIBUTION_ONLY:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
PANGU_SOURCE_LABEL="${PANGU_SOURCE_LABEL:-Pangu-2025 ${FEATURE_SET}}"

case "${FEATURE_SET}" in
    source_full)
        TIANJI_DATA_DIR="${TIANJI_DATA_DIR:-${BASELINE_DIR}/ml_dataset_overlap_tianji_12h_pm10_pm25_source_full}"
        IFS_DATA_DIR="${IFS_DATA_DIR:-${BASELINE_DIR}/ml_dataset_overlap_ifs_12h_pm10_pm25_source_full}"
        ERA5_2025_DATA_DIR="${ERA5_2025_DATA_DIR:-${BASELINE_DIR}/ml_dataset_overlap_era5_2025_12h_pm10_pm25_source_full}"
        PANGU2025_DATA_DIR="${PANGU2025_DATA_DIR:-${BASELINE_DIR}/ml_dataset_overlap_pangu2025_12h_pm10_pm25_source_full}"
        ;;
    q_core_no_rh2m|q_core|q1000_core)
        FEATURE_SET="q_core_no_rh2m"
        TIANJI_DATA_DIR="${TIANJI_DATA_DIR:-${BASELINE_DIR}/ml_dataset_overlap_tianji_12h_pm10_pm25_q_core_no_rh2m}"
        IFS_DATA_DIR="${IFS_DATA_DIR:-${BASELINE_DIR}/ml_dataset_overlap_ifs_12h_pm10_pm25_q_core_no_rh2m}"
        ERA5_2025_DATA_DIR="${ERA5_2025_DATA_DIR:-${BASELINE_DIR}/ml_dataset_overlap_era5_2025_12h_pm10_pm25_q_core_no_rh2m}"
        PANGU2025_DATA_DIR="${PANGU2025_DATA_DIR:-${BASELINE_DIR}/ml_dataset_overlap_pangu2025_12h_pm10_pm25_q_core_no_rh2m}"
        REFERENCE_SOURCE="${REFERENCE_SOURCE:-era5_2025_source_full}"
        PANGU_TAG="${PANGU_TAG:-pangu2025_source_full}"
        NUMERICAL_TAGS="${NUMERICAL_TAGS:-tianji,ifs,era5_2025_source_full}"
        ;;
    *)
        echo "ERROR: unsupported FEATURE_SET=${FEATURE_SET}; use source_full or q_core_no_rh2m." >&2
        exit 2
        ;;
esac

SOURCES_DEFAULT="tianji=${TIANJI_DATA_DIR}|Tianji ${FEATURE_SET}|numerical;ifs=${IFS_DATA_DIR}|IFS ${FEATURE_SET}|numerical;era5_2025_source_full=${ERA5_2025_DATA_DIR}|ERA5 reference analysis ${FEATURE_SET}|reference;pangu2025_source_full=${PANGU2025_DATA_DIR}|${PANGU_SOURCE_LABEL}|ai"
SOURCES="${SOURCES:-${SOURCES_DEFAULT}}"

mkdir -p "${BASELINE_DIR}/logs" "${OUT_DIR}"
cd "${BASELINE_DIR}"

if [[ "${DISTRIBUTION_ONLY}" != "1" && "${DISTRIBUTION_ONLY}" != "true" ]] && \
   [[ "${REQUIRE_CASE_CONTROL}" == "1" || "${REQUIRE_CASE_CONTROL}" == "true" ]]; then
    MISSING_CASE_FILES=()
    for tag in "${PANGU_TAG}" ${NUMERICAL_TAGS//,/ }; do
        expected="${EVAL_DIR}/per_sample_${tag}.csv"
        if [[ ! -s "${expected}" ]]; then
            MISSING_CASE_FILES+=("${expected}")
        fi
    done
    if (( ${#MISSING_CASE_FILES[@]} > 0 )); then
        printf 'ERROR: case-control requires non-empty per-sample files; missing:\n' >&2
        printf '  %s\n' "${MISSING_CASE_FILES[@]}" >&2
        printf 'Generate them first with:\n  NO_PER_SAMPLE_CSV=0 bash submit_static_rnn_source_full_argmax_eval.sh figure1_all_sources\n' >&2
        exit 2
    fi
fi

CMD=(
    python "${SCRIPT_PATH}"
    --sources "${SOURCES}"
    --reference_source "${REFERENCE_SOURCE}"
    --label_source "${REFERENCE_SOURCE}"
    --pangu_tag "${PANGU_TAG}"
    --numerical_tags "${NUMERICAL_TAGS}"
    --features "${FEATURES}"
    --out_dir "${OUT_DIR}"
    --eval_dir "${EVAL_DIR}"
    --bootstrap_iters "${BOOTSTRAP_ITERS}"
    --min_pairs "${MIN_PAIRS}"
)
if [[ "${REQUIRE_CASE_CONTROL}" == "1" || "${REQUIRE_CASE_CONTROL}" == "true" ]]; then
    CMD+=(--require_case_control)
fi
if [[ "${DISTRIBUTION_ONLY}" == "1" || "${DISTRIBUTION_ONLY}" == "true" ]]; then
    CMD+=(--distribution_only)
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

echo "Q1000 mechanism analysis finished at: $(date)"
