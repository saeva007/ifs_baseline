#!/bin/bash
# Corrected canonical Pangu data-quality chain:
# station verification -> corrected source-full dataset -> lineage audit -> Q1000 mechanism analysis.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
mkdir -p logs

BASE="${BASE:-/public/home/putianshu/vis_mlp}"
BASELINE_DIR="${BASELINE_DIR:-${BASE}/ifs_baseline}"
RUN_TAG="${RUN_TAG:-q1000_pangu2025_canonical_$(date +%Y%m%d_%H%M%S)}"
DRY_RUN="${DRY_RUN:-0}"

OLD_PANGU_STATION_FILE="${OLD_PANGU_STATION_FILE:-${BASELINE_DIR}/pangu_station/pangu_station_2025_lead12_23h.nc}"
PANGU2025_STATION_FILE="${PANGU2025_STATION_FILE:-${BASELINE_DIR}/pangu_station/pangu_station_2025_lead12_23h_canonical.nc}"
TARGET_FILE="${TARGET_FILE:-${BASE}/tianji_auto_station/merged_final_all_vars.nc}"
EXPECTED_PANGU_LEAD_MIN_HOURS="${EXPECTED_PANGU_LEAD_MIN_HOURS:-12}"
EXPECTED_PANGU_LEAD_MAX_HOURS="${EXPECTED_PANGU_LEAD_MAX_HOURS:-23}"
INFER_PANGU_LEAD12_23_FROM_VALID_TIME="${INFER_PANGU_LEAD12_23_FROM_VALID_TIME:-1}"

TIANJI_DATA_DIR="${TIANJI_DATA_DIR:-${BASELINE_DIR}/ml_dataset_overlap_tianji_12h_pm10_pm25_source_full}"
IFS_DATA_DIR="${IFS_DATA_DIR:-${BASELINE_DIR}/ml_dataset_overlap_ifs_12h_pm10_pm25_source_full}"
ERA5_2025_DATA_DIR="${ERA5_2025_DATA_DIR:-${BASELINE_DIR}/ml_dataset_overlap_era5_2025_12h_pm10_pm25_source_full}"
CORRECTED_DATA_ROOT="${CORRECTED_DATA_ROOT:-${BASELINE_DIR}/corrected_pangu2025_q1000/${RUN_TAG}}"
PANGU2025_DATA_DIR="${PANGU2025_DATA_DIR:-${CORRECTED_DATA_ROOT}/source_full}"

OUT_ROOT="${OUT_ROOT:-${BASE}/paper_eval_results_pm10_pm25_journal/q1000_corrected_pangu2025/${RUN_TAG}}"
VERIFY_OUT_JSON="${VERIFY_OUT_JSON:-${OUT_ROOT}/station_product_verification.json}"
LINEAGE_OUT_DIR="${LINEAGE_OUT_DIR:-${OUT_ROOT}/lineage_audit}"
MECHANISM_OUT_DIR="${MECHANISM_OUT_DIR:-${OUT_ROOT}/mechanism_analysis}"
BOOTSTRAP_ITERS="${BOOTSTRAP_ITERS:-1000}"
LIMIT_SAMPLES="${LIMIT_SAMPLES:-}"

is_true() {
    [[ "$1" == "1" || "$1" == "true" || "$1" == "TRUE" ]]
}

if [[ ! "${RUN_TAG}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "ERROR: invalid RUN_TAG=${RUN_TAG}" >&2
    exit 2
fi

job_id_from_output() {
    local value="$1"
    value="${value%%;*}"
    value="${value//$'\n'/}"
    if [[ ! "${value}" =~ ^[0-9]+$ ]] && ! is_true "${DRY_RUN}"; then
        echo "ERROR: could not parse sbatch job id from '${1}'" >&2
        exit 2
    fi
    printf '%s\n' "${value}"
}

submit() {
    local label="$1"
    shift
    if is_true "${DRY_RUN}"; then
        printf '[DRY-RUN] sbatch --parsable' >&2
        printf ' %q' "$@" >&2
        printf '\n' >&2
        printf '9%s\n' "$(printf '%s' "${label}" | cksum | awk '{print $1}' | cut -c1-7)"
        return
    fi
    job_id_from_output "$(sbatch --parsable "$@")"
}

require_file() {
    local label="$1"
    local path="$2"
    [[ -s "${path}" ]] || { echo "ERROR: ${label} missing or empty: ${path}" >&2; exit 2; }
}

require_dataset() {
    local label="$1"
    local dir="$2"
    require_file "${label} config" "${dir}/dataset_build_config.json"
    local split
    for split in train val test; do
        require_file "${label} X_${split}" "${dir}/X_${split}.npy"
        require_file "${label} y_${split}" "${dir}/y_${split}.npy"
        require_file "${label} meta_${split}" "${dir}/meta_${split}.csv"
    done
}

if ! is_true "${DRY_RUN}"; then
    require_file "legacy Pangu station product" "${OLD_PANGU_STATION_FILE}"
    require_file "corrected Pangu station product" "${PANGU2025_STATION_FILE}"
    require_file "canonical target" "${TARGET_FILE}"
    require_dataset Tianji "${TIANJI_DATA_DIR}"
    require_dataset IFS "${IFS_DATA_DIR}"
    require_dataset ERA5 "${ERA5_2025_DATA_DIR}"
    for path in "${CORRECTED_DATA_ROOT}" "${OUT_ROOT}"; do
        if [[ -e "${path}" ]]; then
            echo "ERROR: RUN_TAG would reuse existing path: ${path}" >&2
            exit 2
        fi
    done
fi

echo "Corrected Pangu Q1000 data-check chain"
echo "RUN_TAG=${RUN_TAG}"
echo "PANGU2025_STATION_FILE=${PANGU2025_STATION_FILE}"
echo "PANGU2025_DATA_DIR=${PANGU2025_DATA_DIR}"
echo "EXPECTED_PANGU_LEAD=${EXPECTED_PANGU_LEAD_MIN_HOURS}..${EXPECTED_PANGU_LEAD_MAX_HOURS}h"
echo "OUT_ROOT=${OUT_ROOT}"

verify_job="$(submit verify_station \
    --export="ALL,OLD_PANGU_STATION_FILE=${OLD_PANGU_STATION_FILE},PANGU2025_STATION_FILE=${PANGU2025_STATION_FILE},TARGET_FILE=${TARGET_FILE},EXPECTED_LEAD_MIN_HOURS=${EXPECTED_PANGU_LEAD_MIN_HOURS},EXPECTED_LEAD_MAX_HOURS=${EXPECTED_PANGU_LEAD_MAX_HOURS},INFER_PANGU_LEAD12_23_FROM_VALID_TIME=${INFER_PANGU_LEAD12_23_FROM_VALID_TIME},VERIFY_OUT_JSON=${VERIFY_OUT_JSON}" \
    sub_verify_pangu_station_product.slurm)"

pangu_data_job="$(submit pangu_source_full_data \
    --dependency="afterok:${verify_job}" \
    --export="ALL,SOURCE_KIND=station_nc,SOURCE_TAG=pangu2025,YEAR=2025,FEATURE_SET=source_full,SOURCE_FILE=${PANGU2025_STATION_FILE},TARGET_FILE=${TARGET_FILE},OUT_DIR=${PANGU2025_DATA_DIR},EXPECTED_LEAD_MIN_HOURS=${EXPECTED_PANGU_LEAD_MIN_HOURS},EXPECTED_LEAD_MAX_HOURS=${EXPECTED_PANGU_LEAD_MAX_HOURS},INFER_PANGU_LEAD12_23_FROM_VALID_TIME=${INFER_PANGU_LEAD12_23_FROM_VALID_TIME}" \
    sub_station_source_overlap_data.slurm)"

lineage_job="$(submit lineage_audit \
    --dependency="afterok:${pangu_data_job}" \
    --export="ALL,OUT_DIR=${LINEAGE_OUT_DIR},PANGU_STATION_FILE=${PANGU2025_STATION_FILE},TIANJI_FILE=${TARGET_FILE},PANGU_DATASET_DIR=${PANGU2025_DATA_DIR},TIANJI_DATASET_DIR=${TIANJI_DATA_DIR},IFS_DATASET_DIR=${IFS_DATA_DIR},ERA5_DATASET_DIR=${ERA5_2025_DATA_DIR},EXPECTED_PANGU_LEAD_MIN_HOURS=${EXPECTED_PANGU_LEAD_MIN_HOURS},EXPECTED_PANGU_LEAD_MAX_HOURS=${EXPECTED_PANGU_LEAD_MAX_HOURS},INFER_PANGU_LEAD12_23_FROM_VALID_TIME=${INFER_PANGU_LEAD12_23_FROM_VALID_TIME},SKIP_PANGU_GRID=1,STRICT=1" \
    sub_q1000_lineage_audit.slurm)"

mechanism_export="ALL,FEATURE_SET=source_full,TIANJI_DATA_DIR=${TIANJI_DATA_DIR},IFS_DATA_DIR=${IFS_DATA_DIR},ERA5_2025_DATA_DIR=${ERA5_2025_DATA_DIR},PANGU2025_DATA_DIR=${PANGU2025_DATA_DIR},PANGU_SOURCE_LABEL=Pangu-2025 canonical,OUT_DIR=${MECHANISM_OUT_DIR},REQUIRE_CASE_CONTROL=0,DISTRIBUTION_ONLY=0,BOOTSTRAP_ITERS=${BOOTSTRAP_ITERS}"
if [[ -n "${LIMIT_SAMPLES}" ]]; then
    mechanism_export="${mechanism_export},LIMIT_SAMPLES=${LIMIT_SAMPLES}"
fi
mechanism_job="$(submit q1000_mechanism \
    --dependency="afterok:${lineage_job}" \
    --export="${mechanism_export}" \
    sub_q1000_mechanism_analysis.slurm)"

summary_path="logs/q1000_corrected_pangu_${RUN_TAG}_submission.txt"
{
    echo "run_tag=${RUN_TAG}"
    echo "corrected_pangu_station_file=${PANGU2025_STATION_FILE}"
    echo "corrected_pangu_source_full_data=${PANGU2025_DATA_DIR}"
    echo "expected_pangu_lead_hours=${EXPECTED_PANGU_LEAD_MIN_HOURS}..${EXPECTED_PANGU_LEAD_MAX_HOURS}"
    echo "verify_job=${verify_job}"
    echo "pangu_data_job=${pangu_data_job}"
    echo "lineage_audit_job=${lineage_job}"
    echo "q1000_mechanism_job=${mechanism_job}"
    echo "output_root=${OUT_ROOT}"
} | tee "${summary_path}"

echo "Submission manifest: ${summary_path}"
