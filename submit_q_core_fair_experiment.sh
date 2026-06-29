#!/bin/bash
# Submit the complete Pangu-2025 common-variable experiment with hard dependencies:
# data builds -> data audit -> S1 -> four S2 models -> paired argmax evaluation.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
mkdir -p logs

BASE="${BASE:-/public/home/putianshu/vis_mlp}"
BASELINE_DIR="${BASELINE_DIR:-${BASE}/ifs_baseline}"
CKPT_DIR="${CKPT_DIR:-${BASELINE_DIR}/checkpoints}"
FEATURE_SET="q_core_no_rh2m"
RUN_TAG="${RUN_TAG:-qcore_pangu2025_$(date +%Y%m%d_%H%M%S)}"
DRY_RUN="${DRY_RUN:-0}"
BUILD_PANGU_STATION="${BUILD_PANGU_STATION:-0}"
PANGU2025_STATION_FILE="${PANGU2025_STATION_FILE:-}"
ALLOW_EXISTING_RUN="${ALLOW_EXISTING_RUN:-0}"
ALLOW_UNVERIFIED_PANGU_LEAD="${ALLOW_UNVERIFIED_PANGU_LEAD:-0}"

case "${PANGU2025_STATION_FILE}" in
    /path/to/*|PATH_TO_*|REPLACE_ME*)
        echo "[WARN] Ignoring placeholder PANGU2025_STATION_FILE=${PANGU2025_STATION_FILE}; trying known lead24h locations." >&2
        PANGU2025_STATION_FILE=""
        ;;
esac

if [[ ! "${RUN_TAG}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "ERROR: RUN_TAG may contain only letters, digits, dot, underscore, and hyphen: ${RUN_TAG}" >&2
    exit 2
fi

S1_RUN_ID="exp_overlap_static_rnn_s1_q_core_no_rh2m_${RUN_TAG}_pm10_pm25"
TIANJI_RUN_ID="exp_overlap_static_rnn_s2_tianji_q_core_no_rh2m_${RUN_TAG}_pm10_pm25"
IFS_RUN_ID="exp_overlap_static_rnn_s2_ifs_q_core_no_rh2m_${RUN_TAG}_pm10_pm25"
PANGU2025_RUN_ID="exp_overlap_static_rnn_s2_pangu2025_q_core_no_rh2m_${RUN_TAG}_pm10_pm25"
ERA5_2025_RUN_ID="exp_overlap_static_rnn_s2_era5_2025_q_core_no_rh2m_${RUN_TAG}_pm10_pm25"
S1_CKPT="${CKPT_DIR}/${S1_RUN_ID}_S1_best_score.pt"

DATA_ROOT="${DATA_ROOT:-${BASELINE_DIR}/q_core_fair_datasets/${RUN_TAG}}"
TIANJI_DATA_DIR="${TIANJI_DATA_DIR:-${DATA_ROOT}/tianji}"
IFS_DATA_DIR="${IFS_DATA_DIR:-${DATA_ROOT}/ifs}"
PANGU2025_DATA_DIR="${PANGU2025_DATA_DIR:-${DATA_ROOT}/pangu2025}"
ERA5_2025_DATA_DIR="${ERA5_2025_DATA_DIR:-${DATA_ROOT}/era5_2025}"
S1_DATA_DIR="${S1_DATA_DIR:-${DATA_ROOT}/s1}"

is_true() {
    [[ "$1" == "1" || "$1" == "true" || "$1" == "TRUE" ]]
}

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
    local raw
    raw="$(sbatch --parsable "$@")"
    job_id_from_output "${raw}"
}

join_by_colon() {
    local IFS=:
    echo "$*"
}

find_pangu_station_file() {
    local candidates=(
        "${BASELINE_DIR}/pangu_station/pangu_station_2025_lead24h.nc"
        "/data2/share/chenxi/PuTS/mlp/ifs_baseline/pangu_station/pangu_station_2025_lead24h.nc"
    )
    local path
    for path in "${candidates[@]}"; do
        if [[ -s "${path}" ]]; then
            printf '%s\n' "${path}"
            return 0
        fi
    done
    return 1
}

if [[ -z "${PANGU2025_STATION_FILE}" ]] && ! is_true "${DRY_RUN}"; then
    PANGU2025_STATION_FILE="$(find_pangu_station_file || true)"
fi
if [[ -z "${PANGU2025_STATION_FILE}" ]]; then
    PANGU2025_STATION_FILE="${BASELINE_DIR}/pangu_station/pangu_station_2025_lead24h.nc"
fi

if ! is_true "${DRY_RUN}" && ! is_true "${ALLOW_EXISTING_RUN}"; then
    existing=()
    for path in \
        "${S1_CKPT}" \
        "${CKPT_DIR}/${TIANJI_RUN_ID}_S2_PhaseB_best_score.pt" \
        "${CKPT_DIR}/${IFS_RUN_ID}_S2_PhaseB_best_score.pt" \
        "${CKPT_DIR}/${PANGU2025_RUN_ID}_S2_PhaseB_best_score.pt" \
        "${CKPT_DIR}/${ERA5_2025_RUN_ID}_S2_PhaseB_best_score.pt" \
        "${DATA_ROOT}" \
        "${BASE}/paper_eval_results_pm10_pm25_journal/q_core_fair_pangu2025/${RUN_TAG}"
    do
        if [[ -e "${path}" ]]; then
            existing+=("${path}")
        fi
    done
    if (( ${#existing[@]} > 0 )); then
        printf 'ERROR: RUN_TAG=%s would reuse existing artifacts:\n' "${RUN_TAG}" >&2
        printf '  %s\n' "${existing[@]}" >&2
        echo "Choose a new RUN_TAG. Set ALLOW_EXISTING_RUN=1 only for an intentional resume." >&2
        exit 2
    fi
fi

echo "Pangu-2025 q-core fair experiment"
echo "RUN_TAG=${RUN_TAG}"
echo "FEATURE_SET=${FEATURE_SET}"
echo "PANGU2025_STATION_FILE=${PANGU2025_STATION_FILE}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "S1_RUN_ID=${S1_RUN_ID}"

pangu_station_dep=""
if is_true "${BUILD_PANGU_STATION}"; then
    pangu_station_dep="$(submit pangu_station \
        --export="ALL,OUT_FILE=${PANGU2025_STATION_FILE},YEAR=2025" \
        sub_pangu_station_idw.slurm)"
    echo "Pangu station interpolation: job ${pangu_station_dep}"
elif ! is_true "${DRY_RUN}" && [[ ! -s "${PANGU2025_STATION_FILE}" ]]; then
    echo "ERROR: Pangu-2025 station file is missing or empty: ${PANGU2025_STATION_FILE}" >&2
    echo "Set PANGU2025_STATION_FILE to the current lead24h station product, or set BUILD_PANGU_STATION=1." >&2
    exit 2
fi
if ! is_true "${DRY_RUN}" && ! is_true "${ALLOW_UNVERIFIED_PANGU_LEAD}"; then
    if [[ "$(basename "${PANGU2025_STATION_FILE}")" != *lead24h* ]]; then
        echo "ERROR: Pangu station filename does not identify the required lead24h product: ${PANGU2025_STATION_FILE}" >&2
        echo "Rename/regenerate it with lead24h provenance, or set ALLOW_UNVERIFIED_PANGU_LEAD=1 after manual metadata verification." >&2
        exit 2
    fi
fi

s1_data_job="$(submit s1_data \
    --export="ALL,FEATURE_SET=${FEATURE_SET},OUT_DIR=${S1_DATA_DIR}" \
    sub_s1_overlap_data.slurm)"
tianji_data_job="$(submit tianji_data \
    --export="ALL,FEATURE_SET=${FEATURE_SET},OUT_DIR=${TIANJI_DATA_DIR}" \
    sub_tianji_overlap_data.slurm)"
ifs_data_job="$(submit ifs_data \
    --export="ALL,FEATURE_SET=${FEATURE_SET},YEAR=2025,OUT_DIR=${IFS_DATA_DIR}" \
    sub_ifs_data.slurm)"

pangu_data_args=(
    --export="ALL,SOURCE_KIND=station_nc,SOURCE_TAG=pangu2025,YEAR=2025,FEATURE_SET=${FEATURE_SET},SOURCE_FILE=${PANGU2025_STATION_FILE},OUT_DIR=${PANGU2025_DATA_DIR}"
)
if [[ -n "${pangu_station_dep}" ]]; then
    pangu_data_args+=(--dependency="afterok:${pangu_station_dep}")
fi
pangu_data_args+=(sub_station_source_overlap_data.slurm)
pangu_data_job="$(submit pangu_data "${pangu_data_args[@]}")"

era5_data_job="$(submit era5_data \
    --export="ALL,SOURCE_KIND=era5_feature_dir,SOURCE_TAG=era5_2025,YEAR=2025,FEATURE_SET=${FEATURE_SET},OUT_DIR=${ERA5_2025_DATA_DIR}" \
    sub_station_source_overlap_data.slurm)"

data_deps="$(join_by_colon "${s1_data_job}" "${tianji_data_job}" "${ifs_data_job}" "${pangu_data_job}" "${era5_data_job}")"
audit_job="$(submit data_audit \
    --dependency="afterok:${data_deps}" \
    --export="ALL,RUN_TAG=${RUN_TAG},S1_DATA_DIR=${S1_DATA_DIR},TIANJI_DATA_DIR=${TIANJI_DATA_DIR},IFS_DATA_DIR=${IFS_DATA_DIR},PANGU2025_DATA_DIR=${PANGU2025_DATA_DIR},ERA5_2025_DATA_DIR=${ERA5_2025_DATA_DIR}" \
    sub_q_core_fair_data_audit.slurm)"

s1_train_job="$(submit s1_train \
    --dependency="afterok:${audit_job}" \
    --export="ALL,EXPERIMENT=s1_q_core_no_rh2m,MODEL_ARCH=static_rnn,LOWVIS_RNN_RUN_ID=${S1_RUN_ID},OVERLAP_S1_DATA_DIR=${S1_DATA_DIR},LOWVIS_RNN_CLEAN_LOCAL_CACHE=1,LOWVIS_RNN_LOCAL_CACHE_ID=${RUN_TAG}_s1" \
    sub_ifs_overlap_baseline.slurm)"

submit_s2() {
    local label="$1"
    local experiment="$2"
    local run_id="$3"
    local data_dir="$4"
    submit "${label}" \
        --dependency="afterok:${s1_train_job}" \
        --export="ALL,EXPERIMENT=${experiment},MODEL_ARCH=static_rnn,LOWVIS_RNN_RUN_ID=${run_id},OVERLAP_STATIC_RNN_PRETRAINED_CKPT=${S1_CKPT},OVERLAP_S2_DATA_DIR=${data_dir},LOWVIS_RNN_CLEAN_LOCAL_CACHE=1,LOWVIS_RNN_LOCAL_CACHE_ID=${RUN_TAG}_${label}" \
        sub_ifs_overlap_baseline.slurm
}

tianji_train_job="$(submit_s2 tianji_s2 s2_tianji_q_core_no_rh2m "${TIANJI_RUN_ID}" "${TIANJI_DATA_DIR}")"
ifs_train_job="$(submit_s2 ifs_s2 s2_ifs_q_core_no_rh2m "${IFS_RUN_ID}" "${IFS_DATA_DIR}")"
pangu_train_job="$(submit_s2 pangu2025_s2 s2_pangu2025_q_core_no_rh2m "${PANGU2025_RUN_ID}" "${PANGU2025_DATA_DIR}")"
era5_train_job="$(submit_s2 era5_2025_s2 s2_era5_2025_q_core_no_rh2m "${ERA5_2025_RUN_ID}" "${ERA5_2025_DATA_DIR}")"

s2_deps="$(join_by_colon "${tianji_train_job}" "${ifs_train_job}" "${pangu_train_job}" "${era5_train_job}")"
eval_job="$(submit paired_eval \
    --dependency="afterok:${s2_deps}" \
    --export="ALL,RUN_TAG=${RUN_TAG},S1_RUN_ID=${S1_RUN_ID},TIANJI_RUN_ID=${TIANJI_RUN_ID},IFS_RUN_ID=${IFS_RUN_ID},PANGU2025_RUN_ID=${PANGU2025_RUN_ID},ERA5_2025_RUN_ID=${ERA5_2025_RUN_ID},TIANJI_DATA_DIR=${TIANJI_DATA_DIR},IFS_DATA_DIR=${IFS_DATA_DIR},PANGU2025_DATA_DIR=${PANGU2025_DATA_DIR},ERA5_2025_DATA_DIR=${ERA5_2025_DATA_DIR}" \
    sub_static_rnn_q_core_fair_eval.slurm)"

summary_path="logs/q_core_fair_${RUN_TAG}_submission.txt"
{
    echo "experiment_status=scheduled"
    echo "run_tag=${RUN_TAG}"
    echo "feature_set=${FEATURE_SET}"
    echo "threshold_mode=argmax"
    echo "sample_scope=four_source_paired_test_intersection"
    echo "controlled_dimension=common_input_layout"
    echo "lead_time_caveat=pangu_lead24h_vs_tianji_ifs_12_to_lt24h"
    echo "era5_role=reference_analysis"
    echo "pangu_station_file=${PANGU2025_STATION_FILE}"
    echo "data_root=${DATA_ROOT}"
    echo "data_jobs=${data_deps}"
    echo "audit_job=${audit_job}"
    echo "s1_train_job=${s1_train_job}"
    echo "s2_train_jobs=${s2_deps}"
    echo "eval_job=${eval_job}"
    echo "s1_run_id=${S1_RUN_ID}"
    echo "tianji_run_id=${TIANJI_RUN_ID}"
    echo "ifs_run_id=${IFS_RUN_ID}"
    echo "pangu2025_run_id=${PANGU2025_RUN_ID}"
    echo "era5_2025_run_id=${ERA5_2025_RUN_ID}"
} | tee "${summary_path}"

echo "Submission manifest: ${summary_path}"
echo "The evaluation job will start only if every data, audit, S1, and S2 dependency succeeds."
