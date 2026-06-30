#!/bin/bash
# Corrected Pangu-2025 workflow:
# verification -> q-core/source-full data -> q-core audit -> fair and best-effort training/evaluation.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
mkdir -p logs

BASE="${BASE:-/public/home/putianshu/vis_mlp}"
BASELINE_DIR="${BASELINE_DIR:-${BASE}/ifs_baseline}"
CKPT_DIR="${CKPT_DIR:-${BASELINE_DIR}/checkpoints}"
RUN_TAG="${RUN_TAG:-pangu2025_canonical_$(date +%Y%m%d_%H%M%S)}"
DRY_RUN="${DRY_RUN:-0}"

OLD_PANGU_STATION_FILE="${OLD_PANGU_STATION_FILE:-${BASELINE_DIR}/pangu_station/pangu_station_2025_lead12_23h.nc}"
PANGU2025_STATION_FILE="${PANGU2025_STATION_FILE:-${BASELINE_DIR}/pangu_station/pangu_station_2025_lead12_23h_canonical.nc}"
TARGET_FILE="${TARGET_FILE:-${BASE}/tianji_auto_station/merged_final_all_vars.nc}"
EXPECTED_PANGU_LEAD_MIN_HOURS="${EXPECTED_PANGU_LEAD_MIN_HOURS:-12}"
EXPECTED_PANGU_LEAD_MAX_HOURS="${EXPECTED_PANGU_LEAD_MAX_HOURS:-23}"
INFER_PANGU_LEAD12_23_FROM_VALID_TIME="${INFER_PANGU_LEAD12_23_FROM_VALID_TIME:-1}"

REUSED_QCORE_DATA_ROOT="${REUSED_QCORE_DATA_ROOT:-${BASELINE_DIR}/q_core_fair_datasets/qcore_pangu2025_rerun_20260629}"
CORRECTED_DATA_ROOT="${CORRECTED_DATA_ROOT:-${BASELINE_DIR}/corrected_pangu2025_datasets/${RUN_TAG}}"
S1_QCORE_DATA_DIR="${S1_QCORE_DATA_DIR:-${REUSED_QCORE_DATA_ROOT}/s1}"
TIANJI_QCORE_DATA_DIR="${TIANJI_QCORE_DATA_DIR:-${REUSED_QCORE_DATA_ROOT}/tianji}"
IFS_QCORE_DATA_DIR="${IFS_QCORE_DATA_DIR:-${REUSED_QCORE_DATA_ROOT}/ifs}"
ERA5_QCORE_DATA_DIR="${ERA5_QCORE_DATA_DIR:-${REUSED_QCORE_DATA_ROOT}/era5_2025}"
PANGU_QCORE_DATA_DIR="${PANGU_QCORE_DATA_DIR:-${CORRECTED_DATA_ROOT}/q_core_no_rh2m}"
PANGU_SOURCE_FULL_DATA_DIR="${PANGU_SOURCE_FULL_DATA_DIR:-${CORRECTED_DATA_ROOT}/source_full}"

FAIR_EVAL_ROOT="${FAIR_EVAL_ROOT:-${BASE}/paper_eval_results_pm10_pm25_journal/q_core_fair_pangu2025/${RUN_TAG}}"
BEST_EVAL_ROOT="${BEST_EVAL_ROOT:-${BASE}/paper_eval_results_pm10_pm25_journal/best_effort_source_full_argmax/${RUN_TAG}}"
VERIFY_OUT_JSON="${VERIFY_OUT_JSON:-${FAIR_EVAL_ROOT}/station_product_verification.json}"

FAIR_S1_RUN_ID="exp_overlap_static_rnn_s1_q_core_no_rh2m_${RUN_TAG}_pm10_pm25"
FAIR_TIANJI_RUN_ID="exp_overlap_static_rnn_s2_tianji_q_core_no_rh2m_${RUN_TAG}_pm10_pm25"
FAIR_IFS_RUN_ID="exp_overlap_static_rnn_s2_ifs_q_core_no_rh2m_${RUN_TAG}_pm10_pm25"
FAIR_PANGU_RUN_ID="exp_overlap_static_rnn_s2_pangu2025_q_core_no_rh2m_${RUN_TAG}_pm10_pm25"
FAIR_ERA5_RUN_ID="exp_overlap_static_rnn_s2_era5_2025_q_core_no_rh2m_${RUN_TAG}_pm10_pm25"
FAIR_S1_CKPT="${CKPT_DIR}/${FAIR_S1_RUN_ID}_S1_best_score.pt"

BEST_PANGU_RUN_ID="exp_overlap_static_rnn_s2_pangu2025_source_full_${RUN_TAG}_pm10_pm25"
BEST_PANGU_S1_CKPT="${BEST_PANGU_S1_CKPT:-${CKPT_DIR}/exp_overlap_static_rnn_s1_source_full_pangu2025_dyn19_pm10_pm25_S1_best_score.pt}"
BEST_PANGU_CKPT="${CKPT_DIR}/${BEST_PANGU_RUN_ID}_S2_PhaseB_best_score.pt"

TIANJI_SOURCE_FULL_DATA_DIR="${TIANJI_SOURCE_FULL_DATA_DIR:-${BASELINE_DIR}/ml_dataset_overlap_tianji_12h_pm10_pm25_source_full}"
IFS_SOURCE_FULL_DATA_DIR="${IFS_SOURCE_FULL_DATA_DIR:-${BASELINE_DIR}/ml_dataset_overlap_ifs_12h_pm10_pm25_source_full}"
T2ND_SOURCE_FULL_DATA_DIR="${T2ND_SOURCE_FULL_DATA_DIR:-${BASELINE_DIR}/ml_dataset_overlap_tianji_12h_pm10_pm25_T2ND_rh2m_source_full}"
ERA5_SOURCE_FULL_DATA_DIR="${ERA5_SOURCE_FULL_DATA_DIR:-${BASELINE_DIR}/ml_dataset_overlap_era5_2025_12h_pm10_pm25_source_full}"
TIANJI_SOURCE_FULL_CKPT="${TIANJI_SOURCE_FULL_CKPT:-${CKPT_DIR}/exp_overlap_static_rnn_s2_tianji_source_full_pm10_pm25_S2_PhaseB_best_score.pt}"
IFS_SOURCE_FULL_CKPT="${IFS_SOURCE_FULL_CKPT:-${CKPT_DIR}/exp_overlap_static_rnn_s2_ifs_source_full_pm10_pm25_S2_PhaseB_best_score.pt}"
T2ND_SOURCE_FULL_CKPT="${T2ND_SOURCE_FULL_CKPT:-${CKPT_DIR}/exp_overlap_static_rnn_s2_T2ND_rh2m_source_full_pm10_pm25_S2_PhaseB_best_score.pt}"
ERA5_SOURCE_FULL_CKPT="${ERA5_SOURCE_FULL_CKPT:-${CKPT_DIR}/exp_overlap_static_rnn_s2_era5_2025_source_full_pm10_pm25_S2_PhaseB_best_score.pt}"

is_true() {
    [[ "$1" == "1" || "$1" == "true" || "$1" == "TRUE" ]]
}

if [[ ! "${RUN_TAG}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "ERROR: invalid RUN_TAG=${RUN_TAG}" >&2
    exit 2
fi
if ! awk -v lo="${EXPECTED_PANGU_LEAD_MIN_HOURS}" -v hi="${EXPECTED_PANGU_LEAD_MAX_HOURS}" 'BEGIN { exit !(lo <= hi) }'; then
    echo "ERROR: expected Pangu lead minimum exceeds maximum." >&2
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
    local raw
    raw="$(sbatch --parsable "$@")"
    job_id_from_output "${raw}"
}

require_file() {
    local label="$1"
    local path="$2"
    [[ -s "${path}" ]] || { echo "ERROR: ${label} is missing or empty: ${path}" >&2; exit 2; }
}

require_dataset() {
    local label="$1"
    local dir="$2"
    local meta="$3"
    shift 3
    require_file "${label} config" "${dir}/dataset_build_config.json"
    local split
    for split in "$@"; do
        require_file "${label} X_${split}" "${dir}/X_${split}.npy"
        require_file "${label} y_${split}" "${dir}/y_${split}.npy"
        if is_true "${meta}"; then
            require_file "${label} meta_${split}" "${dir}/meta_${split}.csv"
        fi
    done
}

if ! is_true "${DRY_RUN}"; then
    require_file "legacy Pangu station product" "${OLD_PANGU_STATION_FILE}"
    require_file "corrected Pangu station product" "${PANGU2025_STATION_FILE}"
    require_file "canonical target station product" "${TARGET_FILE}"

    require_dataset qcore_s1 "${S1_QCORE_DATA_DIR}" 0 train val
    require_dataset qcore_tianji "${TIANJI_QCORE_DATA_DIR}" 1 train val test
    require_dataset qcore_ifs "${IFS_QCORE_DATA_DIR}" 1 train val test
    require_dataset qcore_era5 "${ERA5_QCORE_DATA_DIR}" 1 train val test

    require_file "Pangu source-full S1 checkpoint" "${BEST_PANGU_S1_CKPT}"
    require_dataset source_full_tianji "${TIANJI_SOURCE_FULL_DATA_DIR}" 1 train val test
    require_dataset source_full_ifs "${IFS_SOURCE_FULL_DATA_DIR}" 1 train val test
    require_dataset source_full_t2nd "${T2ND_SOURCE_FULL_DATA_DIR}" 1 train val test
    require_dataset source_full_era5 "${ERA5_SOURCE_FULL_DATA_DIR}" 1 train val test
    require_file "Tianji source-full checkpoint" "${TIANJI_SOURCE_FULL_CKPT}"
    require_file "IFS source-full checkpoint" "${IFS_SOURCE_FULL_CKPT}"
    require_file "T2ND source-full checkpoint" "${T2ND_SOURCE_FULL_CKPT}"
    require_file "ERA5 source-full checkpoint" "${ERA5_SOURCE_FULL_CKPT}"

    collisions=()
    for path in \
        "${CORRECTED_DATA_ROOT}" \
        "${FAIR_S1_CKPT}" \
        "${CKPT_DIR}/${FAIR_TIANJI_RUN_ID}_S2_PhaseB_best_score.pt" \
        "${CKPT_DIR}/${FAIR_IFS_RUN_ID}_S2_PhaseB_best_score.pt" \
        "${CKPT_DIR}/${FAIR_PANGU_RUN_ID}_S2_PhaseB_best_score.pt" \
        "${CKPT_DIR}/${FAIR_ERA5_RUN_ID}_S2_PhaseB_best_score.pt" \
        "${BEST_PANGU_CKPT}"
    do
        [[ -e "${path}" ]] && collisions+=("${path}")
    done
    if (( ${#collisions[@]} > 0 )); then
        printf 'ERROR: RUN_TAG=%s would overwrite/reuse artifacts:\n' "${RUN_TAG}" >&2
        printf '  %s\n' "${collisions[@]}" >&2
        echo "Choose a fresh RUN_TAG." >&2
        exit 2
    fi
fi

echo "Corrected Pangu-2025 experiment chain"
echo "RUN_TAG=${RUN_TAG}"
echo "OLD_PANGU_STATION_FILE=${OLD_PANGU_STATION_FILE}"
echo "PANGU2025_STATION_FILE=${PANGU2025_STATION_FILE}"
echo "EXPECTED_PANGU_LEAD=${EXPECTED_PANGU_LEAD_MIN_HOURS}..${EXPECTED_PANGU_LEAD_MAX_HOURS}h"
echo "REUSED_QCORE_DATA_ROOT=${REUSED_QCORE_DATA_ROOT}"
echo "CORRECTED_DATA_ROOT=${CORRECTED_DATA_ROOT}"

verify_job="$(submit verify_station \
    --export="ALL,OLD_PANGU_STATION_FILE=${OLD_PANGU_STATION_FILE},PANGU2025_STATION_FILE=${PANGU2025_STATION_FILE},TARGET_FILE=${TARGET_FILE},EXPECTED_LEAD_MIN_HOURS=${EXPECTED_PANGU_LEAD_MIN_HOURS},EXPECTED_LEAD_MAX_HOURS=${EXPECTED_PANGU_LEAD_MAX_HOURS},INFER_PANGU_LEAD12_23_FROM_VALID_TIME=${INFER_PANGU_LEAD12_23_FROM_VALID_TIME},VERIFY_OUT_JSON=${VERIFY_OUT_JSON}" \
    sub_verify_pangu_station_product.slurm)"

qcore_data_job="$(submit pangu_qcore_data \
    --dependency="afterok:${verify_job}" \
    --export="ALL,SOURCE_KIND=station_nc,SOURCE_TAG=pangu2025,YEAR=2025,FEATURE_SET=q_core_no_rh2m,SOURCE_FILE=${PANGU2025_STATION_FILE},TARGET_FILE=${TARGET_FILE},OUT_DIR=${PANGU_QCORE_DATA_DIR},EXPECTED_LEAD_MIN_HOURS=${EXPECTED_PANGU_LEAD_MIN_HOURS},EXPECTED_LEAD_MAX_HOURS=${EXPECTED_PANGU_LEAD_MAX_HOURS},INFER_PANGU_LEAD12_23_FROM_VALID_TIME=${INFER_PANGU_LEAD12_23_FROM_VALID_TIME}" \
    sub_station_source_overlap_data.slurm)"

source_full_data_job="$(submit pangu_source_full_data \
    --dependency="afterok:${verify_job}" \
    --export="ALL,SOURCE_KIND=station_nc,SOURCE_TAG=pangu2025,YEAR=2025,FEATURE_SET=source_full,SOURCE_FILE=${PANGU2025_STATION_FILE},TARGET_FILE=${TARGET_FILE},OUT_DIR=${PANGU_SOURCE_FULL_DATA_DIR},EXPECTED_LEAD_MIN_HOURS=${EXPECTED_PANGU_LEAD_MIN_HOURS},EXPECTED_LEAD_MAX_HOURS=${EXPECTED_PANGU_LEAD_MAX_HOURS},INFER_PANGU_LEAD12_23_FROM_VALID_TIME=${INFER_PANGU_LEAD12_23_FROM_VALID_TIME}" \
    sub_station_source_overlap_data.slurm)"

audit_job="$(submit qcore_audit \
    --dependency="afterok:${qcore_data_job}" \
    --export="ALL,RUN_TAG=${RUN_TAG},S1_DATA_DIR=${S1_QCORE_DATA_DIR},TIANJI_DATA_DIR=${TIANJI_QCORE_DATA_DIR},IFS_DATA_DIR=${IFS_QCORE_DATA_DIR},PANGU2025_DATA_DIR=${PANGU_QCORE_DATA_DIR},ERA5_2025_DATA_DIR=${ERA5_QCORE_DATA_DIR},AUDIT_OUT_DIR=${FAIR_EVAL_ROOT}/data_audit,EXPECTED_PANGU_LEAD_MIN_HOURS=${EXPECTED_PANGU_LEAD_MIN_HOURS},EXPECTED_PANGU_LEAD_MAX_HOURS=${EXPECTED_PANGU_LEAD_MAX_HOURS}" \
    sub_q_core_fair_data_audit.slurm)"

fair_s1_job="$(submit fair_s1 \
    --dependency="afterok:${audit_job}" \
    --export="ALL,EXPERIMENT=s1_q_core_no_rh2m,MODEL_ARCH=static_rnn,LOWVIS_RNN_RUN_ID=${FAIR_S1_RUN_ID},OVERLAP_S1_DATA_DIR=${S1_QCORE_DATA_DIR},LOWVIS_RNN_CLEAN_LOCAL_CACHE=1,LOWVIS_RNN_LOCAL_CACHE_ID=${RUN_TAG}_fair_s1" \
    sub_ifs_overlap_baseline.slurm)"

submit_fair_s2() {
    local label="$1"
    local experiment="$2"
    local run_id="$3"
    local data_dir="$4"
    submit "${label}" \
        --dependency="afterok:${fair_s1_job}" \
        --export="ALL,EXPERIMENT=${experiment},MODEL_ARCH=static_rnn,LOWVIS_RNN_RUN_ID=${run_id},OVERLAP_STATIC_RNN_PRETRAINED_CKPT=${FAIR_S1_CKPT},OVERLAP_S2_DATA_DIR=${data_dir},LOWVIS_RNN_CLEAN_LOCAL_CACHE=1,LOWVIS_RNN_LOCAL_CACHE_ID=${RUN_TAG}_${label}" \
        sub_ifs_overlap_baseline.slurm
}

fair_tianji_job="$(submit_fair_s2 fair_tianji_s2 s2_tianji_q_core_no_rh2m "${FAIR_TIANJI_RUN_ID}" "${TIANJI_QCORE_DATA_DIR}")"
fair_ifs_job="$(submit_fair_s2 fair_ifs_s2 s2_ifs_q_core_no_rh2m "${FAIR_IFS_RUN_ID}" "${IFS_QCORE_DATA_DIR}")"
fair_pangu_job="$(submit_fair_s2 fair_pangu_s2 s2_pangu2025_q_core_no_rh2m "${FAIR_PANGU_RUN_ID}" "${PANGU_QCORE_DATA_DIR}")"
fair_era5_job="$(submit_fair_s2 fair_era5_s2 s2_era5_2025_q_core_no_rh2m "${FAIR_ERA5_RUN_ID}" "${ERA5_QCORE_DATA_DIR}")"
fair_s2_deps="${fair_tianji_job}:${fair_ifs_job}:${fair_pangu_job}:${fair_era5_job}"

fair_eval_job="$(submit fair_eval \
    --dependency="afterok:${fair_s2_deps}" \
    --export="ALL,RUN_TAG=${RUN_TAG},OUT_DIR=${FAIR_EVAL_ROOT}/argmax_paired,S1_RUN_ID=${FAIR_S1_RUN_ID},TIANJI_RUN_ID=${FAIR_TIANJI_RUN_ID},IFS_RUN_ID=${FAIR_IFS_RUN_ID},PANGU2025_RUN_ID=${FAIR_PANGU_RUN_ID},ERA5_2025_RUN_ID=${FAIR_ERA5_RUN_ID},TIANJI_DATA_DIR=${TIANJI_QCORE_DATA_DIR},IFS_DATA_DIR=${IFS_QCORE_DATA_DIR},PANGU2025_DATA_DIR=${PANGU_QCORE_DATA_DIR},ERA5_2025_DATA_DIR=${ERA5_QCORE_DATA_DIR}" \
    sub_static_rnn_q_core_fair_eval.slurm)"

best_train_job="$(submit best_pangu_s2 \
    --dependency="afterok:${audit_job}:${source_full_data_job}" \
    --export="ALL,EXPERIMENT=s2_pangu2025_source_full,MODEL_ARCH=static_rnn,LOWVIS_RNN_RUN_ID=${BEST_PANGU_RUN_ID},OVERLAP_STATIC_RNN_PRETRAINED_CKPT=${BEST_PANGU_S1_CKPT},OVERLAP_S2_DATA_DIR=${PANGU_SOURCE_FULL_DATA_DIR},LOWVIS_RNN_CLEAN_LOCAL_CACHE=1,LOWVIS_RNN_LOCAL_CACHE_ID=${RUN_TAG}_best_pangu_s2" \
    sub_ifs_overlap_baseline.slurm)"

best_eval_job="$(submit best_effort_eval \
    --dependency="afterok:${best_train_job}" \
    --export="ALL,EVAL_SCENARIO=figure1_all_sources,OUT_ROOT=${BEST_EVAL_ROOT},TIANJI_DATA_DIR=${TIANJI_SOURCE_FULL_DATA_DIR},IFS_DATA_DIR=${IFS_SOURCE_FULL_DATA_DIR},T2ND_DATA_DIR=${T2ND_SOURCE_FULL_DATA_DIR},PANGU2025_DATA_DIR=${PANGU_SOURCE_FULL_DATA_DIR},ERA5_2025_DATA_DIR=${ERA5_SOURCE_FULL_DATA_DIR},TIANJI_CKPT=${TIANJI_SOURCE_FULL_CKPT},IFS_CKPT=${IFS_SOURCE_FULL_CKPT},T2ND_CKPT=${T2ND_SOURCE_FULL_CKPT},PANGU2025_CKPT=${BEST_PANGU_CKPT},ERA5_2025_CKPT=${ERA5_SOURCE_FULL_CKPT},THRESHOLD_MODE=argmax,STRICT_META=1" \
    sub_static_rnn_source_full_argmax_eval.slurm)"

summary_path="logs/corrected_pangu2025_${RUN_TAG}_submission.txt"
{
    echo "experiment_status=scheduled"
    echo "run_tag=${RUN_TAG}"
    echo "old_pangu_station_file=${OLD_PANGU_STATION_FILE}"
    echo "corrected_pangu_station_file=${PANGU2025_STATION_FILE}"
    echo "canonical_target_file=${TARGET_FILE}"
    echo "expected_pangu_lead_hours=${EXPECTED_PANGU_LEAD_MIN_HOURS}..${EXPECTED_PANGU_LEAD_MAX_HOURS}"
    echo "lead_provenance_requirement=metadata_or_explicit_stitched_schedule"
    echo "infer_pangu_lead12_23_from_valid_time=${INFER_PANGU_LEAD12_23_FROM_VALID_TIME}"
    echo "reused_qcore_data_root=${REUSED_QCORE_DATA_ROOT}"
    echo "corrected_data_root=${CORRECTED_DATA_ROOT}"
    echo "verify_job=${verify_job}"
    echo "qcore_pangu_data_job=${qcore_data_job}"
    echo "source_full_pangu_data_job=${source_full_data_job}"
    echo "qcore_audit_job=${audit_job}"
    echo "fair_s1_job=${fair_s1_job}"
    echo "fair_s2_jobs=${fair_s2_deps}"
    echo "fair_eval_job=${fair_eval_job}"
    echo "best_effort_pangu_s2_job=${best_train_job}"
    echo "best_effort_eval_job=${best_eval_job}"
    echo "fair_s1_run_id=${FAIR_S1_RUN_ID}"
    echo "fair_tianji_run_id=${FAIR_TIANJI_RUN_ID}"
    echo "fair_ifs_run_id=${FAIR_IFS_RUN_ID}"
    echo "fair_pangu_run_id=${FAIR_PANGU_RUN_ID}"
    echo "fair_era5_run_id=${FAIR_ERA5_RUN_ID}"
    echo "best_pangu_run_id=${BEST_PANGU_RUN_ID}"
    echo "best_pangu_s1_checkpoint_reused=${BEST_PANGU_S1_CKPT}"
} | tee "${summary_path}"

echo "Submission manifest: ${summary_path}"
echo "No data build or training job can start unless its upstream verification/audit succeeds."
