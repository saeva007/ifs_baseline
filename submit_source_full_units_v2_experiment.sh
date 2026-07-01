#!/bin/bash
# Canonical-unit source-full rebuild -> audit -> all S1/S2 training -> argmax evaluation.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
mkdir -p logs

BASE="${BASE:-/public/home/putianshu/vis_mlp}"
BASELINE_DIR="${BASELINE_DIR:-${BASE}/ifs_baseline}"
CKPT_DIR="${CKPT_DIR:-${BASELINE_DIR}/checkpoints}"
RUN_TAG="${RUN_TAG:-source_full_units_v2_$(date +%Y%m%d_%H%M%S)}"
DRY_RUN="${DRY_RUN:-0}"
RESUME_FROM_AUDIT="${RESUME_FROM_AUDIT:-0}"
DATA_ROOT="${DATA_ROOT:-${BASELINE_DIR}/source_full_units_v2_datasets/${RUN_TAG}}"
EVAL_ROOT="${EVAL_ROOT:-${BASE}/paper_eval_results_pm10_pm25_journal/best_effort_source_full_argmax/${RUN_TAG}}"

OLD_PANGU_STATION_FILE="${OLD_PANGU_STATION_FILE:-${BASELINE_DIR}/pangu_station/pangu_station_2025_lead12_23h.nc}"
PANGU2025_STATION_FILE="${PANGU2025_STATION_FILE:-${BASELINE_DIR}/pangu_station/pangu_station_2025_lead12_23h_canonical.nc}"
TARGET_FILE="${TARGET_FILE:-${BASE}/tianji_auto_station/merged_final_all_vars.nc}"
T2ND_RH2M_FILE="${T2ND_RH2M_FILE:-${BASELINE_DIR}/tianji_rh2m_station/T2ND_rh2m_station_2025.nc}"
EXPECTED_PANGU_LEAD_MIN_HOURS="${EXPECTED_PANGU_LEAD_MIN_HOURS:-12}"
EXPECTED_PANGU_LEAD_MAX_HOURS="${EXPECTED_PANGU_LEAD_MAX_HOURS:-23}"

S1_TIANJI_DIR="${DATA_ROOT}/s1_tianji_dyn27"
S1_IFS_DIR="${DATA_ROOT}/s1_ifs_dyn24"
S1_PANGU_DIR="${DATA_ROOT}/s1_pangu2025_dyn19"
TIANJI_DIR="${DATA_ROOT}/tianji"
T2ND_DIR="${DATA_ROOT}/t2nd"
IFS_DIR="${DATA_ROOT}/ifs"
PANGU_DIR="${DATA_ROOT}/pangu2025"
ERA5_DIR="${DATA_ROOT}/era5_2025"
AUDIT_DIR="${EVAL_ROOT}/data_audit"
VERIFY_JSON="${EVAL_ROOT}/station_product_verification.json"

S1_TIANJI_RUN_ID="exp_overlap_static_rnn_s1_source_full_tianji_dyn27_${RUN_TAG}_pm10_pm25"
S1_IFS_RUN_ID="exp_overlap_static_rnn_s1_source_full_ifs_dyn24_${RUN_TAG}_pm10_pm25"
S1_PANGU_RUN_ID="exp_overlap_static_rnn_s1_source_full_pangu2025_dyn19_${RUN_TAG}_pm10_pm25"
TIANJI_RUN_ID="exp_overlap_static_rnn_s2_tianji_source_full_${RUN_TAG}_pm10_pm25"
T2ND_RUN_ID="exp_overlap_static_rnn_s2_T2ND_rh2m_source_full_${RUN_TAG}_pm10_pm25"
IFS_RUN_ID="exp_overlap_static_rnn_s2_ifs_source_full_${RUN_TAG}_pm10_pm25"
PANGU_RUN_ID="exp_overlap_static_rnn_s2_pangu2025_source_full_${RUN_TAG}_pm10_pm25"
ERA5_RUN_ID="exp_overlap_static_rnn_s2_era5_2025_source_full_${RUN_TAG}_pm10_pm25"
S1_TIANJI_CKPT="${CKPT_DIR}/${S1_TIANJI_RUN_ID}_S1_best_score.pt"
S1_IFS_CKPT="${CKPT_DIR}/${S1_IFS_RUN_ID}_S1_best_score.pt"
S1_PANGU_CKPT="${CKPT_DIR}/${S1_PANGU_RUN_ID}_S1_best_score.pt"

is_true() { [[ "$1" == "1" || "$1" == "true" || "$1" == "TRUE" ]]; }

submit() {
    local label="$1"
    shift
    if is_true "${DRY_RUN}"; then
        printf '[DRY-RUN] sbatch --parsable' >&2
        printf ' %q' "$@" >&2
        printf '\n' >&2
        printf '9%s\n' "$(printf '%s' "${label}" | cksum | awk '{print $1}' | cut -c1-7)"
    else
        local value
        value="$(sbatch --parsable "$@")"
        value="${value%%;*}"
        [[ "${value}" =~ ^[0-9]+$ ]] || { echo "ERROR: invalid sbatch id: ${value}" >&2; exit 2; }
        printf '%s\n' "${value}"
    fi
}

join_colon() { local IFS=:; echo "$*"; }
require_file() { [[ -s "$2" ]] || { echo "ERROR: $1 missing or empty: $2" >&2; exit 2; }; }

require_dataset_files() {
    local label="$1" dir="$2" require_meta="$3"
    shift 3
    local split path
    require_file "${label} dataset config" "${dir}/dataset_build_config.json"
    for split in "$@"; do
        require_file "${label} X_${split}" "${dir}/X_${split}.npy"
        require_file "${label} y_${split}" "${dir}/y_${split}.npy"
        if is_true "${require_meta}"; then
            path="${dir}/meta_${split}.csv"
            require_file "${label} meta_${split}" "${path}"
        fi
    done
}

if [[ ! "${RUN_TAG}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "ERROR: invalid RUN_TAG=${RUN_TAG}" >&2
    exit 2
fi
if ! is_true "${DRY_RUN}"; then
    if is_true "${RESUME_FROM_AUDIT}"; then
        require_dataset_files s1_tianji "${S1_TIANJI_DIR}" 0 train val
        require_dataset_files s1_ifs "${S1_IFS_DIR}" 0 train val
        require_dataset_files s1_pangu2025 "${S1_PANGU_DIR}" 0 train val
        require_dataset_files tianji "${TIANJI_DIR}" 1 train val test
        require_dataset_files t2nd "${T2ND_DIR}" 1 train val test
        require_dataset_files ifs "${IFS_DIR}" 1 train val test
        require_dataset_files pangu2025 "${PANGU_DIR}" 1 train val test
        require_dataset_files era5_2025 "${ERA5_DIR}" 1 train val test
    else
        require_file "old Pangu station product" "${OLD_PANGU_STATION_FILE}"
        require_file "canonical Pangu station product" "${PANGU2025_STATION_FILE}"
        require_file "canonical target" "${TARGET_FILE}"
        require_file "T2ND RH2M station product" "${T2ND_RH2M_FILE}"
        [[ ! -e "${DATA_ROOT}" ]] || { echo "ERROR: DATA_ROOT exists: ${DATA_ROOT}" >&2; exit 2; }
    fi
    for ckpt in \
        "${S1_TIANJI_CKPT}" "${S1_IFS_CKPT}" "${S1_PANGU_CKPT}" \
        "${CKPT_DIR}/${TIANJI_RUN_ID}_S2_PhaseB_best_score.pt" \
        "${CKPT_DIR}/${T2ND_RUN_ID}_S2_PhaseB_best_score.pt" \
        "${CKPT_DIR}/${IFS_RUN_ID}_S2_PhaseB_best_score.pt" \
        "${CKPT_DIR}/${PANGU_RUN_ID}_S2_PhaseB_best_score.pt" \
        "${CKPT_DIR}/${ERA5_RUN_ID}_S2_PhaseB_best_score.pt"
    do
        [[ ! -e "${ckpt}" ]] || { echo "ERROR: checkpoint exists: ${ckpt}" >&2; exit 2; }
    done
fi

echo "Canonical source-full experiment RUN_TAG=${RUN_TAG}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "RESUME_FROM_AUDIT=${RESUME_FROM_AUDIT}"

sources="tianji=${TIANJI_DIR};t2nd=${T2ND_DIR};ifs=${IFS_DIR};pangu2025=${PANGU_DIR};era5_2025=${ERA5_DIR}"
s1_profiles="tianji=${S1_TIANJI_DIR};ifs=${S1_IFS_DIR};pangu2025=${S1_PANGU_DIR}"
if is_true "${RESUME_FROM_AUDIT}"; then
    verify_job="reused_existing_verification"
    all_data_deps="reused_existing_datasets"
    audit_job="$(submit data_audit --export="ALL,SOURCES=${sources},S1_PROFILES=${s1_profiles},AUDIT_OUT_DIR=${AUDIT_DIR}" sub_source_full_canonical_audit.slurm)"
else
    verify_job="$(submit verify_station \
        --export="ALL,OLD_PANGU_STATION_FILE=${OLD_PANGU_STATION_FILE},PANGU2025_STATION_FILE=${PANGU2025_STATION_FILE},TARGET_FILE=${TARGET_FILE},EXPECTED_LEAD_MIN_HOURS=${EXPECTED_PANGU_LEAD_MIN_HOURS},EXPECTED_LEAD_MAX_HOURS=${EXPECTED_PANGU_LEAD_MAX_HOURS},INFER_PANGU_LEAD12_23_FROM_VALID_TIME=1,VERIFY_OUT_JSON=${VERIFY_JSON}" \
        sub_verify_pangu_station_product.slurm)"

    data_dependency=(--dependency="afterok:${verify_job}")
    s1_tianji_data_job="$(submit s1_tianji_data "${data_dependency[@]}" --export="ALL,FEATURE_SET=source_full,SOURCE_FULL_PROFILE=tianji,OUT_DIR=${S1_TIANJI_DIR}" sub_s1_overlap_data.slurm)"
    s1_ifs_data_job="$(submit s1_ifs_data "${data_dependency[@]}" --export="ALL,FEATURE_SET=source_full,SOURCE_FULL_PROFILE=ifs,OUT_DIR=${S1_IFS_DIR}" sub_s1_overlap_data.slurm)"
    s1_pangu_data_job="$(submit s1_pangu_data "${data_dependency[@]}" --export="ALL,FEATURE_SET=source_full,SOURCE_FULL_PROFILE=pangu2025,OUT_DIR=${S1_PANGU_DIR}" sub_s1_overlap_data.slurm)"
    tianji_data_job="$(submit tianji_data "${data_dependency[@]}" --export="ALL,FEATURE_SET=source_full,OUT_DIR=${TIANJI_DIR}" sub_tianji_overlap_data.slurm)"
    t2nd_data_job="$(submit t2nd_data "${data_dependency[@]}" --export="ALL,FEATURE_SET=source_full,OUT_DIR=${T2ND_DIR},RH2M_OVERRIDE_FILE=${T2ND_RH2M_FILE},RH2M_SOURCE_TAG=T2ND_rh2m" sub_tianji_overlap_data.slurm)"
    ifs_data_job="$(submit ifs_data "${data_dependency[@]}" --export="ALL,FEATURE_SET=source_full,YEAR=2025,OUT_DIR=${IFS_DIR}" sub_ifs_data.slurm)"
    pangu_data_job="$(submit pangu_data "${data_dependency[@]}" --export="ALL,SOURCE_KIND=station_nc,SOURCE_TAG=pangu2025,YEAR=2025,FEATURE_SET=source_full,SOURCE_FILE=${PANGU2025_STATION_FILE},TARGET_FILE=${TARGET_FILE},OUT_DIR=${PANGU_DIR},EXPECTED_LEAD_MIN_HOURS=${EXPECTED_PANGU_LEAD_MIN_HOURS},EXPECTED_LEAD_MAX_HOURS=${EXPECTED_PANGU_LEAD_MAX_HOURS},INFER_PANGU_LEAD12_23_FROM_VALID_TIME=1" sub_station_source_overlap_data.slurm)"
    era5_data_job="$(submit era5_data "${data_dependency[@]}" --export="ALL,SOURCE_KIND=era5_feature_dir,SOURCE_TAG=era5_2025,YEAR=2025,FEATURE_SET=source_full,TARGET_FILE=${TARGET_FILE},OUT_DIR=${ERA5_DIR}" sub_station_source_overlap_data.slurm)"

    all_data_deps="$(join_colon "${s1_tianji_data_job}" "${s1_ifs_data_job}" "${s1_pangu_data_job}" "${tianji_data_job}" "${t2nd_data_job}" "${ifs_data_job}" "${pangu_data_job}" "${era5_data_job}")"
    audit_job="$(submit data_audit --dependency="afterok:${all_data_deps}" --export="ALL,SOURCES=${sources},S1_PROFILES=${s1_profiles},AUDIT_OUT_DIR=${AUDIT_DIR}" sub_source_full_canonical_audit.slurm)"
fi

train_s1() {
    local label="$1" experiment="$2" run_id="$3" data_dir="$4"
    submit "${label}" --dependency="afterok:${audit_job}" --export="ALL,EXPERIMENT=${experiment},MODEL_ARCH=static_rnn,LOWVIS_RNN_RUN_ID=${run_id},OVERLAP_S1_DATA_DIR=${data_dir},LOWVIS_RNN_CLEAN_LOCAL_CACHE=1,LOWVIS_RNN_LOCAL_CACHE_ID=${RUN_TAG}_${label}" sub_ifs_overlap_baseline.slurm
}
s1_tianji_job="$(train_s1 s1_tianji s1_source_full_tianji "${S1_TIANJI_RUN_ID}" "${S1_TIANJI_DIR}")"
s1_ifs_job="$(train_s1 s1_ifs s1_source_full_ifs "${S1_IFS_RUN_ID}" "${S1_IFS_DIR}")"
s1_pangu_job="$(train_s1 s1_pangu s1_source_full_pangu2025 "${S1_PANGU_RUN_ID}" "${S1_PANGU_DIR}")"

train_s2() {
    local label="$1" dep="$2" experiment="$3" run_id="$4" data_dir="$5" ckpt="$6"
    submit "${label}" --dependency="afterok:${dep}" --export="ALL,EXPERIMENT=${experiment},MODEL_ARCH=static_rnn,LOWVIS_RNN_RUN_ID=${run_id},OVERLAP_STATIC_RNN_PRETRAINED_CKPT=${ckpt},OVERLAP_S2_DATA_DIR=${data_dir},LOWVIS_RNN_CLEAN_LOCAL_CACHE=1,LOWVIS_RNN_LOCAL_CACHE_ID=${RUN_TAG}_${label}" sub_ifs_overlap_baseline.slurm
}
tianji_s2_job="$(train_s2 tianji_s2 "${s1_tianji_job}" s2_tianji_source_full "${TIANJI_RUN_ID}" "${TIANJI_DIR}" "${S1_TIANJI_CKPT}")"
t2nd_s2_job="$(train_s2 t2nd_s2 "${s1_tianji_job}" s2_tianji_T2ND_rh2m_source_full "${T2ND_RUN_ID}" "${T2ND_DIR}" "${S1_TIANJI_CKPT}")"
ifs_s2_job="$(train_s2 ifs_s2 "${s1_ifs_job}" s2_ifs_source_full "${IFS_RUN_ID}" "${IFS_DIR}" "${S1_IFS_CKPT}")"
pangu_s2_job="$(train_s2 pangu_s2 "${s1_pangu_job}" s2_pangu2025_source_full "${PANGU_RUN_ID}" "${PANGU_DIR}" "${S1_PANGU_CKPT}")"
era5_s2_job="$(train_s2 era5_s2 "${s1_tianji_job}" s2_era5_2025_source_full "${ERA5_RUN_ID}" "${ERA5_DIR}" "${S1_TIANJI_CKPT}")"

s2_deps="$(join_colon "${tianji_s2_job}" "${t2nd_s2_job}" "${ifs_s2_job}" "${pangu_s2_job}" "${era5_s2_job}")"
eval_job="$(submit argmax_eval --dependency="afterok:${s2_deps}" --export="ALL,EVAL_SCENARIO=figure1_all_sources,OUT_ROOT=${EVAL_ROOT},TIANJI_DATA_DIR=${TIANJI_DIR},IFS_DATA_DIR=${IFS_DIR},T2ND_DATA_DIR=${T2ND_DIR},PANGU2025_DATA_DIR=${PANGU_DIR},ERA5_2025_DATA_DIR=${ERA5_DIR},TIANJI_CKPT=${CKPT_DIR}/${TIANJI_RUN_ID}_S2_PhaseB_best_score.pt,IFS_CKPT=${CKPT_DIR}/${IFS_RUN_ID}_S2_PhaseB_best_score.pt,T2ND_CKPT=${CKPT_DIR}/${T2ND_RUN_ID}_S2_PhaseB_best_score.pt,PANGU2025_CKPT=${CKPT_DIR}/${PANGU_RUN_ID}_S2_PhaseB_best_score.pt,ERA5_2025_CKPT=${CKPT_DIR}/${ERA5_RUN_ID}_S2_PhaseB_best_score.pt,THRESHOLD_MODE=argmax,STRICT_META=1" sub_static_rnn_source_full_argmax_eval.slurm)"

if is_true "${RESUME_FROM_AUDIT}"; then
    manifest="logs/source_full_units_v2_${RUN_TAG}_resume_$(date +%Y%m%d_%H%M%S)_submission.txt"
else
    manifest="logs/source_full_units_v2_${RUN_TAG}_submission.txt"
fi
{
    echo "run_tag=${RUN_TAG}"
    echo "canonical_unit_policy=pmst_canonical_units_v2_20260630"
    echo "resume_from_audit=${RESUME_FROM_AUDIT}"
    echo "data_root=${DATA_ROOT}"
    echo "verify_job=${verify_job}"
    echo "data_jobs=${all_data_deps}"
    echo "audit_job=${audit_job}"
    echo "s1_jobs=${s1_tianji_job}:${s1_ifs_job}:${s1_pangu_job}"
    echo "s2_jobs=${s2_deps}"
    echo "eval_job=${eval_job}"
    echo "s1_tianji_run_id=${S1_TIANJI_RUN_ID}"
    echo "s1_ifs_run_id=${S1_IFS_RUN_ID}"
    echo "s1_pangu_run_id=${S1_PANGU_RUN_ID}"
    echo "tianji_run_id=${TIANJI_RUN_ID}"
    echo "t2nd_run_id=${T2ND_RUN_ID}"
    echo "ifs_run_id=${IFS_RUN_ID}"
    echo "pangu_run_id=${PANGU_RUN_ID}"
    echo "era5_run_id=${ERA5_RUN_ID}"
} | tee "${manifest}"
echo "Submission manifest: ${manifest}"
