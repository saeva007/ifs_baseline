#!/bin/bash
# Submit the default T925--moisture diagnosis with zero model training.
#
# Reuse mode (one analysis job):
#   RUN_TAG=qcore_t925_diag_v1_20260714 bash submit_q_core_t925_diagnostics.sh
#
# If existing source-full configs fail the strict provenance/unit preflight,
# rebuild only the three diagnostic datasets (still zero training):
#   RUN_TAG=qcore_t925_diag_rebuild_v1_20260714 BUILD_DATA=1 \
#     bash submit_q_core_t925_diagnostics.sh

set -euo pipefail

BASE="${BASE:-/public/home/putianshu/vis_mlp}"
BASELINE_DIR="${BASELINE_DIR:-${BASE}/ifs_baseline}"
RUN_TAG="${RUN_TAG:?RUN_TAG is required and must be a fresh diagnostic tag}"
DRY_RUN="${DRY_RUN:-0}"
BUILD_DATA="${BUILD_DATA:-0}"
ALLOW_EXISTING_OUTPUT="${ALLOW_EXISTING_OUTPUT:-0}"
BOOTSTRAP_ITERS="${BOOTSTRAP_ITERS:-1000}"
FIT_MAX_ROWS="${FIT_MAX_ROWS:-200000}"
TEST_MAX_ROWS="${TEST_MAX_ROWS:-0}"
RFF_DIM="${RFF_DIM:-512}"
RFF_BANDWIDTH_SAMPLE="${RFF_BANDWIDTH_SAMPLE:-2000}"
MIN_EVENT_COVERAGE="${MIN_EVENT_COVERAGE:-0.80}"
MT2PW_RUN_TAG="${MT2PW_RUN_TAG:-qcore_hybrid_mt2pw_formal_v1_20260708}"
EVENT_ANALYSIS_DIR="${EVENT_ANALYSIS_DIR:-${BASE}/paper_eval_results_pm10_pm25_journal/q_core_hybrid_factorial/${MT2PW_RUN_TAG}/analysis}"
OUT_ROOT="${OUT_ROOT:-${BASE}/paper_eval_results_pm10_pm25_journal/q_core_t925_diagnostics/${RUN_TAG}}"
OUT_DIR="${OUT_DIR:-${OUT_ROOT}/analysis}"
PANGU2025_STATION_FILE="${PANGU2025_STATION_FILE:-${BASELINE_DIR}/pangu_station/pangu_station_2025_lead12_23h_canonical.nc}"

if [[ "${BUILD_DATA}" == "1" ]]; then
  DATA_ROOT="${DATA_ROOT:-${BASELINE_DIR}/q_core_t925_diagnostic_datasets/${RUN_TAG}}"
  PANGU_DATA_DIR="${PANGU_DATA_DIR:-${DATA_ROOT}/pangu2025}"
  TIANJI_DATA_DIR="${TIANJI_DATA_DIR:-${DATA_ROOT}/tianji}"
  ERA5_DATA_DIR="${ERA5_DATA_DIR:-${DATA_ROOT}/era5_2025}"
else
  DATA_ROOT="${DATA_ROOT:-}"
  PANGU_DATA_DIR="${PANGU_DATA_DIR:-${BASELINE_DIR}/ml_dataset_overlap_pangu2025_12h_pm10_pm25_source_full}"
  TIANJI_DATA_DIR="${TIANJI_DATA_DIR:-${BASELINE_DIR}/ml_dataset_overlap_tianji_12h_pm10_pm25_source_full}"
  ERA5_DATA_DIR="${ERA5_DATA_DIR:-${BASELINE_DIR}/ml_dataset_overlap_era5_2025_12h_pm10_pm25_source_full}"
fi

case "${RUN_TAG}" in
  *[!A-Za-z0-9_.-]*|"") echo "ERROR: invalid RUN_TAG=${RUN_TAG}" >&2; exit 2 ;;
esac
[[ "${BUILD_DATA}" == "0" || "${BUILD_DATA}" == "1" ]] || {
  echo "ERROR: BUILD_DATA must be 0 or 1" >&2; exit 2;
}

if [[ "${DRY_RUN}" != "1" ]]; then
  [[ -s "${EVENT_ANALYSIS_DIR}/event_case_control_samples.csv.gz" ]] || {
    echo "ERROR: completed MTW/mt2pw event table is missing: ${EVENT_ANALYSIS_DIR}" >&2
    exit 2
  }
  if [[ "${ALLOW_EXISTING_OUTPUT}" != "1" && -e "${OUT_ROOT}" ]]; then
    echo "ERROR: output root already exists; use a fresh RUN_TAG: ${OUT_ROOT}" >&2
    exit 2
  fi
  if [[ "${BUILD_DATA}" == "1" ]]; then
    [[ -s "${PANGU2025_STATION_FILE}" ]] || {
      echo "ERROR: canonical Pangu station product is missing: ${PANGU2025_STATION_FILE}" >&2
      exit 2
    }
    [[ ! -e "${DATA_ROOT}" ]] || {
      echo "ERROR: fresh diagnostic DATA_ROOT already exists: ${DATA_ROOT}" >&2
      exit 2
    }
  else
    for dir in "${PANGU_DATA_DIR}" "${TIANJI_DATA_DIR}" "${ERA5_DATA_DIR}"; do
      [[ -s "${dir}/dataset_build_config.json" ]] || {
        echo "ERROR: reusable source-full dataset is missing its config: ${dir}" >&2
        echo "Rerun with BUILD_DATA=1 to build diagnostic-only data without training." >&2
        exit 2
      }
    done
  fi
fi

mkdir -p "${BASELINE_DIR}/logs"
cd "${BASELINE_DIR}"

submit() {
  local label="$1"; shift
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '[DRY-RUN] %s: sbatch' "${label}" >&2
    printf ' %q' "$@" >&2
    printf '\n' >&2
    printf 'dry_%s\n' "${label}"
  else
    local job_id
    job_id="$(sbatch --parsable "$@")"
    job_id="${job_id%%;*}"
    [[ "${job_id}" =~ ^[0-9]+$ ]] || { echo "ERROR: invalid sbatch id ${job_id}" >&2; exit 2; }
    echo "[SUBMITTED] ${label}: ${job_id}" >&2
    printf '%s\n' "${job_id}"
  fi
}

echo "T925--moisture diagnostic-only chain"
echo "RUN_TAG=${RUN_TAG}"
echo "BUILD_DATA=${BUILD_DATA}"
echo "TRAINING_JOBS=0"
echo "PANGU_DATA_DIR=${PANGU_DATA_DIR}"
echo "TIANJI_DATA_DIR=${TIANJI_DATA_DIR}"
echo "ERA5_DATA_DIR=${ERA5_DATA_DIR}"
echo "EVENT_ANALYSIS_DIR=${EVENT_ANALYSIS_DIR}"
echo "OUT_DIR=${OUT_DIR}"

data_jobs=""
if [[ "${BUILD_DATA}" == "1" ]]; then
  tianji_job=$(submit t925_diag_tianji_data \
    --export="ALL,FEATURE_SET=q_core_t925_no_rh2m,OUT_DIR=${TIANJI_DATA_DIR}" \
    sub_tianji_overlap_data.slurm)
  pangu_job=$(submit t925_diag_pangu_data \
    --export="ALL,SOURCE_KIND=station_nc,SOURCE_TAG=pangu2025,YEAR=2025,FEATURE_SET=q_core_t925_no_rh2m,SOURCE_FILE=${PANGU2025_STATION_FILE},OUT_DIR=${PANGU_DATA_DIR},EXPECTED_LEAD_MIN_HOURS=12,EXPECTED_LEAD_MAX_HOURS=23,INFER_PANGU_LEAD12_23_FROM_VALID_TIME=1" \
    sub_station_source_overlap_data.slurm)
  era5_job=$(submit t925_diag_era5_data \
    --export="ALL,SOURCE_KIND=era5_feature_dir,SOURCE_TAG=era5_2025,YEAR=2025,FEATURE_SET=q_core_t925_no_rh2m,OUT_DIR=${ERA5_DATA_DIR}" \
    sub_station_source_overlap_data.slurm)
  data_jobs="${tianji_job}:${pangu_job}:${era5_job}"
fi

analysis_args=(
  --job-name="qcore_t925_diag"
  --export="ALL,RUN_TAG=${RUN_TAG},ANALYSIS_MODE=diagnostic_only,PANGU_DATA_DIR=${PANGU_DATA_DIR},TIANJI_DATA_DIR=${TIANJI_DATA_DIR},ERA5_DATA_DIR=${ERA5_DATA_DIR},EVENT_ANALYSIS_DIR=${EVENT_ANALYSIS_DIR},EVAL_ROOT=${OUT_ROOT},OUT_DIR=${OUT_DIR},FIT_MAX_ROWS=${FIT_MAX_ROWS},TEST_MAX_ROWS=${TEST_MAX_ROWS},BOOTSTRAP_ITERS=${BOOTSTRAP_ITERS},RFF_DIM=${RFF_DIM},RFF_BANDWIDTH_SAMPLE=${RFF_BANDWIDTH_SAMPLE},MIN_EVENT_COVERAGE=${MIN_EVENT_COVERAGE}"
)
if [[ -n "${data_jobs}" && "${DRY_RUN}" != "1" ]]; then
  analysis_args+=(--dependency="afterok:${data_jobs}")
fi
analysis_job=$(submit t925_joint_diagnosis "${analysis_args[@]}" sub_q_core_t925_joint_structure.slurm)

if [[ "${DRY_RUN}" != "1" ]]; then
  mkdir -p "${OUT_ROOT}"
  {
    echo "status=scheduled"
    echo "analysis_mode=diagnostic_only"
    echo "new_training_jobs=0"
    echo "run_tag=${RUN_TAG}"
    echo "build_data=${BUILD_DATA}"
    echo "pangu_data_dir=${PANGU_DATA_DIR}"
    echo "tianji_data_dir=${TIANJI_DATA_DIR}"
    echo "era5_data_dir=${ERA5_DATA_DIR}"
    echo "event_analysis_dir=${EVENT_ANALYSIS_DIR}"
    echo "analysis_job=${analysis_job}"
    echo "claim_limit=association_between_joint_structure_discrepancy_and_existing_hit_miss_outcomes"
  } > "${OUT_ROOT}/submission_manifest_${RUN_TAG}.txt"
fi

echo "analysis_job=${analysis_job}"
echo "results=${OUT_DIR}"
