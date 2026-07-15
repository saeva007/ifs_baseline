#!/bin/bash
# Submit one observation-anchored RH2M quality job. No data build or training.
#
# Example:
#   RUN_TAG=qcore_rh2m_quality_v1_20260715 \
#     bash submit_q_core_rh2m_observation_quality.sh

set -euo pipefail

BASE="${BASE:-/public/home/putianshu/vis_mlp}"
BASELINE_DIR="${BASELINE_DIR:-${BASE}/ifs_baseline}"
RUN_TAG="${RUN_TAG:?RUN_TAG is required and must be a fresh analysis tag}"
FEATURE_SET="${FEATURE_SET:-common_core}"
TIANJI_RH2M_DATA_DIR="${TIANJI_RH2M_DATA_DIR:-${BASELINE_DIR}/ml_dataset_overlap_tianji_12h_pm10_pm25_${FEATURE_SET}}"
T2ND_RH2M_DATA_DIR="${T2ND_RH2M_DATA_DIR:-${BASELINE_DIR}/ml_dataset_overlap_tianji_12h_pm10_pm25_T2ND_rh2m_${FEATURE_SET}}"
ERA5_RH2M_DATA_DIR="${ERA5_RH2M_DATA_DIR:-${BASELINE_DIR}/ml_dataset_overlap_era5_2025_12h_pm10_pm25_${FEATURE_SET}}"
OBS_ROOT="${OBS_ROOT:-${BASE}/auto_station}"
OUT_ROOT="${OUT_ROOT:-${BASE}/paper_eval_results_pm10_pm25_journal/q_core_rh2m_observation_quality/${RUN_TAG}}"
OUT_DIR="${OUT_DIR:-${OUT_ROOT}/analysis}"
BOOTSTRAP_ITERS="${BOOTSTRAP_ITERS:-1000}"
BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-20260715}"
LIMIT_SAMPLES="${LIMIT_SAMPLES:-0}"
DRY_RUN="${DRY_RUN:-0}"
ALLOW_EXISTING_OUTPUT="${ALLOW_EXISTING_OUTPUT:-0}"
PREFLIGHT_PYTHON="${PREFLIGHT_PYTHON:-/public/home/jarvis226/miniconda3/envs/torch/bin/python}"
PREFLIGHT_SCRIPT="${PREFLIGHT_SCRIPT:-${BASELINE_DIR}/preflight_q_core_rh2m_observation_quality.py}"
ANALYSIS_SCRIPT="${ANALYSIS_SCRIPT:-${BASE}/paper_eval/analyze_multi_source_rh2m_quality.py}"

case "${RUN_TAG}" in
  *[!A-Za-z0-9_.-]*|"") echo "ERROR: invalid RUN_TAG=${RUN_TAG}" >&2; exit 2 ;;
esac
[[ "${FEATURE_SET}" == "common_core" || "${FEATURE_SET}" == "source_full" ]] || {
  echo "ERROR: FEATURE_SET must be common_core or source_full so RH2M is present" >&2
  exit 2
}
[[ "${BOOTSTRAP_ITERS}" =~ ^[0-9]+$ ]] && (( BOOTSTRAP_ITERS >= 100 )) || {
  echo "ERROR: BOOTSTRAP_ITERS must be an integer >=100" >&2
  exit 2
}
[[ "${LIMIT_SAMPLES}" =~ ^[0-9]+$ ]] || {
  echo "ERROR: LIMIT_SAMPLES must be a non-negative integer" >&2
  exit 2
}
[[ -x "${PREFLIGHT_PYTHON}" ]] || {
  echo "ERROR: PREFLIGHT_PYTHON is missing: ${PREFLIGHT_PYTHON}" >&2
  exit 2
}
[[ -s "${PREFLIGHT_SCRIPT}" ]] || {
  echo "ERROR: preflight script is missing: ${PREFLIGHT_SCRIPT}" >&2
  exit 2
}

preflight() {
  local role="$1"
  local data_dir="$2"
  local rc
  if "${PREFLIGHT_PYTHON}" "${PREFLIGHT_SCRIPT}" --role "${role}" --data-dir "${data_dir}"; then
    return 0
  else
    rc=$?
  fi
  if [[ "${rc}" == "2" ]]; then
    echo "ERROR: ${role} dataset failed RH2M provenance/layout preflight." >&2
  else
    echo "ERROR: ${role} preflight infrastructure failed with exit code ${rc}." >&2
  fi
  echo "No job was submitted." >&2
  exit "${rc}"
}

preflight tianji_product "${TIANJI_RH2M_DATA_DIR}"
preflight t2nd_raw "${T2ND_RH2M_DATA_DIR}"
preflight era5_reference_analysis "${ERA5_RH2M_DATA_DIR}"

if [[ "${DRY_RUN}" != "1" ]]; then
  [[ -d "${OBS_ROOT}" ]] || { echo "ERROR: observation root is missing: ${OBS_ROOT}" >&2; exit 2; }
  [[ -s "${ANALYSIS_SCRIPT}" ]] || { echo "ERROR: RH2M analysis script is missing: ${ANALYSIS_SCRIPT}" >&2; exit 2; }
  if [[ "${ALLOW_EXISTING_OUTPUT}" != "1" && -e "${OUT_ROOT}" ]]; then
    echo "ERROR: output root already exists; use a fresh RUN_TAG: ${OUT_ROOT}" >&2
    exit 2
  fi
fi

mkdir -p "${BASELINE_DIR}/logs"
cd "${BASELINE_DIR}"

EXPORTS="ALL,RUN_TAG=${RUN_TAG},FEATURE_SET=${FEATURE_SET},TIANJI_RH2M_DATA_DIR=${TIANJI_RH2M_DATA_DIR},T2ND_RH2M_DATA_DIR=${T2ND_RH2M_DATA_DIR},ERA5_RH2M_DATA_DIR=${ERA5_RH2M_DATA_DIR},OBS_ROOT=${OBS_ROOT},OUT_DIR=${OUT_DIR},BOOTSTRAP_ITERS=${BOOTSTRAP_ITERS},BOOTSTRAP_SEED=${BOOTSTRAP_SEED},LIMIT_SAMPLES=${LIMIT_SAMPLES},SCRIPT_PATH=${ANALYSIS_SCRIPT}"

echo "RH2M observation-quality task"
echo "RUN_TAG=${RUN_TAG}"
echo "FEATURE_SET=${FEATURE_SET}"
echo "TRAINING_JOBS=0"
echo "PANGU_RH2M_PROXY=excluded"
echo "TIANJI_RH2M_DATA_DIR=${TIANJI_RH2M_DATA_DIR}"
echo "T2ND_RH2M_DATA_DIR=${T2ND_RH2M_DATA_DIR}"
echo "ERA5_RH2M_DATA_DIR=${ERA5_RH2M_DATA_DIR}"
echo "OUT_DIR=${OUT_DIR}"

if [[ "${DRY_RUN}" == "1" ]]; then
  printf '[DRY-RUN] sbatch --job-name=qcore_rh2m_qa --export=%q %q\n' \
    "${EXPORTS}" "${BASELINE_DIR}/sub_q_core_rh2m_observation_quality.slurm"
  echo "analysis_job=dry_qcore_rh2m_qa"
else
  job_id="$(sbatch --parsable --job-name=qcore_rh2m_qa --export="${EXPORTS}" \
    "${BASELINE_DIR}/sub_q_core_rh2m_observation_quality.slurm")"
  job_id="${job_id%%;*}"
  [[ "${job_id}" =~ ^[0-9]+$ ]] || { echo "ERROR: invalid sbatch id ${job_id}" >&2; exit 2; }
  echo "analysis_job=${job_id}"
fi
echo "results=${OUT_DIR}"
