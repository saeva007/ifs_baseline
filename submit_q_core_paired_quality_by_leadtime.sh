#!/bin/bash
# Submit the leadtime-resolved (12-24 h) paired source-quality diagnosis.
#
# Example:
#   RUN_TAG=qcore_paired_quality_by_leadtime_v1_20260820 \
#     bash submit_q_core_paired_quality_by_leadtime.sh

set -euo pipefail

BASE="${BASE:-/public/home/putianshu/vis_mlp}"
BASELINE_DIR="${BASELINE_DIR:-${BASE}/ifs_baseline}"
RUN_TAG="${RUN_TAG:?RUN_TAG is required and must be a fresh analysis tag}"
DATA_RUN_TAG="${DATA_RUN_TAG:-qcore_t925_diag_auto_v2_20260715}"
DATA_ROOT="${DATA_ROOT:-${BASELINE_DIR}/q_core_t925_diagnostic_datasets/${DATA_RUN_TAG}}"
PANGU_DATA_DIR="${PANGU_DATA_DIR:-${DATA_ROOT}/pangu2025}"
TIANJI_DATA_DIR="${TIANJI_DATA_DIR:-${DATA_ROOT}/tianji}"
ERA5_DATA_DIR="${ERA5_DATA_DIR:-${DATA_ROOT}/era5_2025}"
OBS_ROOT="${OBS_ROOT:-${BASE}/auto_station}"
PAPER_EVAL_DIR="${PAPER_EVAL_DIR:-${BASE}/paper_eval}"
OUT_ROOT="${OUT_ROOT:-${BASE}/paper_eval_results_pm10_pm25_journal/q_core_paired_source_quality/${RUN_TAG}}"
OUT_DIR="${OUT_DIR:-${OUT_ROOT}/analysis}"
BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-20260715}"
TEST_MAX_ROWS="${TEST_MAX_ROWS:-0}"
LOW_VIS_THRESHOLD_M="${LOW_VIS_THRESHOLD_M:-1000}"
DRY_RUN="${DRY_RUN:-0}"
ALLOW_EXISTING_OUTPUT="${ALLOW_EXISTING_OUTPUT:-0}"
PREFLIGHT_PYTHON="${PREFLIGHT_PYTHON:-/public/home/jarvis226/miniconda3/envs/torch/bin/python}"
PREFLIGHT_SCRIPT="${PREFLIGHT_SCRIPT:-${BASELINE_DIR}/preflight_q_core_paired_source_quality.py}"
ANALYSIS_SCRIPT="${ANALYSIS_SCRIPT:-${BASELINE_DIR}/analyze_q_core_paired_quality_by_leadtime.py}"

case "${RUN_TAG}" in
  *[!A-Za-z0-9_.-]*|"") echo "ERROR: invalid RUN_TAG=${RUN_TAG}" >&2; exit 2 ;;
esac
[[ -x "${PREFLIGHT_PYTHON}" ]] || {
  echo "ERROR: PREFLIGHT_PYTHON is missing: ${PREFLIGHT_PYTHON}" >&2
  exit 2
}
[[ -s "${PREFLIGHT_SCRIPT}" ]] || {
  echo "ERROR: preflight script is missing: ${PREFLIGHT_SCRIPT}" >&2
  exit 2
}
[[ -s "${ANALYSIS_SCRIPT}" ]] || {
  echo "ERROR: leadtime analysis script is missing: ${ANALYSIS_SCRIPT}" >&2
  exit 2
}

preflight() {
  local source="$1"
  local data_dir="$2"
  local rc
  if "${PREFLIGHT_PYTHON}" "${PREFLIGHT_SCRIPT}" --source "${source}" --data-dir "${data_dir}"; then
    return 0
  else
    rc=$?
  fi
  echo "ERROR: ${source} input failed paired-quality preflight." >&2
  exit "${rc}"
}

preflight pangu "${PANGU_DATA_DIR}"
preflight tianji "${TIANJI_DATA_DIR}"
preflight era5_reference_analysis "${ERA5_DATA_DIR}"

if [[ "${DRY_RUN}" != "1" ]]; then
  [[ -d "${OBS_ROOT}" ]] || { echo "ERROR: observation root is missing: ${OBS_ROOT}" >&2; exit 2; }
  [[ -s "${PAPER_EVAL_DIR}/analyze_key_variable_quality.py" ]] || {
    echo "ERROR: observation alignment helper is missing: ${PAPER_EVAL_DIR}/analyze_key_variable_quality.py" >&2
    exit 2
  }
  if [[ "${ALLOW_EXISTING_OUTPUT}" != "1" && -e "${OUT_ROOT}" ]]; then
    echo "ERROR: output root already exists; use a fresh RUN_TAG: ${OUT_ROOT}" >&2
    exit 2
  fi
fi

mkdir -p "${BASELINE_DIR}/logs"
cd "${BASELINE_DIR}"

EXPORTS="ALL,RUN_TAG=${RUN_TAG},DATA_RUN_TAG=${DATA_RUN_TAG},PANGU_DATA_DIR=${PANGU_DATA_DIR},TIANJI_DATA_DIR=${TIANJI_DATA_DIR},ERA5_DATA_DIR=${ERA5_DATA_DIR},OBS_ROOT=${OBS_ROOT},PAPER_EVAL_DIR=${PAPER_EVAL_DIR},OUT_DIR=${OUT_DIR},BOOTSTRAP_SEED=${BOOTSTRAP_SEED},TEST_MAX_ROWS=${TEST_MAX_ROWS},LOW_VIS_THRESHOLD_M=${LOW_VIS_THRESHOLD_M},ANALYSIS_SCRIPT=${ANALYSIS_SCRIPT},PLOT_ERROR_REGIMES=0"

echo "Paired source-quality by-leadtime task"
echo "RUN_TAG=${RUN_TAG}"
echo "ANALYSIS_SCRIPT=${ANALYSIS_SCRIPT}"
echo "OUT_DIR=${OUT_DIR}"

if [[ "${DRY_RUN}" == "1" ]]; then
  printf '[DRY-RUN] sbatch --job-name=qcore_pair_lead --export=%q %q\n' \
    "${EXPORTS}" "${BASELINE_DIR}/sub_q_core_paired_source_quality.slurm"
  echo "analysis_job=dry_qcore_pair_lead"
else
  job_id="$(sbatch --parsable --job-name=qcore_pair_lead --export="${EXPORTS}" \
    "${BASELINE_DIR}/sub_q_core_paired_source_quality.slurm")"
  job_id="${job_id%%;*}"
  [[ "${job_id}" =~ ^[0-9]+$ ]] || { echo "ERROR: invalid sbatch id ${job_id}" >&2; exit 2; }
  echo "analysis_job=${job_id}"
fi
echo "results=${OUT_DIR}"
