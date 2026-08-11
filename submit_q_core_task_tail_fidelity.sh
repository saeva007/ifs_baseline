#!/bin/bash
# Submit a no-training, validation-frozen task-tail fidelity diagnosis.
#
# Example:
#   RUN_TAG=qcore_task_tail_fidelity_v1_20260811 \
#     bash submit_q_core_task_tail_fidelity.sh

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
OUT_ROOT="${OUT_ROOT:-${BASE}/paper_eval_results_pm10_pm25_journal/q_core_task_tail_fidelity/${RUN_TAG}}"
OUT_DIR="${OUT_DIR:-${OUT_ROOT}/analysis}"
FIT_MAX_ROWS="${FIT_MAX_ROWS:-0}"
TEST_MAX_ROWS="${TEST_MAX_ROWS:-0}"
BOOTSTRAP_ITERS="${BOOTSTRAP_ITERS:-1000}"
BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-20260811}"
LOW_VIS_THRESHOLD_M="${LOW_VIS_THRESHOLD_M:-1000}"
JOINT_MIN_COUNT="${JOINT_MIN_COUNT:-2}"
DRY_RUN="${DRY_RUN:-0}"
PREFLIGHT_PYTHON="${PREFLIGHT_PYTHON:-/public/home/jarvis226/miniconda3/envs/torch/bin/python}"
PREFLIGHT_SCRIPT="${PREFLIGHT_SCRIPT:-${BASELINE_DIR}/preflight_q_core_paired_source_quality.py}"

case "${RUN_TAG}" in
  *[!A-Za-z0-9_.-]*|"") echo "ERROR: invalid RUN_TAG=${RUN_TAG}" >&2; exit 2 ;;
esac
for value_name in FIT_MAX_ROWS TEST_MAX_ROWS; do
  value="${!value_name}"
  [[ "${value}" =~ ^[0-9]+$ ]] || {
    echo "ERROR: ${value_name} must be a non-negative integer" >&2
    exit 2
  }
done
[[ "${BOOTSTRAP_ITERS}" =~ ^[0-9]+$ ]] && (( BOOTSTRAP_ITERS >= 100 )) || {
  echo "ERROR: BOOTSTRAP_ITERS must be an integer >=100" >&2
  exit 2
}
[[ "${JOINT_MIN_COUNT}" =~ ^[1-9][0-9]*$ ]] || {
  echo "ERROR: JOINT_MIN_COUNT must be a positive integer" >&2
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

for pair in \
  "pangu:${PANGU_DATA_DIR}" \
  "tianji:${TIANJI_DATA_DIR}" \
  "era5_reference_analysis:${ERA5_DATA_DIR}"; do
  source="${pair%%:*}"
  data_dir="${pair#*:}"
  "${PREFLIGHT_PYTHON}" "${PREFLIGHT_SCRIPT}" \
    --source "${source}" --data-dir "${data_dir}"
done

if [[ "${DRY_RUN}" != "1" ]]; then
  [[ -d "${OBS_ROOT}" ]] || {
    echo "ERROR: observation root is missing: ${OBS_ROOT}" >&2
    exit 2
  }
  [[ -s "${PAPER_EVAL_DIR}/analyze_key_variable_quality.py" ]] || {
    echo "ERROR: observation alignment helper is missing: ${PAPER_EVAL_DIR}/analyze_key_variable_quality.py" >&2
    exit 2
  }
  [[ ! -e "${OUT_ROOT}" ]] || {
    echo "ERROR: output root already exists; use a fresh RUN_TAG: ${OUT_ROOT}" >&2
    exit 2
  }
fi

mkdir -p "${BASELINE_DIR}/logs"
cd "${BASELINE_DIR}"

EXPORTS="ALL,RUN_TAG=${RUN_TAG},DATA_RUN_TAG=${DATA_RUN_TAG},PANGU_DATA_DIR=${PANGU_DATA_DIR},TIANJI_DATA_DIR=${TIANJI_DATA_DIR},ERA5_DATA_DIR=${ERA5_DATA_DIR},OBS_ROOT=${OBS_ROOT},PAPER_EVAL_DIR=${PAPER_EVAL_DIR},OUT_DIR=${OUT_DIR},FIT_MAX_ROWS=${FIT_MAX_ROWS},TEST_MAX_ROWS=${TEST_MAX_ROWS},BOOTSTRAP_ITERS=${BOOTSTRAP_ITERS},BOOTSTRAP_SEED=${BOOTSTRAP_SEED},LOW_VIS_THRESHOLD_M=${LOW_VIS_THRESHOLD_M},JOINT_MIN_COUNT=${JOINT_MIN_COUNT}"

echo "Task-tail fidelity analysis"
echo "RUN_TAG=${RUN_TAG}"
echo "DATA_RUN_TAG=${DATA_RUN_TAG}"
echo "TRAINING_JOBS=0"
echo "OUT_DIR=${OUT_DIR}"

if [[ "${DRY_RUN}" == "1" ]]; then
  printf '[DRY-RUN] sbatch --job-name=qcore_tail_fidelity --export=%q %q\n' \
    "${EXPORTS}" "${BASELINE_DIR}/sub_q_core_task_tail_fidelity.slurm"
  echo "analysis_job=dry_qcore_tail_fidelity"
else
  job_id="$(sbatch --parsable --job-name=qcore_tail_fidelity --export="${EXPORTS}" \
    "${BASELINE_DIR}/sub_q_core_task_tail_fidelity.slurm")"
  job_id="${job_id%%;*}"
  [[ "${job_id}" =~ ^[0-9]+$ ]] || {
    echo "ERROR: invalid sbatch id ${job_id}" >&2
    exit 2
  }
  echo "analysis_job=${job_id}"
fi
echo "results=${OUT_DIR}"

