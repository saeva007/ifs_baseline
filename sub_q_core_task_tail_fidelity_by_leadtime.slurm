#!/bin/bash
#SBATCH -J qcore_tail_lead
#SBATCH -p kshcexclu04
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH -t 12:00:00
#SBATCH -D /public/home/putianshu/vis_mlp/ifs_baseline
#SBATCH -o /public/home/putianshu/vis_mlp/ifs_baseline/logs/%j_%x.out
#SBATCH -e /public/home/putianshu/vis_mlp/ifs_baseline/logs/%j_%x.err
#SBATCH --exclusive

set -euo pipefail

BASE="${BASE:-/public/home/putianshu/vis_mlp}"
BASELINE_DIR="${BASELINE_DIR:-${BASE}/ifs_baseline}"
RUN_TAG="${RUN_TAG:?RUN_TAG is required}"
DATA_RUN_TAG="${DATA_RUN_TAG:-qcore_t925_diag_auto_v2_20260715}"
DATA_ROOT="${DATA_ROOT:-${BASELINE_DIR}/q_core_t925_diagnostic_datasets/${DATA_RUN_TAG}}"
PANGU_DATA_DIR="${PANGU_DATA_DIR:-${DATA_ROOT}/pangu2025}"
TIANJI_DATA_DIR="${TIANJI_DATA_DIR:-${DATA_ROOT}/tianji}"
ERA5_DATA_DIR="${ERA5_DATA_DIR:-${DATA_ROOT}/era5_2025}"
OBS_ROOT="${OBS_ROOT:-${BASE}/auto_station}"
PAPER_EVAL_DIR="${PAPER_EVAL_DIR:-${BASE}/paper_eval}"
OUT_DIR="${OUT_DIR:-${BASE}/paper_eval_results_pm10_pm25_journal/q_core_task_tail_fidelity/${RUN_TAG}/analysis}"
# Keep the two figure features local to this script.  A comma-containing value
# must not be transported through Slurm's comma-delimited --export argument.
FEATURES="T2M,Q_1000"
FIT_MAX_ROWS="${FIT_MAX_ROWS:-0}"
TEST_MAX_ROWS="${TEST_MAX_ROWS:-0}"
SELECTION_SEED="${SELECTION_SEED:-20260820}"
LOW_VIS_THRESHOLD_M="${LOW_VIS_THRESHOLD_M:-1000}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/jarvis226/miniconda3/envs/torch/bin/python}"
ANALYSIS_SCRIPT="${ANALYSIS_SCRIPT:-${BASELINE_DIR}/analyze_q_core_task_tail_fidelity_by_leadtime.py}"

mkdir -p "${BASELINE_DIR}/logs" "$(dirname "${OUT_DIR}")"
[[ -x "${PYTHON_BIN}" ]] || {
  echo "ERROR: analysis Python is missing: ${PYTHON_BIN}" >&2
  exit 2
}
[[ -s "${ANALYSIS_SCRIPT}" ]] || {
  echo "ERROR: analysis script is missing: ${ANALYSIS_SCRIPT}" >&2
  exit 2
}
[[ ! -e "${OUT_DIR}" ]] || {
  echo "ERROR: output directory already exists; use a fresh RUN_TAG: ${OUT_DIR}" >&2
  exit 2
}

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"

CMD=(
  "${PYTHON_BIN}" "${ANALYSIS_SCRIPT}"
  --pangu-dir "${PANGU_DATA_DIR}"
  --tianji-dir "${TIANJI_DATA_DIR}"
  --era5-dir "${ERA5_DATA_DIR}"
  --obs-root "${OBS_ROOT}"
  --paper-eval-dir "${PAPER_EVAL_DIR}"
  --out-dir "${OUT_DIR}"
  --features "${FEATURES}"
  --fit-max-rows "${FIT_MAX_ROWS}"
  --test-max-rows "${TEST_MAX_ROWS}"
  --selection-seed "${SELECTION_SEED}"
  --low-vis-threshold-m "${LOW_VIS_THRESHOLD_M}"
)

echo "Task-tail CSI by forecast lead (training jobs: 0)"
echo "Validation fixes direction and threshold; test is evaluation-only"
echo "Command: ${CMD[*]}"
"${CMD[@]}"
