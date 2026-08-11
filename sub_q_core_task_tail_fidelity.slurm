#!/bin/bash
#SBATCH -J qcore_tail_fidelity
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
FIT_MAX_ROWS="${FIT_MAX_ROWS:-0}"
TEST_MAX_ROWS="${TEST_MAX_ROWS:-0}"
BOOTSTRAP_ITERS="${BOOTSTRAP_ITERS:-1000}"
BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-20260811}"
LOW_VIS_THRESHOLD_M="${LOW_VIS_THRESHOLD_M:-1000}"
JOINT_MIN_COUNT="${JOINT_MIN_COUNT:-2}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/jarvis226/miniconda3/envs/torch/bin/python}"

mkdir -p "${BASELINE_DIR}/logs" "$(dirname "${OUT_DIR}")"
[[ -x "${PYTHON_BIN}" ]] || {
  echo "ERROR: analysis Python is missing: ${PYTHON_BIN}" >&2
  exit 2
}
[[ ! -e "${OUT_DIR}" ]] || {
  echo "ERROR: output directory already exists; use a fresh RUN_TAG: ${OUT_DIR}" >&2
  exit 2
}

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"

CMD=(
  "${PYTHON_BIN}" "${BASELINE_DIR}/analyze_q_core_task_tail_fidelity.py"
  --pangu-dir "${PANGU_DATA_DIR}"
  --tianji-dir "${TIANJI_DATA_DIR}"
  --era5-dir "${ERA5_DATA_DIR}"
  --obs-root "${OBS_ROOT}"
  --paper-eval-dir "${PAPER_EVAL_DIR}"
  --out-dir "${OUT_DIR}"
  --fit-max-rows "${FIT_MAX_ROWS}"
  --test-max-rows "${TEST_MAX_ROWS}"
  --bootstrap-iters "${BOOTSTRAP_ITERS}"
  --bootstrap-seed "${BOOTSTRAP_SEED}"
  --low-vis-threshold-m "${LOW_VIS_THRESHOLD_M}"
  --joint-min-count "${JOINT_MIN_COUNT}"
)

echo "Task-relevant marginal and joint tail-fidelity diagnosis (training jobs: 0)"
echo "Validation fixes thresholds; test is evaluation-only"
echo "Command: ${CMD[*]}"
"${CMD[@]}"

