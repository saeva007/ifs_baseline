#!/bin/bash
#SBATCH -J qcore_t925_joint
#SBATCH -p kshcexclu04
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH -t 24:00:00
#SBATCH -D /public/home/putianshu/vis_mlp/ifs_baseline
#SBATCH -o /public/home/putianshu/vis_mlp/ifs_baseline/logs/%j_%x.out
#SBATCH -e /public/home/putianshu/vis_mlp/ifs_baseline/logs/%j_%x.err
#SBATCH --exclusive

set -euo pipefail

BASE="${BASE:-/public/home/putianshu/vis_mlp}"
BASELINE_DIR="${BASELINE_DIR:-${BASE}/ifs_baseline}"
RUN_TAG="${RUN_TAG:?RUN_TAG is required}"
DATA_ROOT="${DATA_ROOT:-${BASELINE_DIR}/q_core_t925_joint_datasets/${RUN_TAG}}"
PANGU_DATA_DIR="${PANGU_DATA_DIR:-${DATA_ROOT}/pangu2025}"
TIANJI_DATA_DIR="${TIANJI_DATA_DIR:-${DATA_ROOT}/tianji}"
ERA5_DATA_DIR="${ERA5_DATA_DIR:-${DATA_ROOT}/era5_2025}"
EVAL_ROOT="${EVAL_ROOT:-${BASE}/paper_eval_results_pm10_pm25_journal/q_core_t925_joint/${RUN_TAG}}"
FACTORIAL_ANALYSIS_DIR="${FACTORIAL_ANALYSIS_DIR:-${EVAL_ROOT}/factorial_analysis}"
OUT_DIR="${OUT_DIR:-${EVAL_ROOT}/joint_structure_analysis}"
FIT_MAX_ROWS="${FIT_MAX_ROWS:-200000}"
TEST_MAX_ROWS="${TEST_MAX_ROWS:-0}"
BOOTSTRAP_ITERS="${BOOTSTRAP_ITERS:-1000}"
BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-20260714}"
RFF_DIM="${RFF_DIM:-512}"
RFF_BANDWIDTH_SAMPLE="${RFF_BANDWIDTH_SAMPLE:-2000}"
LOW_VIS_THRESHOLD_M="${LOW_VIS_THRESHOLD_M:-1000}"
NO_FIGURES="${NO_FIGURES:-0}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/jarvis226/miniconda3/envs/torch/bin/python}"

mkdir -p "${BASELINE_DIR}/logs" "${OUT_DIR}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: analysis Python is missing: ${PYTHON_BIN}" >&2
  exit 2
fi
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"

CMD=(
  "${PYTHON_BIN}" "${BASELINE_DIR}/analyze_q_core_t925_joint_structure.py"
  --pangu-dir "${PANGU_DATA_DIR}"
  --tianji-dir "${TIANJI_DATA_DIR}"
  --era5-dir "${ERA5_DATA_DIR}"
  --factorial-analysis-dir "${FACTORIAL_ANALYSIS_DIR}"
  --out-dir "${OUT_DIR}"
  --fit-max-rows "${FIT_MAX_ROWS}"
  --test-max-rows "${TEST_MAX_ROWS}"
  --bootstrap-iters "${BOOTSTRAP_ITERS}"
  --bootstrap-seed "${BOOTSTRAP_SEED}"
  --rff-dim "${RFF_DIM}"
  --rff-bandwidth-sample "${RFF_BANDWIDTH_SAMPLE}"
  --low-vis-threshold-m "${LOW_VIS_THRESHOLD_M}"
)
if [[ "${NO_FIGURES}" == "1" ]]; then
  CMD+=(--no-figures)
fi

echo "Command: ${CMD[*]}"
"${CMD[@]}"
