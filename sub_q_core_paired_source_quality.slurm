#!/bin/bash
#SBATCH -J qcore_pair_qa
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
OUT_DIR="${OUT_DIR:-${BASE}/paper_eval_results_pm10_pm25_journal/q_core_paired_source_quality/${RUN_TAG}/analysis}"
FIGURE_OUT_DIR="${FIGURE_OUT_DIR:-${OUT_DIR}/figures_common_error_regimes}"
TEST_MAX_ROWS="${TEST_MAX_ROWS:-0}"
BOOTSTRAP_ITERS="${BOOTSTRAP_ITERS:-1000}"
BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-20260715}"
LOW_VIS_THRESHOLD_M="${LOW_VIS_THRESHOLD_M:-1000}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/jarvis226/miniconda3/envs/torch/bin/python}"
PLOT_ERROR_REGIMES="${PLOT_ERROR_REGIMES:-1}"
PLOT_SCRIPT="${PLOT_SCRIPT:-${BASELINE_DIR}/plot_common_variable_error_regimes.py}"
ANALYSIS_SCRIPT="${ANALYSIS_SCRIPT:-${BASELINE_DIR}/analyze_q_core_paired_source_quality.py}"

mkdir -p "${BASELINE_DIR}/logs" "${OUT_DIR}" "${FIGURE_OUT_DIR}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: analysis Python is missing: ${PYTHON_BIN}" >&2
  exit 2
fi

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
  --test-max-rows "${TEST_MAX_ROWS}"
  --bootstrap-iters "${BOOTSTRAP_ITERS}"
  --bootstrap-seed "${BOOTSTRAP_SEED}"
  --low-vis-threshold-m "${LOW_VIS_THRESHOLD_M}"
)

echo "Paired pressure-level/surface quality diagnosis (training jobs: 0)"
echo "Command: ${CMD[*]}"
"${CMD[@]}"

if [[ "${PLOT_ERROR_REGIMES}" == "1" || "${PLOT_ERROR_REGIMES}" == "true" ]]; then
  PLOT_CMD=(
    "${PYTHON_BIN}" "${PLOT_SCRIPT}"
    --paired-quality-dir "${OUT_DIR}"
    --out-dir "${FIGURE_OUT_DIR}"
    --formats "svg,pdf,png,tiff"
    --dpi 600
  )
  echo "Common-variable error-regime figure: ${PLOT_CMD[*]}"
  "${PLOT_CMD[@]}"
fi
