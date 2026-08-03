#!/bin/bash
#SBATCH -J best_effort_fig
#SBATCH -p kshcexclu04
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH -t 02:00:00
#SBATCH -D /public/home/putianshu/vis_mlp/ifs_baseline
#SBATCH -o /public/home/putianshu/vis_mlp/ifs_baseline/logs/%j_%x.out
#SBATCH -e /public/home/putianshu/vis_mlp/ifs_baseline/logs/%j_%x.err
#SBATCH --exclusive

set -euo pipefail

BASE="${BASE:-/public/home/putianshu/vis_mlp}"
BASELINE_DIR="${BASELINE_DIR:-${BASE}/ifs_baseline}"
RUN_DIR="${RUN_DIR:-${BASE}/paper_eval_results_pm10_pm25_journal/best_effort_source_full_argmax/figure1_all_sources}"
OUT_DIR="${OUT_DIR:-${RUN_DIR}}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/jarvis226/miniconda3/envs/torch/bin/python}"
ANALYZE_SCRIPT="${ANALYZE_SCRIPT:-${BASELINE_DIR}/analyze_best_effort_extended.py}"
PLOT_SCRIPT="${PLOT_SCRIPT:-${BASELINE_DIR}/plot_best_effort_extended_figure.py}"
OPERATIONAL_PLOT_SCRIPT="${OPERATIONAL_PLOT_SCRIPT:-${BASELINE_DIR}/plot_operational_source_ensembles.py}"

mkdir -p "${BASELINE_DIR}/logs" "${OUT_DIR}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: analysis Python is missing: ${PYTHON_BIN}" >&2
  exit 2
fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"

METRICS_TABLE="${RUN_DIR}/best_effort_paired_metrics.csv"
if [[ ! -s "${METRICS_TABLE}" ]]; then
  ENSEMBLE_SAMPLE="${RUN_DIR}/per_sample_tianji_t2nd_ifs_mean_softmax.csv"
  if [[ ! -s "${ENSEMBLE_SAMPLE}" ]]; then
    echo "ERROR: compact analysis tables and ensemble per-sample CSV are both absent." >&2
    echo "Rerun figure1_all_sources; the evaluator now writes compact diagnostics directly." >&2
    exit 2
  fi
  ANALYZE_CMD=(
    "${PYTHON_BIN}" "${ANALYZE_SCRIPT}"
    --sample-dir "${RUN_DIR}"
    --out-dir "${OUT_DIR}"
  )
  echo "Best-effort compact analysis: ${ANALYZE_CMD[*]}"
  "${ANALYZE_CMD[@]}"
fi

PLOT_CMD=(
  "${PYTHON_BIN}" "${PLOT_SCRIPT}"
  --analysis-dir "${OUT_DIR}"
  --out-dir "${OUT_DIR}"
  --formats "svg,pdf,png,tiff"
  --dpi 600
)
echo "Best-effort extended figure: ${PLOT_CMD[*]}"
"${PLOT_CMD[@]}"

if [[ -s "${RUN_DIR}/operational_ensemble_recipe_weights.csv" ]]; then
  OPERATIONAL_PLOT_CMD=(
    "${PYTHON_BIN}" "${OPERATIONAL_PLOT_SCRIPT}"
    --analysis-dir "${RUN_DIR}"
    --out-dir "${OUT_DIR}"
    --formats "svg,pdf,png,tiff"
    --dpi 600
  )
  echo "Operational source-ensemble figure: ${OPERATIONAL_PLOT_CMD[*]}"
  "${OPERATIONAL_PLOT_CMD[@]}"
fi
