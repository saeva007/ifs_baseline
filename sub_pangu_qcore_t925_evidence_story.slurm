#!/bin/bash
#SBATCH -J qct925_story
#SBATCH -p kshcexclu04
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -D /public/home/putianshu/vis_mlp/ifs_baseline
#SBATCH -o /public/home/putianshu/vis_mlp/ifs_baseline/logs/%j_%x.out
#SBATCH -e /public/home/putianshu/vis_mlp/ifs_baseline/logs/%j_%x.err

set -euo pipefail

BASE="${BASE:-/public/home/putianshu/vis_mlp}"
BASELINE_DIR="${BASELINE_DIR:-${BASE}/ifs_baseline}"
SOURCE_RUN_TAG="${SOURCE_RUN_TAG:-qcore_t925_fair_formal_v1_20260721}"
UPPER_RUN_TAG="${UPPER_RUN_TAG:-qcore_t925_upper_air_disagreement_v1_20260722}"
FAIR_EVAL_ROOT="${FAIR_EVAL_ROOT:-${BASE}/paper_eval_results_pm10_pm25_journal/q_core_t925_fair/${SOURCE_RUN_TAG}}"
PAIRED_QUALITY_DIR="${PAIRED_QUALITY_DIR:-${BASE}/paper_eval_results_pm10_pm25_journal/q_core_paired_source_quality/qcore_paired_quality_strictlt_v1_20260716/analysis}"
EVENT_ANALYSIS_DIR="${EVENT_ANALYSIS_DIR:-${BASE}/paper_eval_results_pm10_pm25_journal/q_core_t925_upper_air_disagreement/${UPPER_RUN_TAG}/analysis}"
OUT_DIR="${OUT_DIR:-${FAIR_EVAL_ROOT}/evidence_story_figures_t925_nc_v1}"
FORMATS="${FORMATS:-svg:pdf:png:tiff}"
DPI="${DPI:-600}"

mkdir -p "${BASELINE_DIR}/logs"
cd "${BASELINE_DIR}"
source "${BASELINE_DIR}/activate_eval_torch_runtime.sh"
export PYTHONUNBUFFERED=1

CMD=(
  python "${BASELINE_DIR}/plot_pangu_qcore_t925_evidence_story.py"
  --fair-eval-root "${FAIR_EVAL_ROOT}"
  --paired-quality-dir "${PAIRED_QUALITY_DIR}"
  --event-analysis-dir "${EVENT_ANALYSIS_DIR}"
  --out-dir "${OUT_DIR}"
  --formats "${FORMATS}"
  --dpi "${DPI}"
  --surface-view both
)

echo "Q-core+T925 evidence-story redraw started at: $(date)"
echo "FAIR_EVAL_ROOT=${FAIR_EVAL_ROOT}"
echo "EVENT_ANALYSIS_DIR=${EVENT_ANALYSIS_DIR}"
echo "OUT_DIR=${OUT_DIR}"
echo "Shapley is intentionally omitted until the MHTPW factorial is complete."
echo "Command: ${CMD[*]}"
"${CMD[@]}"
echo "Q-core+T925 evidence-story redraw completed at: $(date)"
