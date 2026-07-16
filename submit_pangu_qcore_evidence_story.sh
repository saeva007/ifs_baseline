#!/bin/bash
# Submit the final zero-training quality refresh and, only after it succeeds,
# render the complete q-core evidence story.  The factorial checkpoints and
# inference outputs are reused; this chain does not train any model.

set -euo pipefail

BASE="${BASE:-/public/home/putianshu/vis_mlp}"
BASELINE_DIR="${BASELINE_DIR:-${BASE}/ifs_baseline}"
RUN_TAG="${RUN_TAG:-qcore_hybrid_mt2pw_formal_v1_20260708}"
QUALITY_RUN_TAG="${QUALITY_RUN_TAG:-qcore_paired_quality_strictlt_v1_20260716}"
QUALITY_SCOPE="${QUALITY_SCOPE:-true_low_visibility}"
UPPER_SCOPE="${UPPER_SCOPE:-all_paired_test}"
FORMATS="${FORMATS:-svg,pdf,png,tiff}"
DPI="${DPI:-600}"
DRY_RUN="${DRY_RUN:-0}"

EVAL_ROOT="${EVAL_ROOT:-${BASE}/paper_eval_results_pm10_pm25_journal/q_core_hybrid_factorial/${RUN_TAG}}"
PAIRED_QUALITY_DIR="${PAIRED_QUALITY_DIR:-${BASE}/paper_eval_results_pm10_pm25_journal/q_core_paired_source_quality/${QUALITY_RUN_TAG}/analysis}"
OUT_DIR="${OUT_DIR:-${EVAL_ROOT}/evidence_story_figures}"

cd "${BASELINE_DIR}"
mkdir -p logs

if [[ "${DRY_RUN}" == "1" ]]; then
  RUN_TAG="${QUALITY_RUN_TAG}" DRY_RUN=1 \
    bash "${BASELINE_DIR}/submit_q_core_paired_source_quality.sh"
  printf '[DRY-RUN] sbatch --dependency=afterok:<quality_job> --export=%q %q\n' \
    "ALL,RUN_TAG=${RUN_TAG},QUALITY_RUN_TAG=${QUALITY_RUN_TAG},EVAL_ROOT=${EVAL_ROOT},PAIRED_QUALITY_DIR=${PAIRED_QUALITY_DIR},OUT_DIR=${OUT_DIR},QUALITY_SCOPE=${QUALITY_SCOPE},UPPER_SCOPE=${UPPER_SCOPE},FORMATS=${FORMATS},DPI=${DPI}" \
    "${BASELINE_DIR}/sub_pangu_qcore_mechanism_ppt.slurm"
  exit 0
fi

quality_output="$(RUN_TAG="${QUALITY_RUN_TAG}" \
  bash "${BASELINE_DIR}/submit_q_core_paired_source_quality.sh")"
printf '%s\n' "${quality_output}"
quality_job="$(printf '%s\n' "${quality_output}" | sed -n 's/^analysis_job=//p' | tail -n 1)"
[[ "${quality_job}" =~ ^[0-9]+$ ]] || {
  echo "ERROR: failed to parse paired-quality Slurm job id" >&2
  exit 2
}

plot_job="$(sbatch --parsable --dependency="afterok:${quality_job}" \
  --export="ALL,RUN_TAG=${RUN_TAG},QUALITY_RUN_TAG=${QUALITY_RUN_TAG},EVAL_ROOT=${EVAL_ROOT},PAIRED_QUALITY_DIR=${PAIRED_QUALITY_DIR},OUT_DIR=${OUT_DIR},QUALITY_SCOPE=${QUALITY_SCOPE},UPPER_SCOPE=${UPPER_SCOPE},FORMATS=${FORMATS},DPI=${DPI}" \
  "${BASELINE_DIR}/sub_pangu_qcore_mechanism_ppt.slurm")"
plot_job="${plot_job%%;*}"
[[ "${plot_job}" =~ ^[0-9]+$ ]] || {
  echo "ERROR: invalid plotting Slurm job id: ${plot_job}" >&2
  exit 2
}

echo "quality_job=${quality_job}"
echo "plot_job=${plot_job}"
echo "results=${OUT_DIR}"
