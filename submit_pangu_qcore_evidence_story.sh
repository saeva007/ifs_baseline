#!/bin/bash
# Submit the final zero-training quality refresh and, only after it succeeds,
# render the complete q-core evidence story.  The factorial checkpoints and
# inference outputs are reused; this chain does not train any model.

set -euo pipefail

BASE="${BASE:-/public/home/putianshu/vis_mlp}"
BASELINE_DIR="${BASELINE_DIR:-${BASE}/ifs_baseline}"
RUN_TAG="${RUN_TAG:-qcore_hybrid_mt2pw_formal_v1_20260708}"
QUALITY_RUN_TAG="${QUALITY_RUN_TAG:-qcore_paired_quality_strictlt_v1_20260716}"
SURFACE_VIEW="${SURFACE_VIEW:-both}"
UPPER_SCOPE="${UPPER_SCOPE:-all_paired_test}"
FORMATS="${FORMATS:-svg,pdf,png,tiff}"
DPI="${DPI:-600}"
DRY_RUN="${DRY_RUN:-0}"
REUSE_QUALITY="${REUSE_QUALITY:-auto}"
SBATCH_BIN="${SBATCH_BIN:-sbatch}"

EVAL_ROOT="${EVAL_ROOT:-${BASE}/paper_eval_results_pm10_pm25_journal/q_core_hybrid_factorial/${RUN_TAG}}"
PAIRED_QUALITY_DIR="${PAIRED_QUALITY_DIR:-${BASE}/paper_eval_results_pm10_pm25_journal/q_core_paired_source_quality/${QUALITY_RUN_TAG}/analysis}"
OUT_DIR="${OUT_DIR:-${EVAL_ROOT}/evidence_story_figures_nc_v5}"
# Slurm uses commas to separate --export entries, so a comma-delimited format
# value would be truncated to "svg".  Export a colon-delimited value and let
# the Python entrypoint accept both delimiters.
FORMATS_EXPORT="${FORMATS//,/:}"
PLOT_EXPORTS="ALL,RUN_TAG=${RUN_TAG},QUALITY_RUN_TAG=${QUALITY_RUN_TAG},EVAL_ROOT=${EVAL_ROOT},PAIRED_QUALITY_DIR=${PAIRED_QUALITY_DIR},OUT_DIR=${OUT_DIR},SURFACE_VIEW=${SURFACE_VIEW},UPPER_SCOPE=${UPPER_SCOPE},FORMATS=${FORMATS_EXPORT},DPI=${DPI}"

cd "${BASELINE_DIR}"
mkdir -p logs

if [[ "${DRY_RUN}" == "1" ]]; then
  if [[ "${REUSE_QUALITY}" == "auto" && -s "${PAIRED_QUALITY_DIR}/paired_source_quality_report.json" ]]; then
    echo "[DRY-RUN] reusing paired quality: ${PAIRED_QUALITY_DIR}"
    printf '[DRY-RUN] sbatch --export=%q %q\n' \
      "${PLOT_EXPORTS}" "${BASELINE_DIR}/sub_pangu_qcore_mechanism_ppt.slurm"
  else
    RUN_TAG="${QUALITY_RUN_TAG}" DRY_RUN=1 \
      bash "${BASELINE_DIR}/submit_q_core_paired_source_quality.sh"
    printf '[DRY-RUN] sbatch --dependency=afterok:<quality_job> --export=%q %q\n' \
      "${PLOT_EXPORTS}" "${BASELINE_DIR}/sub_pangu_qcore_mechanism_ppt.slurm"
  fi
  exit 0
fi

quality_job="reused"
if [[ "${REUSE_QUALITY}" == "auto" && -s "${PAIRED_QUALITY_DIR}/paired_source_quality_report.json" ]]; then
  echo "Reusing completed paired-quality directory: ${PAIRED_QUALITY_DIR}"
else
  quality_output="$(RUN_TAG="${QUALITY_RUN_TAG}" \
    bash "${BASELINE_DIR}/submit_q_core_paired_source_quality.sh")"
  printf '%s\n' "${quality_output}"
  quality_job="$(printf '%s\n' "${quality_output}" | sed -n 's/^analysis_job=//p' | tail -n 1)"
  [[ "${quality_job}" =~ ^[0-9]+$ ]] || {
    echo "ERROR: failed to parse paired-quality Slurm job id" >&2
    exit 2
  }
fi

if [[ "${quality_job}" == "reused" ]]; then
  plot_job="$("${SBATCH_BIN}" --parsable \
    --export="${PLOT_EXPORTS}" "${BASELINE_DIR}/sub_pangu_qcore_mechanism_ppt.slurm")"
else
  plot_job="$("${SBATCH_BIN}" --parsable --dependency="afterok:${quality_job}" \
    --export="${PLOT_EXPORTS}" "${BASELINE_DIR}/sub_pangu_qcore_mechanism_ppt.slurm")"
fi
plot_job="${plot_job%%;*}"
[[ "${plot_job}" =~ ^[0-9]+$ ]] || {
  echo "ERROR: invalid plotting Slurm job id: ${plot_job}" >&2
  exit 2
}

echo "quality_job=${quality_job}"
echo "plot_job=${plot_job}"
echo "results=${OUT_DIR}"
