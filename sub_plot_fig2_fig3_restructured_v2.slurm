#!/bin/bash
#SBATCH -J viscast_fig2_fig3_v2
#SBATCH -p kshcexclu04
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -D /public/home/putianshu/vis_mlp/ifs_baseline
#SBATCH -o /public/home/putianshu/vis_mlp/ifs_baseline/logs/%j_%x.out
#SBATCH -e /public/home/putianshu/vis_mlp/ifs_baseline/logs/%j_%x.err

set -euo pipefail

BASE="${BASE:-/public/home/putianshu/vis_mlp}"
REPO_ROOT="${REPO_ROOT:-${BASE}/ifs_baseline}"
MAPPING_CV_DIR="${MAPPING_CV_DIR:-${BASE}/train}"
WHICH="${WHICH:-both}"
OUT_DIR="${OUT_DIR:-${BASE}/paper_eval_results_pm10_pm25_journal/fig2_fig3_restructured_v2}"
FIG2_STEM="${FIG2_STEM:-fig2_restructured_v2}"
FIG3_STEM="${FIG3_STEM:-fig3_restructured_v2}"

IFS_COMPARISON_DIR="${IFS_COMPARISON_DIR:-}"
SPATIAL_RESULT_ROOT="${SPATIAL_RESULT_ROOT:-}"
TEMPORAL_RESULT_ROOT="${TEMPORAL_RESULT_ROOT:-}"

ANALYSIS_DIR="${ANALYSIS_DIR:-${BASE}/paper_eval_results_pm10_pm25_journal/q_core_t925_factorial/qcore_t925_mhtpw_formal_v2_20260728/analysis}"
ENDPOINT_DIR="${ENDPOINT_DIR:-${BASE}/paper_eval_results_pm10_pm25_journal/q_core_t925_fair/qcore_t925_fair_formal_v1_20260721/analysis}"
TAIL_ANALYSIS_DIR="${TAIL_ANALYSIS_DIR:-${BASE}/paper_eval_results_pm10_pm25_journal/q_core_task_tail_fidelity/qcore_task_tail_fidelity_v1_20260811/analysis}"

PLOT_ENV="${PLOT_ENV:-/public/home/jarvis226/miniconda3/envs/torch}"
PLOT_PYTHON="${PLOT_PYTHON:-${PLOT_ENV}/bin/python}"
PLOT_SCRIPT="${PLOT_SCRIPT:-${REPO_ROOT}/plot_fig2_fig3_restructured_v2.py}"

if [ ! -x "${PLOT_PYTHON}" ]; then
    echo "ERROR: plotting Python is missing: ${PLOT_PYTHON}" >&2
    exit 2
fi
if [ ! -s "${PLOT_SCRIPT}" ]; then
    echo "ERROR: missing or empty ${PLOT_SCRIPT}" >&2
    exit 2
fi
case "${WHICH}" in
    both|fig2|fig3) ;;
    *) echo "ERROR: invalid WHICH=${WHICH} (use both|fig2|fig3)" >&2; exit 2 ;;
esac

unset LD_PRELOAD
export PATH="${PLOT_ENV}/bin:${PATH}"
export LD_LIBRARY_PATH="${PLOT_ENV}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MPLCONFIGDIR="${OUT_DIR}/.matplotlib"

mkdir -p "${REPO_ROOT}/logs" "${OUT_DIR}" "${MPLCONFIGDIR}"
cd "${REPO_ROOT}"

if [[ "${WHICH}" == "both" || "${WHICH}" == "fig2" ]]; then
    if [ -z "${IFS_COMPARISON_DIR}" ] || [ -z "${SPATIAL_RESULT_ROOT}" ] || [ -z "${TEMPORAL_RESULT_ROOT}" ]; then
        echo "ERROR: fig2 requires IFS_COMPARISON_DIR, SPATIAL_RESULT_ROOT, TEMPORAL_RESULT_ROOT" >&2
        exit 2
    fi
    for path in \
        "${IFS_COMPARISON_DIR}/ifs_diagnostic_matched_metrics.csv" \
        "${SPATIAL_RESULT_ROOT}/aggregate/fold_metrics.csv" \
        "${SPATIAL_RESULT_ROOT}/aggregate/pooled_metrics.csv" \
        "${SPATIAL_RESULT_ROOT}/aggregate/coverage_manifest.json" \
        "${SPATIAL_RESULT_ROOT}/folds/fold_manifest.json" \
        "${TEMPORAL_RESULT_ROOT}/aggregate/fold_metrics.csv" \
        "${TEMPORAL_RESULT_ROOT}/aggregate/pooled_metrics.csv" \
        "${TEMPORAL_RESULT_ROOT}/aggregate/coverage_manifest.json" \
        "${TEMPORAL_RESULT_ROOT}/folds/fold_manifest.json"; do
        test -s "${path}" || { echo "ERROR: missing or empty ${path}" >&2; exit 3; }
    done
fi

if [[ "${WHICH}" == "both" || "${WHICH}" == "fig3" ]]; then
    for path in \
        "${ENDPOINT_DIR}/qcore_t925_metrics_by_seed.csv" \
        "${ENDPOINT_DIR}/qcore_t925_bootstrap_gap_draws.csv" \
        "${TAIL_ANALYSIS_DIR}/reference_tail_placement_metrics.csv" \
        "${TAIL_ANALYSIS_DIR}/joint_tail_feature_definitions.csv" \
        "${TAIL_ANALYSIS_DIR}/reference_tail_placement_utc_date_bootstrap_ci.csv" \
        "${TAIL_ANALYSIS_DIR}/joint_tail_placement_utc_date_bootstrap_ci.csv" \
        "${TAIL_ANALYSIS_DIR}/joint_tail_placement_metrics.csv"; do
        test -s "${path}" || { echo "ERROR: missing or empty ${path}" >&2; exit 3; }
    done
fi

CMD=(
    "${PLOT_PYTHON}" -u "${PLOT_SCRIPT}"
    --which "${WHICH}"
    --output-dir "${OUT_DIR}"
    --mapping-cv-dir "${MAPPING_CV_DIR}"
    --fig2-stem "${FIG2_STEM}"
    --fig3-stem "${FIG3_STEM}"
)
if [[ -n "${IFS_COMPARISON_DIR}" ]]; then
    CMD+=(--fig2-ifs-comparison-dir "${IFS_COMPARISON_DIR}")
fi
if [[ -n "${SPATIAL_RESULT_ROOT}" ]]; then
    CMD+=(--fig2-spatial-result-root "${SPATIAL_RESULT_ROOT}")
fi
if [[ -n "${TEMPORAL_RESULT_ROOT}" ]]; then
    CMD+=(--fig2-temporal-result-root "${TEMPORAL_RESULT_ROOT}")
fi
CMD+=(
    --fig3-analysis-dir "${ANALYSIS_DIR}"
    --fig3-endpoint-dir "${ENDPOINT_DIR}"
    --fig3-tail-analysis-dir "${TAIL_ANALYSIS_DIR}"
)

echo "[preflight] plotting imports"
"${PLOT_PYTHON}" -c "import matplotlib, numpy, pandas; print('plotting imports=OK')"
echo "[run] ${CMD[*]}"
"${CMD[@]}"
echo "[done] outputs=${OUT_DIR}"
