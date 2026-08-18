#!/bin/bash
#SBATCH -J viscast_fig3_restructured
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
ANALYSIS_DIR="${ANALYSIS_DIR:-${BASE}/paper_eval_results_pm10_pm25_journal/q_core_t925_factorial/qcore_t925_mhtpw_formal_v2_20260728/analysis}"
ENDPOINT_DIR="${ENDPOINT_DIR:-${BASE}/paper_eval_results_pm10_pm25_journal/q_core_t925_fair/qcore_t925_fair_formal_v1_20260721/analysis}"
TAIL_ANALYSIS_DIR="${TAIL_ANALYSIS_DIR:-${ANALYSIS_DIR}}"
PAIRED_QUALITY_DIR="${PAIRED_QUALITY_DIR:-}"
OUT_DIR="${OUT_DIR:-${BASE}/paper_eval_results_pm10_pm25_journal/fig3_restructured_${SLURM_JOB_ID}}"
FIGURE_STEM="${FIGURE_STEM:-fig3_forecast_source_tail_constraints}"
PLOT_ENV="${PLOT_ENV:-/public/home/jarvis226/miniconda3/envs/torch}"
PLOT_PYTHON="${PLOT_PYTHON:-${PLOT_ENV}/bin/python}"

if [ -z "${PAIRED_QUALITY_DIR}" ]; then
    paired_quality_file="$(
        find "${BASE}" -maxdepth 8 -type f \
            -name pressure_level_paired_rmse_utc_date_bootstrap_ci.csv \
            -printf '%T@ %p\n' 2>/dev/null \
            | sort -nr \
            | head -n 1 \
            | cut -d' ' -f2-
    )"
    test -n "${paired_quality_file}" || {
        echo "ERROR: could not locate paired variable-quality inputs" >&2
        exit 2
    }
    PAIRED_QUALITY_DIR="$(dirname "${paired_quality_file}")"
fi

if [ ! -x "${PLOT_PYTHON}" ]; then
    echo "ERROR: plotting Python is missing: ${PLOT_PYTHON}" >&2
    exit 2
fi
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

for path in \
    "${REPO_ROOT}/plot_qcore_t925_manuscript_composite.py" \
    "${REPO_ROOT}/plot_q_core_task_tail_fidelity_preview.py" \
    "${ENDPOINT_DIR}/qcore_t925_metrics_by_seed.csv" \
    "${ENDPOINT_DIR}/qcore_t925_bootstrap_gap_draws.csv" \
    "${PAIRED_QUALITY_DIR}/pressure_level_paired_rmse_utc_date_bootstrap_ci.csv" \
    "${PAIRED_QUALITY_DIR}/surface_observation_three_source_rmse_utc_date_bootstrap_ci.csv" \
    "${PAIRED_QUALITY_DIR}/surface_observation_pairwise_rmse_delta_utc_date_bootstrap_ci.csv" \
    "${TAIL_ANALYSIS_DIR}/reference_tail_placement_metrics.csv" \
    "${TAIL_ANALYSIS_DIR}/joint_tail_feature_definitions.csv" \
    "${TAIL_ANALYSIS_DIR}/reference_tail_placement_utc_date_bootstrap_ci.csv" \
    "${TAIL_ANALYSIS_DIR}/joint_tail_placement_utc_date_bootstrap_ci.csv" \
    "${TAIL_ANALYSIS_DIR}/joint_tail_placement_metrics.csv"; do
    test -s "${path}" || { echo "ERROR: missing or empty ${path}" >&2; exit 3; }
done

CMD=(
    "${PLOT_PYTHON}" -u "${REPO_ROOT}/plot_qcore_t925_manuscript_composite.py"
    --analysis-dir "${ANALYSIS_DIR}"
    --endpoint-dir "${ENDPOINT_DIR}"
    --tail-analysis-dir "${TAIL_ANALYSIS_DIR}"
    --paired-quality-dir "${PAIRED_QUALITY_DIR}"
    --out-dir "${OUT_DIR}"
    --figure-stem "${FIGURE_STEM}"
    --dpi 600
)

echo "[preflight] plotting imports"
"${PLOT_PYTHON}" -c "import matplotlib, numpy, pandas; print('plotting imports=OK')"
echo "[run] ${CMD[*]}"
"${CMD[@]}"
echo "[done] figure=${OUT_DIR}/${FIGURE_STEM}.png"
