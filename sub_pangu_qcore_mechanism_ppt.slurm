#!/bin/bash
#SBATCH -J qcore_story_figs
#SBATCH -p kshcexclu04
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH -t 00:20:00
#SBATCH -D /public/home/putianshu/vis_mlp/ifs_baseline
#SBATCH -o /public/home/putianshu/vis_mlp/ifs_baseline/logs/%j_%x.out
#SBATCH -e /public/home/putianshu/vis_mlp/ifs_baseline/logs/%j_%x.err

set -euo pipefail

BASE="${BASE:-/public/home/putianshu/vis_mlp}"
BASELINE_DIR="${BASELINE_DIR:-${BASE}/ifs_baseline}"
RUN_TAG="${RUN_TAG:-qcore_hybrid_mt2pw_formal_v1_20260708}"
EVAL_ROOT="${EVAL_ROOT:-${BASE}/paper_eval_results_pm10_pm25_journal/q_core_hybrid_factorial/${RUN_TAG}}"
QUALITY_RUN_TAG="${QUALITY_RUN_TAG:-qcore_paired_quality_strictlt_v1_20260716}"
PAIRED_QUALITY_DIR="${PAIRED_QUALITY_DIR:-${BASE}/paper_eval_results_pm10_pm25_journal/q_core_paired_source_quality/${QUALITY_RUN_TAG}/analysis}"
OUT_DIR="${OUT_DIR:-${EVAL_ROOT}/evidence_story_figures_nc_v9}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/jarvis226/miniconda3/envs/torch/bin/python}"
FORMATS="${FORMATS:-svg,pdf,png,tiff}"
DPI="${DPI:-600}"
SURFACE_VIEW="${SURFACE_VIEW:-both}"
UPPER_SCOPE="${UPPER_SCOPE:-all_paired_test}"

mkdir -p "${OUT_DIR}"
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"

# Plotting is CPU-only and deliberately does not import Torch/HIP.
"${PYTHON_BIN}" -c 'import numpy, pandas, matplotlib; print("[plot-env] python/data/plot dependencies OK")'

CMD=(
    "${PYTHON_BIN}"
    "${BASELINE_DIR}/plot_pangu_qcore_mechanism_ppt.py"
    --eval-root "${EVAL_ROOT}"
    --paired-quality-dir "${PAIRED_QUALITY_DIR}"
    --out-dir "${OUT_DIR}"
    --surface-view "${SURFACE_VIEW}"
    --upper-scope "${UPPER_SCOPE}"
    --formats "${FORMATS}"
    --dpi "${DPI}"
)

echo "Command: ${CMD[*]}"
"${CMD[@]}"
