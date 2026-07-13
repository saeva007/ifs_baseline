#!/bin/bash
#SBATCH -J qcore_ppt_figs
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
OUT_DIR="${OUT_DIR:-${EVAL_ROOT}/ppt_figures}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/jarvis226/miniconda3/envs/torch/bin/python}"
FORMATS="${FORMATS:-svg,pdf,png}"
DPI="${DPI:-300}"

mkdir -p "${OUT_DIR}"
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"

# Plotting is CPU-only and deliberately does not import Torch/HIP.
"${PYTHON_BIN}" -c 'import numpy, pandas, matplotlib; print("[plot-env] python/data/plot dependencies OK")'

CMD=(
    "${PYTHON_BIN}"
    "${BASELINE_DIR}/plot_pangu_qcore_mechanism_ppt.py"
    --eval-root "${EVAL_ROOT}"
    --out-dir "${OUT_DIR}"
    --formats "${FORMATS}"
    --dpi "${DPI}"
)

echo "Command: ${CMD[*]}"
"${CMD[@]}"
