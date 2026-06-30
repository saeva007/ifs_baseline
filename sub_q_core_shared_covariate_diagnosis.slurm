#!/bin/bash
#SBATCH -J qcore_shared_diag
#SBATCH -p kshcexclu04
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -D /public/home/putianshu/vis_mlp/ifs_baseline
#SBATCH -o /public/home/putianshu/vis_mlp/ifs_baseline/logs/%j_%x.out
#SBATCH -e /public/home/putianshu/vis_mlp/ifs_baseline/logs/%j_%x.err

set -euo pipefail

module purge
module load compiler/devtoolset/7.3.1
export CONDA_NO_PLUGINS=true
export CONDA_SOLVER=classic
source /public/home/putianshu/vis_mlp/ifs_baseline/activate_torch_runtime.sh

BASELINE_DIR="${BASELINE_DIR:-/public/home/putianshu/vis_mlp/ifs_baseline}"
LEFT_DIR="${LEFT_DIR:?Set LEFT_DIR to the Tianji q-core dataset}"
RIGHT_DIR="${RIGHT_DIR:?Set RIGHT_DIR to the Pangu q-core dataset}"
LEFT_TAG="${LEFT_TAG:-tianji}"
RIGHT_TAG="${RIGHT_TAG:-pangu2025}"
SPLIT="${SPLIT:-train}"
MAX_COMMON_ROWS="${MAX_COMMON_ROWS:-50000}"
OUT_DIR="${OUT_DIR:-${BASELINE_DIR}/diagnostics/q_core_shared_covariates}"

mkdir -p "${BASELINE_DIR}/logs" "${OUT_DIR}"
cd "${BASELINE_DIR}"

python "${BASELINE_DIR}/diagnose_q_core_shared_covariates.py" \
    --left-dir "${LEFT_DIR}" \
    --right-dir "${RIGHT_DIR}" \
    --left-tag "${LEFT_TAG}" \
    --right-tag "${RIGHT_TAG}" \
    --split "${SPLIT}" \
    --max-common-rows "${MAX_COMMON_ROWS}" \
    --out-dir "${OUT_DIR}"
