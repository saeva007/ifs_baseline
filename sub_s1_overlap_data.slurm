#!/bin/bash
#SBATCH -J s1_overlap_data
#SBATCH -p kshcexclu04
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH -t 48:00:00
#SBATCH -D /public/home/putianshu/vis_mlp/ifs_baseline
#SBATCH -o /public/home/putianshu/vis_mlp/ifs_baseline/logs/%j_%x.out
#SBATCH -e /public/home/putianshu/vis_mlp/ifs_baseline/logs/%j_%x.err
#SBATCH --exclusive

set -euo pipefail

echo "S1 overlap data build started at: $(date)"
echo "Node: $(hostname)"

module purge
module load compiler/devtoolset/7.3.1
source /public/home/putianshu/miniconda3/bin/activate
conda activate torch

cd /public/home/putianshu/vis_mlp/ifs_baseline
mkdir -p logs

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"

SCRIPT_PATH="/public/home/putianshu/vis_mlp/ifs_baseline/build_s1_pm10_overlap_from_full.py"
SOURCE_DIR="${SOURCE_DIR:-}"
OUT_DIR="${OUT_DIR:-}"
FEATURE_SET="${FEATURE_SET:-overlap_full}"
CHUNK_ROWS="${CHUNK_ROWS:-}"
MERGE_TRAIN_VAL="${MERGE_TRAIN_VAL:-0}"
DATA_EXTRA_ARGS="${DATA_EXTRA_ARGS:-}"

if [[ -z "${OUT_DIR}" ]]; then
    if [[ "${FEATURE_SET}" == "common_core" ]]; then
        OUT_DIR="/public/home/putianshu/vis_mlp/ifs_baseline/ml_dataset_pmst_v5_aligned_12h_pm10_pm25_common_core"
    elif [[ "${FEATURE_SET}" == "compact_common_core" ]]; then
        OUT_DIR="/public/home/putianshu/vis_mlp/ifs_baseline/ml_dataset_pmst_v5_aligned_12h_pm10_pm25_compact_common_core"
    elif [[ "${FEATURE_SET}" == "source_full" ]]; then
        OUT_DIR="/public/home/putianshu/vis_mlp/ifs_baseline/ml_dataset_pmst_v5_aligned_12h_pm10_pm25_source_full"
    else
        OUT_DIR="/public/home/putianshu/vis_mlp/ifs_baseline/ml_dataset_pmst_v5_aligned_12h_pm10_pm25_overlap"
    fi
fi

if [[ -z "${SOURCE_DIR}" ]]; then
    if [[ "${FEATURE_SET}" == "common_core" ]]; then
        SOURCE_DIR="/public/home/putianshu/vis_mlp/ifs_baseline/ml_dataset_pmst_v5_aligned_12h_pm10_pm25_overlap"
    else
        SOURCE_DIR="/public/home/putianshu/vis_mlp/ml_dataset_pmst_v5_aligned_12h_pm10_pm25"
    fi
fi

PY_ARGS=()
PY_ARGS+=(--feature_set "${FEATURE_SET}")
PY_ARGS+=(--source_dir "${SOURCE_DIR}")
PY_ARGS+=(--out_dir "${OUT_DIR}")
if [[ -n "${CHUNK_ROWS}" ]]; then
    PY_ARGS+=(--chunk_rows "${CHUNK_ROWS}")
fi
if [[ "${MERGE_TRAIN_VAL}" == "1" || "${MERGE_TRAIN_VAL}" == "true" ]]; then
    PY_ARGS+=(--merge_train_val)
fi

echo "Running Python script: ${SCRIPT_PATH}"
if [[ -n "${DATA_EXTRA_ARGS}" ]]; then
    # shellcheck disable=SC2206
    EXTRA_TOKENS=( ${DATA_EXTRA_ARGS} )
    PY_ARGS+=("${EXTRA_TOKENS[@]}")
fi

if ((${#PY_ARGS[@]} > 0)); then
    echo "PY_ARGS=${PY_ARGS[*]}"
    python "${SCRIPT_PATH}" "${PY_ARGS[@]}"
else
    echo "PY_ARGS="
    python "${SCRIPT_PATH}"
fi

echo "S1 overlap data build finished at: $(date)"
