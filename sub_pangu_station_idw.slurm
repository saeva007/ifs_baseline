#!/bin/bash
#SBATCH -J pangu_station_idw
#SBATCH -p kshcexclu04
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH -t 24:00:00
#SBATCH -D /public/home/putianshu/vis_mlp/ifs_baseline
#SBATCH -o /public/home/putianshu/vis_mlp/ifs_baseline/logs/%j_%x.out
#SBATCH -e /public/home/putianshu/vis_mlp/ifs_baseline/logs/%j_%x.err
#SBATCH --exclusive

set -euo pipefail

echo "Pangu station interpolation started at: $(date)"
echo "Node: $(hostname)"

module purge
module load compiler/devtoolset/7.3.1
export CONDA_NO_PLUGINS=true
export CONDA_SOLVER=classic
source /public/home/putianshu/miniconda3/bin/activate
conda activate torch

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"

cd /public/home/putianshu/vis_mlp/ifs_baseline
mkdir -p logs

SCRIPT_PATH="${SCRIPT_PATH:-/public/home/putianshu/vis_mlp/ifs_baseline/interpolate_pangu_to_stations.py}"
INPUT_DIR="${INPUT_DIR:-/public/home/putianshu/vis_mlp/pangu_2021_china_monthly}"
INPUT_GLOB="${INPUT_GLOB:-}"
INPUT_FILES="${INPUT_FILES:-}"
TARGET_FILE="${TARGET_FILE:-/public/home/putianshu/vis_mlp/CMA_visibility_2021_2023_GeoCoords_1.nc}"
YEAR="${YEAR:-2021}"
OUT_FILE="${OUT_FILE:-/public/home/putianshu/vis_mlp/ifs_baseline/pangu_station/pangu_station_2021_lead24h.nc}"
K="${K:-4}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

CMD=(python "${SCRIPT_PATH}" --input_dir "${INPUT_DIR}" --target_file "${TARGET_FILE}" --year "${YEAR}" --out_file "${OUT_FILE}" --k "${K}")
if [[ -n "${INPUT_GLOB}" ]]; then
    CMD+=(--input_glob "${INPUT_GLOB}")
fi
if [[ -n "${INPUT_FILES}" ]]; then
    CMD+=(--input_files "${INPUT_FILES}")
fi
if [[ -n "${EXTRA_ARGS}" ]]; then
    # shellcheck disable=SC2206
    EXTRA_ARGV=(${EXTRA_ARGS})
    CMD+=("${EXTRA_ARGV[@]}")
fi

echo "Command: ${CMD[*]}"
"${CMD[@]}"

echo "Pangu station interpolation finished at: $(date)"
