#!/bin/bash
#SBATCH -J station_source_data
#SBATCH --partition=cpu_parallel
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH -t 48:00:00
#SBATCH -D /data2/share/chenxi/PuTS/mlp/ifs_baseline
#SBATCH -o /data2/share/chenxi/PuTS/mlp/ifs_baseline/logs/%j_%x.out
#SBATCH -e /data2/share/chenxi/PuTS/mlp/ifs_baseline/logs/%j_%x.err
#SBATCH --exclusive

set -euo pipefail

echo "Station-source overlap data build started at: $(date)"
echo "Node: $(hostname)"

source /lustre/chenxi/yliang/miniforge3/bin/activate
module purge
module load compiler/dtk/23.10
module load compiler/devtoolset/7.3.1
module load mpi/hpcx/2.7.4/gcc-7.3.1
source /lustre/chenxi/yliang/work/cast3/env.sh

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"

WORK_DIR="${WORK_DIR:-/data2/share/chenxi/PuTS/mlp}"
cd "${WORK_DIR}/ifs_baseline"
mkdir -p logs

SCRIPT_PATH="${SCRIPT_PATH:-${WORK_DIR}/ifs_baseline/build_dataset_station_source_overlap_12h.py}"
SOURCE_KIND="${SOURCE_KIND:-station_nc}"
SOURCE_TAG="${SOURCE_TAG:-pangu2021}"
SOURCE_FILE="${SOURCE_FILE:-}"
SOURCE_GLOB="${SOURCE_GLOB:-}"
FEATURE_DIR="${FEATURE_DIR:-}"
YEAR="${YEAR:-2021}"
TARGET_FILE="${TARGET_FILE:-}"
FEATURE_SET="${FEATURE_SET:-common_core}"
FEATURE_SUFFIX="${FEATURE_SET}"
if [[ "${FEATURE_SET}" == compact_common_core* ]]; then
    FEATURE_SUFFIX="compact_common_core_no_rh2m"
fi
OUT_DIR="${OUT_DIR:-}"
PM10_FILE="${PM10_FILE:-}"
PM10_DIR="${PM10_DIR:-}"
PM25_FILE="${PM25_FILE:-}"
PM25_DIR="${PM25_DIR:-}"
VEG_FILE="${VEG_FILE:-}"
ORO_FILE="${ORO_FILE:-}"
CHUNK_WINS="${CHUNK_WINS:-}"
STAGING_DIR="${STAGING_DIR:-}"
KEEP_STAGING="${KEEP_STAGING:-0}"
DATA_EXTRA_ARGS="${DATA_EXTRA_ARGS:-}"

if [[ "${SOURCE_TAG}" =~ ^pangu([0-9]{4})$ ]]; then
    YEAR="${BASH_REMATCH[1]}"
fi
if [[ -z "${SOURCE_FILE}" && "${SOURCE_KIND}" == "station_nc" && "${SOURCE_TAG}" == pangu* ]]; then
    SOURCE_FILE="${WORK_DIR}/ifs_baseline/pangu_station/pangu_station_${YEAR}_lead24h.nc"
fi
if [[ -z "${TARGET_FILE}" ]]; then
    if [[ "${YEAR}" == "2025" ]]; then
        TARGET_FILE="${WORK_DIR}/tianji_auto_station/merged_final_all_vars.nc"
    else
        TARGET_FILE="${WORK_DIR}/CMA_visibility_2021_2023_GeoCoords_1.nc"
    fi
fi
if [[ -z "${OUT_DIR}" ]]; then
    OUT_DIR="${WORK_DIR}/ifs_baseline/ml_dataset_overlap_${SOURCE_TAG}_12h_pm10_pm25_${FEATURE_SUFFIX}"
fi
if [[ -z "${VEG_FILE}" ]]; then
    if [[ -f "${WORK_DIR}/data_vegtype.nc" ]]; then
        VEG_FILE="${WORK_DIR}/data_vegtype.nc"
    elif [[ -f "/data2/share/chenxi/PuTS/tianji/data_vegtype.nc" ]]; then
        VEG_FILE="/data2/share/chenxi/PuTS/tianji/data_vegtype.nc"
    fi
fi
if [[ -z "${ORO_FILE}" ]]; then
    if [[ -f "${WORK_DIR}/data_orography.nc" ]]; then
        ORO_FILE="${WORK_DIR}/data_orography.nc"
    elif [[ -f "/data2/share/chenxi/PuTS/tianji/data_orography.nc" ]]; then
        ORO_FILE="/data2/share/chenxi/PuTS/tianji/data_orography.nc"
    fi
fi

PY_ARGS=(--source_kind "${SOURCE_KIND}" --source_tag "${SOURCE_TAG}" --year "${YEAR}" --target_file "${TARGET_FILE}" --feature_set "${FEATURE_SET}" --out_dir "${OUT_DIR}")
if [[ -n "${SOURCE_FILE}" ]]; then
    PY_ARGS+=(--source_file "${SOURCE_FILE}")
fi
if [[ -n "${SOURCE_GLOB}" ]]; then
    PY_ARGS+=(--source_glob "${SOURCE_GLOB}")
fi
if [[ -n "${FEATURE_DIR}" ]]; then
    PY_ARGS+=(--feature_dir "${FEATURE_DIR}")
fi
if [[ -n "${PM10_FILE}" ]]; then
    PY_ARGS+=(--pm10_file "${PM10_FILE}")
fi
if [[ -n "${PM10_DIR}" ]]; then
    PY_ARGS+=(--pm10_dir "${PM10_DIR}")
fi
if [[ -n "${PM25_FILE}" ]]; then
    PY_ARGS+=(--pm25_file "${PM25_FILE}")
fi
if [[ -n "${PM25_DIR}" ]]; then
    PY_ARGS+=(--pm25_dir "${PM25_DIR}")
fi
if [[ -n "${VEG_FILE}" ]]; then
    PY_ARGS+=(--veg_file "${VEG_FILE}")
fi
if [[ -n "${ORO_FILE}" ]]; then
    PY_ARGS+=(--oro_file "${ORO_FILE}")
fi
if [[ -n "${CHUNK_WINS}" ]]; then
    PY_ARGS+=(--chunk_wins "${CHUNK_WINS}")
fi
if [[ -n "${STAGING_DIR}" ]]; then
    PY_ARGS+=(--staging_dir "${STAGING_DIR}")
fi
if [[ "${KEEP_STAGING}" == "1" || "${KEEP_STAGING}" == "true" ]]; then
    PY_ARGS+=(--keep_staging)
fi
if [[ -n "${DATA_EXTRA_ARGS}" ]]; then
    # shellcheck disable=SC2206
    EXTRA_TOKENS=( ${DATA_EXTRA_ARGS} )
    PY_ARGS+=("${EXTRA_TOKENS[@]}")
fi

echo "PY_ARGS=${PY_ARGS[*]}"
python "${SCRIPT_PATH}" "${PY_ARGS[@]}"

echo "Station-source overlap data build finished at: $(date)"
