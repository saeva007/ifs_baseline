#!/bin/bash
#SBATCH -J q1000_lineage
#SBATCH -p kshcexclu04
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH -t 12:00:00
#SBATCH -D /public/home/putianshu/vis_mlp/ifs_baseline
#SBATCH -o /public/home/putianshu/vis_mlp/ifs_baseline/logs/%j_%x.out
#SBATCH -e /public/home/putianshu/vis_mlp/ifs_baseline/logs/%j_%x.err

set -euo pipefail

module purge
module load compiler/devtoolset/7.3.1
source /public/home/putianshu/miniconda3/bin/activate
conda activate torch

BASE="${BASE:-/public/home/putianshu/vis_mlp}"
BASELINE_DIR="${BASELINE_DIR:-${BASE}/ifs_baseline}"
OUT_DIR="${OUT_DIR:-${BASE}/paper_eval_results_pm10_pm25_journal/q1000_lineage_audit}"
PANGU_GRID_GLOB="${PANGU_GRID_GLOB:-${BASE}/pangu_2025_china_chunks/pangu_china_2025*_lead*.nc}"
PANGU_STATION_FILE="${PANGU_STATION_FILE:-${BASELINE_DIR}/pangu_station/pangu_station_2025_lead24h.nc}"
TIANJI_FILE="${TIANJI_FILE:-${BASE}/tianji_auto_station/merged_final_all_vars.nc}"
PANGU_DATASET_DIR="${PANGU_DATASET_DIR:-}"
TIANJI_DATASET_DIR="${TIANJI_DATASET_DIR:-}"
IFS_DATASET_DIR="${IFS_DATASET_DIR:-}"
ERA5_DATASET_DIR="${ERA5_DATASET_DIR:-}"
STRICT="${STRICT:-1}"

if ! compgen -G "${PANGU_GRID_GLOB}" >/dev/null; then
    candidate="/data2/share/chenxi/PuTS/mlp/pangu_2025_china_chunks/pangu_china_2025*_lead*.nc"
    if compgen -G "${candidate}" >/dev/null; then
        PANGU_GRID_GLOB="${candidate}"
    fi
fi

mkdir -p "${BASELINE_DIR}/logs" "${OUT_DIR}"
cd "${BASELINE_DIR}"

CMD=(
    python "${BASELINE_DIR}/audit_pangu_tianji_q1000_lineage.py"
    --pangu-grid-glob "${PANGU_GRID_GLOB}"
    --pangu-station-file "${PANGU_STATION_FILE}"
    --tianji-file "${TIANJI_FILE}"
    --out-dir "${OUT_DIR}"
)
for spec in \
    "pangu-dataset-dir=${PANGU_DATASET_DIR}" \
    "tianji-dataset-dir=${TIANJI_DATASET_DIR}" \
    "ifs-dataset-dir=${IFS_DATASET_DIR}" \
    "era5-dataset-dir=${ERA5_DATASET_DIR}"
do
    key="${spec%%=*}"
    value="${spec#*=}"
    if [[ -n "${value}" ]]; then
        CMD+=("--${key}" "${value}")
    fi
done
if [[ "${STRICT}" == "1" || "${STRICT}" == "true" ]]; then
    CMD+=(--strict)
fi

echo "Command: ${CMD[*]}"
"${CMD[@]}"
