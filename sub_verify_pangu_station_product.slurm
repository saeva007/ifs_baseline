#!/bin/bash
#SBATCH -J verify_pangu_station
#SBATCH -p kshcexclu04
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
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

BASE="${BASE:-/public/home/putianshu/vis_mlp}"
BASELINE_DIR="${BASELINE_DIR:-${BASE}/ifs_baseline}"
OLD_PANGU_STATION_FILE="${OLD_PANGU_STATION_FILE:?OLD_PANGU_STATION_FILE is required}"
PANGU2025_STATION_FILE="${PANGU2025_STATION_FILE:?PANGU2025_STATION_FILE is required}"
TARGET_FILE="${TARGET_FILE:-${BASE}/tianji_auto_station/merged_final_all_vars.nc}"
EXPECTED_LEAD_MIN_HOURS="${EXPECTED_LEAD_MIN_HOURS:?EXPECTED_LEAD_MIN_HOURS is required}"
EXPECTED_LEAD_MAX_HOURS="${EXPECTED_LEAD_MAX_HOURS:?EXPECTED_LEAD_MAX_HOURS is required}"
VERIFY_OUT_JSON="${VERIFY_OUT_JSON:?VERIFY_OUT_JSON is required}"
INFER_PANGU_LEAD12_23_FROM_VALID_TIME="${INFER_PANGU_LEAD12_23_FROM_VALID_TIME:-0}"

mkdir -p "${BASELINE_DIR}/logs" "$(dirname "${VERIFY_OUT_JSON}")"
cd "${BASELINE_DIR}"

CMD=(
    python "${BASELINE_DIR}/verify_pangu_station_product.py"
    --old-file "${OLD_PANGU_STATION_FILE}"
    --new-file "${PANGU2025_STATION_FILE}"
    --target-file "${TARGET_FILE}"
    --out-json "${VERIFY_OUT_JSON}"
    --year 2025
    --expected-lead-min-hours "${EXPECTED_LEAD_MIN_HOURS}"
    --expected-lead-max-hours "${EXPECTED_LEAD_MAX_HOURS}"
)
if [[ "${INFER_PANGU_LEAD12_23_FROM_VALID_TIME}" == "1" || "${INFER_PANGU_LEAD12_23_FROM_VALID_TIME}" == "true" ]]; then
    CMD+=(--infer-stitched-lead12-23)
fi
echo "Command: ${CMD[*]}"
"${CMD[@]}"
