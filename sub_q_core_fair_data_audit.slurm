#!/bin/bash
#SBATCH -J qcore_data_audit
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

module purge
module load compiler/devtoolset/7.3.1
export CONDA_NO_PLUGINS=true
export CONDA_SOLVER=classic
source /public/home/putianshu/vis_mlp/ifs_baseline/activate_torch_runtime.sh

BASE="${BASE:-/public/home/putianshu/vis_mlp}"
BASELINE_DIR="${BASELINE_DIR:-${BASE}/ifs_baseline}"
RUN_TAG="${RUN_TAG:-q_core_no_rh2m_pangu2025}"
OUT_DIR="${AUDIT_OUT_DIR:-${BASE}/paper_eval_results_pm10_pm25_journal/q_core_fair_pangu2025/${RUN_TAG}/data_audit}"
S1_DATA_DIR="${S1_DATA_DIR:-${BASELINE_DIR}/ml_dataset_pmst_v5_aligned_12h_pm10_pm25_q_core_no_rh2m}"
TIANJI_DATA_DIR="${TIANJI_DATA_DIR:-${BASELINE_DIR}/ml_dataset_overlap_tianji_12h_pm10_pm25_q_core_no_rh2m}"
IFS_DATA_DIR="${IFS_DATA_DIR:-${BASELINE_DIR}/ml_dataset_overlap_ifs_12h_pm10_pm25_q_core_no_rh2m}"
PANGU2025_DATA_DIR="${PANGU2025_DATA_DIR:-${BASELINE_DIR}/ml_dataset_overlap_pangu2025_12h_pm10_pm25_q_core_no_rh2m}"
ERA5_2025_DATA_DIR="${ERA5_2025_DATA_DIR:-${BASELINE_DIR}/ml_dataset_overlap_era5_2025_12h_pm10_pm25_q_core_no_rh2m}"
MIN_FEATURE_FINITE_FRACTION="${MIN_FEATURE_FINITE_FRACTION:-0.99}"
MAX_OUTSIDE_PLAUSIBLE_FRACTION="${MAX_OUTSIDE_PLAUSIBLE_FRACTION:-0.01}"
MAX_ROWS_PER_SPLIT="${AUDIT_MAX_ROWS_PER_SPLIT:-0}"
NOMINAL_YEAR="${NOMINAL_YEAR:-2025}"
NEXT_YEAR_SPILL_DAYS="${NEXT_YEAR_SPILL_DAYS:-1}"
EXPECTED_PANGU_LEAD_MIN_HOURS="${EXPECTED_PANGU_LEAD_MIN_HOURS:-12}"
EXPECTED_PANGU_LEAD_MAX_HOURS="${EXPECTED_PANGU_LEAD_MAX_HOURS:-23}"

mkdir -p "${BASELINE_DIR}/logs" "${OUT_DIR}"
cd "${BASELINE_DIR}"

SOURCES="tianji=${TIANJI_DATA_DIR};ifs=${IFS_DATA_DIR};pangu2025=${PANGU2025_DATA_DIR};era5_2025=${ERA5_2025_DATA_DIR}"
CMD=(
    python "${BASELINE_DIR}/audit_q_core_fair_datasets.py"
    --sources "${SOURCES}"
    --s1-dir "${S1_DATA_DIR}"
    --out-dir "${OUT_DIR}"
    --min-feature-finite-fraction "${MIN_FEATURE_FINITE_FRACTION}"
    --max-outside-plausible-fraction "${MAX_OUTSIDE_PLAUSIBLE_FRACTION}"
    --max-rows-per-split "${MAX_ROWS_PER_SPLIT}"
    --nominal-year "${NOMINAL_YEAR}"
    --next-year-spill-days "${NEXT_YEAR_SPILL_DAYS}"
    --expected-pangu-lead-min-hours "${EXPECTED_PANGU_LEAD_MIN_HOURS}"
    --expected-pangu-lead-max-hours "${EXPECTED_PANGU_LEAD_MAX_HOURS}"
)
echo "Command: ${CMD[*]}"
"${CMD[@]}"
