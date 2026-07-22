#!/bin/bash
#SBATCH -J qct925_upper
#SBATCH -p kshcexclu04
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH -t 12:00:00
#SBATCH -D /public/home/putianshu/vis_mlp/ifs_baseline
#SBATCH -o /public/home/putianshu/vis_mlp/ifs_baseline/logs/%j_%x.out
#SBATCH -e /public/home/putianshu/vis_mlp/ifs_baseline/logs/%j_%x.err

set -euo pipefail

BASE="${BASE:-/public/home/putianshu/vis_mlp}"
BASELINE_DIR="${BASELINE_DIR:-${BASE}/ifs_baseline}"
SOURCE_RUN_TAG="${SOURCE_RUN_TAG:-qcore_t925_fair_formal_v1_20260721}"
RUN_TAG="${RUN_TAG:-qcore_t925_upper_air_disagreement_v1_20260722}"
EVAL_ROOT="${EVAL_ROOT:-${BASE}/paper_eval_results_pm10_pm25_journal/q_core_t925_fair/${SOURCE_RUN_TAG}}"
DATA_ROOT="${DATA_ROOT:-${BASELINE_DIR}/q_core_t925_fair_datasets/${SOURCE_RUN_TAG}}"
PANGU_DATA_DIR="${PANGU_DATA_DIR:-${DATA_ROOT}/pangu2025}"
TIANJI_DATA_DIR="${TIANJI_DATA_DIR:-${DATA_ROOT}/tianji}"
ERA5_DATA_DIR="${ERA5_DATA_DIR:-${BASELINE_DIR}/q_core_t925_diagnostic_datasets/qcore_t925_diag_auto_v2_20260715/era5_2025}"
OUT_DIR="${OUT_DIR:-${BASE}/paper_eval_results_pm10_pm25_journal/q_core_t925_upper_air_disagreement/${RUN_TAG}/analysis}"
SEEDS="${SEEDS:-42:2025:20260702}"
PANGU_TAG="${PANGU_TAG:-pangu2025_q_core_t925_no_rh2m}"
TIANJI_TAG="${TIANJI_TAG:-tianji}"
BOOTSTRAP_ITERS="${BOOTSTRAP_ITERS:-1000}"
BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-20260722}"
FORMATS="${FORMATS:-svg:pdf:png:tiff}"
DPI="${DPI:-600}"
OBS_ROOT="${OBS_ROOT:-${BASE}/auto_station}"
PAPER_EVAL_DIR="${PAPER_EVAL_DIR:-${BASE}/paper_eval}"

mkdir -p "${BASELINE_DIR}/logs"
cd "${BASELINE_DIR}"
source "${BASELINE_DIR}/activate_eval_torch_runtime.sh"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"

for path in \
  "${EVAL_ROOT}/seed_42/per_sample_${PANGU_TAG}.csv" \
  "${EVAL_ROOT}/seed_42/per_sample_${TIANJI_TAG}.csv" \
  "${PANGU_DATA_DIR}/dataset_build_config.json" \
  "${TIANJI_DATA_DIR}/dataset_build_config.json" \
  "${ERA5_DATA_DIR}/dataset_build_config.json"
do
  [[ -s "${path}" ]] || { echo "ERROR: required input is missing: ${path}" >&2; exit 2; }
done

CMD=(
  python "${BASELINE_DIR}/analyze_q_core_t925_upper_air_disagreement.py"
  --eval-root "${EVAL_ROOT}"
  --pangu-data-dir "${PANGU_DATA_DIR}"
  --tianji-data-dir "${TIANJI_DATA_DIR}"
  --era5-data-dir "${ERA5_DATA_DIR}"
  --out-dir "${OUT_DIR}"
  --seeds "${SEEDS}"
  --pangu-tag "${PANGU_TAG}"
  --tianji-tag "${TIANJI_TAG}"
  --bootstrap-iters "${BOOTSTRAP_ITERS}"
  --bootstrap-seed "${BOOTSTRAP_SEED}"
  --min-event-coverage "${MIN_EVENT_COVERAGE:-0.98}"
  --obs-root "${OBS_ROOT}"
  --paper-eval-dir "${PAPER_EVAL_DIR}"
  --formats "${FORMATS}"
  --dpi "${DPI}"
)

echo "Upper-air disagreement diagnostic started at: $(date)"
echo "SOURCE_RUN_TAG=${SOURCE_RUN_TAG}"
echo "EVAL_ROOT=${EVAL_ROOT}"
echo "OUT_DIR=${OUT_DIR}"
echo "No training jobs are submitted or used by this diagnostic."
echo "Command: ${CMD[*]}"
"${CMD[@]}"
echo "Upper-air disagreement diagnostic completed at: $(date)"
