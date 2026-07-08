#!/bin/bash
#SBATCH -J qcore_hybrid_data
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

BASE="${BASE:-/public/home/putianshu/vis_mlp}"
BASELINE_DIR="${BASELINE_DIR:-${BASE}/ifs_baseline}"
RUN_TAG="${RUN_TAG:?RUN_TAG is required}"
MODE="${MODE:-build}"
PANGU_DATA_DIR="${PANGU_DATA_DIR:-${BASELINE_DIR}/q_core_fair_datasets/${RUN_TAG}/pangu2025}"
TIANJI_DATA_DIR="${TIANJI_DATA_DIR:-${BASELINE_DIR}/q_core_fair_datasets/${RUN_TAG}/tianji}"
HYBRID_DATA_ROOT="${HYBRID_DATA_ROOT:-${BASELINE_DIR}/q_core_hybrid_datasets/${RUN_TAG}}"
GROUP_PROFILE="${GROUP_PROFILE:-mtw}"
HYBRID_DATASET_PREFIX="${HYBRID_DATASET_PREFIX:-}"
MASKS="${MASKS:-all}"
SPLITS="${SPLITS:-train,val,test}"
CHUNK_ROWS="${CHUNK_ROWS:-20000}"
LIMIT_ROWS="${LIMIT_ROWS:-0}"
ENDPOINT_FOG_ATOL="${ENDPOINT_FOG_ATOL:-5e-5}"

mkdir -p "${BASELINE_DIR}/logs" "$(dirname "${HYBRID_DATA_ROOT}")"
source "${BASELINE_DIR}/activate_torch_runtime.sh"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"

CMD=(
    python "${BASELINE_DIR}/build_q_core_hybrid_factorial.py"
    --pangu-dir "${PANGU_DATA_DIR}"
    --tianji-dir "${TIANJI_DATA_DIR}"
    --out-root "${HYBRID_DATA_ROOT}"
    --group-profile "${GROUP_PROFILE}"
    --masks "${MASKS}"
    --splits "${SPLITS}"
    --chunk-rows "${CHUNK_ROWS}"
    --limit-rows "${LIMIT_ROWS}"
    --endpoint-fog-atol "${ENDPOINT_FOG_ATOL}"
)
if [[ -n "${HYBRID_DATASET_PREFIX}" ]]; then
    CMD+=(--dataset-prefix "${HYBRID_DATASET_PREFIX}")
fi
if [[ "${MODE}" == "audit" ]]; then
    CMD+=(--audit-only)
elif [[ "${MODE}" != "build" ]]; then
    echo "ERROR: MODE must be build or audit, got ${MODE}" >&2
    exit 2
fi

echo "Command: ${CMD[*]}"
"${CMD[@]}"
