#!/bin/bash
#SBATCH -J source_full_audit
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
source /public/home/putianshu/vis_mlp/ifs_baseline/activate_torch_runtime.sh

BASE="${BASE:-/public/home/putianshu/vis_mlp}"
BASELINE_DIR="${BASELINE_DIR:-${BASE}/ifs_baseline}"
: "${SOURCES:?SOURCES is required}"
: "${S1_PROFILES:?S1_PROFILES is required}"
: "${AUDIT_OUT_DIR:?AUDIT_OUT_DIR is required}"
AUDIT_MAX_ROWS_PER_SPLIT="${AUDIT_MAX_ROWS_PER_SPLIT:-0}"

mkdir -p "${BASELINE_DIR}/logs" "${AUDIT_OUT_DIR}"
cd "${BASELINE_DIR}"
python "${BASELINE_DIR}/audit_source_full_canonical_datasets.py" \
    --sources "${SOURCES}" \
    --s1-profiles "${S1_PROFILES}" \
    --out-dir "${AUDIT_OUT_DIR}" \
    --max-rows-per-split "${AUDIT_MAX_ROWS_PER_SPLIT}"
