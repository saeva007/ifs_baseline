#!/bin/bash
#SBATCH -J sf_p13_agg
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
AGGREGATE_SCRIPT="${AGGREGATE_SCRIPT:-${BASELINE_DIR}/aggregate_source_full_p13.py}"
SEED_EVAL_ROOT="${SEED_EVAL_ROOT:?set SEED_EVAL_ROOT}"
OUT_ROOT="${OUT_ROOT:?set OUT_ROOT}"
P13_SEEDS="${P13_SEEDS:-42:314:2718}"
MAINLINE_P13_PER_SAMPLE="${MAINLINE_P13_PER_SAMPLE:?set MAINLINE_P13_PER_SAMPLE}"

test -s "${AGGREGATE_SCRIPT}"
test -s "${MAINLINE_P13_PER_SAMPLE}"
test -d "${SEED_EVAL_ROOT}"
if [ -e "${OUT_ROOT}/ifs/overall_metrics.csv" ] || [ -e "${OUT_ROOT}/pangu2025_source_full/overall_metrics.csv" ]; then
    echo "ERROR: refusing to overwrite completed source-full P13 aggregate in ${OUT_ROOT}" >&2
    exit 3
fi
mkdir -p "${BASELINE_DIR}/logs" "${OUT_ROOT}"
cd "${BASELINE_DIR}"

source "${BASELINE_DIR}/activate_eval_torch_runtime.sh"
PYTHON_BIN="$(command -v python)"
"${PYTHON_BIN}" -c "import numpy, pandas, ssl, sys, torch; print('[p13-source-aggregate] python=' + sys.executable); print('[p13-source-aggregate] torch=' + torch.__version__); print('[p13-source-aggregate] ssl=' + ssl.OPENSSL_VERSION)"

"${PYTHON_BIN}" -u "${AGGREGATE_SCRIPT}" \
    --source-tag ifs \
    --label "IFS-driven VisCast P13" \
    --seeds "${P13_SEEDS}" \
    --seed-root "${SEED_EVAL_ROOT}/ifs" \
    --out-dir "${OUT_ROOT}/ifs" \
    --mainline-p13-per-sample "${MAINLINE_P13_PER_SAMPLE}"

"${PYTHON_BIN}" -u "${AGGREGATE_SCRIPT}" \
    --source-tag pangu2025_source_full \
    --label "Pangu-driven VisCast P13" \
    --seeds "${P13_SEEDS}" \
    --seed-root "${SEED_EVAL_ROOT}/pangu2025_source_full" \
    --out-dir "${OUT_ROOT}/pangu2025_source_full" \
    --mainline-p13-per-sample "${MAINLINE_P13_PER_SAMPLE}"

test -s "${OUT_ROOT}/ifs/per_sample_ifs_p13.csv"
test -s "${OUT_ROOT}/pangu2025_source_full/per_sample_pangu2025_source_full_p13.csv"
echo "[done] source-full P13 aggregate=${OUT_ROOT} at $(date)"
