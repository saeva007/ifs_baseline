#!/bin/bash
#SBATCH -J sf_p13_eval
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
TRAIN_DIR="${TRAIN_DIR:-${BASE}/train}"
EVAL_SCRIPT="${EVAL_SCRIPT:-${BASELINE_DIR}/test_PMST_overlap_forecast_source_s2.py}"
SOURCE_TAG="${SOURCE_TAG:?set SOURCE_TAG to ifs or pangu2025_source_full}"
SOURCE_SEED="${SOURCE_SEED:?set SOURCE_SEED}"
SOURCE_DATA_DIR="${SOURCE_DATA_DIR:?set SOURCE_DATA_DIR}"
SOURCE_CKPT="${SOURCE_CKPT:?set SOURCE_CKPT}"
OUT_DIR="${OUT_DIR:?set OUT_DIR}"

case "${SOURCE_TAG}" in
    ifs|pangu2025_source_full) ;;
    *) echo "ERROR: unsupported SOURCE_TAG=${SOURCE_TAG}" >&2; exit 2 ;;
esac
case "${SOURCE_SEED}" in
    ''|*[!0-9]*) echo "ERROR: SOURCE_SEED must be an integer, got ${SOURCE_SEED}" >&2; exit 2 ;;
esac

test -s "${EVAL_SCRIPT}"
test -s "${SOURCE_CKPT}"
test -d "${SOURCE_DATA_DIR}"
if [ -e "${OUT_DIR}/per_sample_${SOURCE_TAG}.csv" ] || [ -e "${OUT_DIR}/overall_metrics.csv" ]; then
    echo "ERROR: refusing to overwrite completed seed evaluation in ${OUT_DIR}" >&2
    exit 3
fi
mkdir -p "${BASELINE_DIR}/logs" "${OUT_DIR}"
cd "${BASELINE_DIR}"

# This activation owns the loader order (Torch -> OpenSSL 1.1 -> HIPNN).
# Import after activation, never through an inherited login-node Python.
source "${BASELINE_DIR}/activate_eval_torch_runtime.sh"
PYTHON_BIN="$(command -v python)"
"${PYTHON_BIN}" -c "import joblib, numpy, pandas, ssl, sys, torch; print('[p13-source-eval] python=' + sys.executable); print('[p13-source-eval] torch=' + torch.__version__); print('[p13-source-eval] ssl=' + ssl.OPENSSL_VERSION)"

COMMON_ARGS=(
    "${PYTHON_BIN}" -u "${EVAL_SCRIPT}"
    --model_arch static_rnn
    --static_rnn_train_dir "${TRAIN_DIR}"
    --independent_sources
    --source_subset "${SOURCE_TAG}"
    --threshold_mode argmax
    --skip_validation_inference
    --no_temp_scaling
    --skip_bootstrap
    --skip_operational_ensemble_analysis
    --no_figures
    --device cpu
    --batch_size "${BATCH_SIZE:-4096}"
    --num_workers "${NUM_WORKERS:-0}"
    --out_dir "${OUT_DIR}"
)

case "${SOURCE_TAG}" in
    ifs)
        COMMON_ARGS+=(
            --ifs_data_dir "${SOURCE_DATA_DIR}"
            --ifs_ckpt "${SOURCE_CKPT}"
        )
        ;;
    pangu2025_source_full)
        COMMON_ARGS+=(
            --extra_sources "pangu2025_source_full=${SOURCE_DATA_DIR}|${SOURCE_CKPT}|AUTO|Pangu-driven VisCast P13"
        )
        ;;
esac

echo "[preflight] source=${SOURCE_TAG} seed=${SOURCE_SEED}"
echo "[preflight] data=${SOURCE_DATA_DIR}"
echo "[preflight] checkpoint=${SOURCE_CKPT}"
echo "[preflight] out=${OUT_DIR}"
"${COMMON_ARGS[@]}"

test -s "${OUT_DIR}/per_sample_${SOURCE_TAG}.csv"
test -s "${OUT_DIR}/overall_metrics.csv"
echo "[done] source=${SOURCE_TAG} seed=${SOURCE_SEED} out=${OUT_DIR} at $(date)"
