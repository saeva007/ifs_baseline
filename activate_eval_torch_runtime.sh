#!/bin/bash
# Evaluation/inference requires the same working Torch runtime as training.
# Ignore an inherited TORCH_ENV so --export=ALL cannot select putianshu's
# broken environment (its torch extension requires unavailable libssl.so.1.1).

set -euo pipefail

RUNTIME_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_OPENSSL_LIB="${EVAL_OPENSSL_LIB:-/public/home/xichen/.conda/envs/py310_ppy/openssl/lib}"
EVAL_HIPNN_LIB="${EVAL_HIPNN_LIB:-/public/home/xichen/ncydata/panpy_test_liud/hipnn/lib/release}"
if [[ ! -e "${EVAL_OPENSSL_LIB}/libssl.so.1.1" ]]; then
    echo "ERROR: required OpenSSL 1.1 compatibility library is missing: ${EVAL_OPENSSL_LIB}/libssl.so.1.1" >&2
    return 2 2>/dev/null || exit 2
fi
export LD_LIBRARY_PATH="${EVAL_OPENSSL_LIB}:${EVAL_HIPNN_LIB}:${LD_LIBRARY_PATH:-}"
export TORCH_ENV="${EVAL_TORCH_ENV:-/public/home/jarvis226/miniconda3/envs/torch}"
source "${RUNTIME_DIR}/activate_torch_runtime.sh"

python -c "import ssl, sys, torch; print('[eval-env] python=' + sys.executable); print('[eval-env] torch=' + torch.__version__); print('[eval-env] ssl=' + ssl.OPENSSL_VERSION)"
