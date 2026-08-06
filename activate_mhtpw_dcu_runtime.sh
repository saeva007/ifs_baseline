#!/bin/bash
# Known-good Jarvis/DCU loader order for q-core MHTPW training.

set -euo pipefail

module purge
module load compiler/devtoolset/7.3.1
module load mpi/hpcx/2.11.0/gcc-7.3.1
source /public/home/xichen/ncydata/dtk/dtk-24.04.1/env.sh

DTK_HYHAL_LIB="${MHTPW_DTK_HYHAL_LIB:-/public/home/xichen/ncydata/dtk/dtk-24.04.1/.hyhal/lib}"
OPENSSL_COMPAT_LIB="${MHTPW_OPENSSL_LIB:-/public/home/xichen/.conda/envs/py310_ppy/openssl/lib}"
HIPNN_COMPAT_LIB="${MHTPW_HIPNN_LIB:-/public/home/xichen/ncydata/panpy_test_liud/hipnn/lib/release}"
export TORCH_ENV="${TORCH_ENV:-/public/home/jarvis226/miniconda3/envs/torch}"
# Slurm executes a copied batch script from its spool directory.  On this
# cluster BASH_SOURCE may be unavailable while a file is sourced, so falling
# back to $0 incorrectly resolves the runtime beside the spool copy.  The
# submitters already pass BASELINE_DIR; keep the activation path explicit and
# deterministic instead of deriving it from the batch-script location.
MHTPW_RUNTIME_DIR="${MHTPW_RUNTIME_DIR:-${BASELINE_DIR:-/public/home/putianshu/vis_mlp/ifs_baseline}}"
[[ -f "${MHTPW_RUNTIME_DIR}/activate_torch_runtime.sh" ]] || {
    echo "ERROR: base Torch activation script missing: ${MHTPW_RUNTIME_DIR}/activate_torch_runtime.sh" >&2
    return 2 2>/dev/null || exit 2
}

for path in "${DTK_HYHAL_LIB}" "${OPENSSL_COMPAT_LIB}" "${HIPNN_COMPAT_LIB}"; do
    [ -d "${path}" ] || { echo "ERROR: MHTPW runtime directory missing: ${path}" >&2; return 2 2>/dev/null || exit 2; }
done
[ -e "${OPENSSL_COMPAT_LIB}/libssl.so.1.1" ] || {
    echo "ERROR: OpenSSL compatibility library missing: ${OPENSSL_COMPAT_LIB}/libssl.so.1.1" >&2
    return 2 2>/dev/null || exit 2
}

# Build the compatibility portion first. activate_torch_runtime.sh prepends
# TORCH_ENV/lib last, yielding:
#   Torch -> OpenSSL 1.1 -> compatible HIPNN -> DTK/hyhal -> inherited.
export LD_LIBRARY_PATH="${OPENSSL_COMPAT_LIB}:${HIPNN_COMPAT_LIB}:${DTK_HYHAL_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
source "${MHTPW_RUNTIME_DIR}/activate_torch_runtime.sh"

export PYTHONUNBUFFERED=1
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export MHTPW_RANK_STAGGER_SECONDS="${MHTPW_RANK_STAGGER_SECONDS:-5}"

python -c "import ssl, sys, torch; print('[mhtpw-env] python=' + sys.executable); print('[mhtpw-env] torch=' + torch.__version__); print('[mhtpw-env] openssl=' + ssl.OPENSSL_VERSION); print('[mhtpw-env] loader_order=Torch>OpenSSL>HIPNN>DTK')"
