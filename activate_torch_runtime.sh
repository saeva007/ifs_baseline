#!/bin/bash
# Activate the cluster torch environment without invoking the broken Conda
# plugin loader (conda-libmamba-solver currently misses libarchive.so.20).

set -euo pipefail

TORCH_ENV="${TORCH_ENV:-/public/home/putianshu/miniconda3/envs/torch}"
if [[ ! -x "${TORCH_ENV}/bin/python" ]]; then
    echo "ERROR: torch Python is missing: ${TORCH_ENV}/bin/python" >&2
    return 2 2>/dev/null || exit 2
fi

export PATH="${TORCH_ENV}/bin:${PATH}"
export LD_LIBRARY_PATH="${TORCH_ENV}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export CONDA_PREFIX="${TORCH_ENV}"
export CONDA_DEFAULT_ENV="torch"
export CONDA_SHLVL="1"
export CONDA_NO_PLUGINS="true"
export CONDA_SOLVER="classic"

actual_python="$(command -v python)"
if [[ "${actual_python}" != "${TORCH_ENV}/bin/python" ]]; then
    echo "ERROR: expected ${TORCH_ENV}/bin/python, got ${actual_python}" >&2
    return 2 2>/dev/null || exit 2
fi
echo "[env] python=${actual_python} (direct torch runtime; Conda CLI bypassed)"
