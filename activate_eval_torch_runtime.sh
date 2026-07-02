#!/bin/bash
# Evaluation/inference requires the same working Torch runtime as training.
# Ignore an inherited TORCH_ENV so --export=ALL cannot select putianshu's
# broken environment (its torch extension requires unavailable libssl.so.1.1).

set -euo pipefail

RUNTIME_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TORCH_ENV="${EVAL_TORCH_ENV:-/public/home/jarvis226/miniconda3/envs/torch}"
source "${RUNTIME_DIR}/activate_torch_runtime.sh"

python -c "import sys, torch; print('[eval-env] python=' + sys.executable); print('[eval-env] torch=' + torch.__version__)"
