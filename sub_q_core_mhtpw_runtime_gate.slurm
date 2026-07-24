#!/bin/bash
#SBATCH -J mhtpw_runtime_gate
#SBATCH -p kshdexclu01
#SBATCH -N 5
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=dcu:4
#SBATCH --mem=0
#SBATCH -t 01:00:00
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err
#SBATCH --exclusive

set -euo pipefail

BASELINE_DIR="${BASELINE_DIR:-/public/home/putianshu/vis_mlp/ifs_baseline}"
cd "${BASELINE_DIR}"
mkdir -p logs
source "${BASELINE_DIR}/activate_mhtpw_dcu_runtime.sh"

TORCHRUN_BIN="$(command -v torchrun)"
PROBE_SCRIPT="${BASELINE_DIR}/probe_mhtpw_dcu_runtime.py"
[ -x "${TORCHRUN_BIN}" ] || { echo "ERROR: torchrun unavailable" >&2; exit 2; }
[ -s "${PROBE_SCRIPT}" ] || { echo "ERROR: probe missing: ${PROBE_SCRIPT}" >&2; exit 2; }

echo "MHTPW shared runtime gate started at: $(date)"
echo "nodes=${SLURM_JOB_NODELIST}"
srun --export=ALL \
     --nodes=${SLURM_NNODES} \
     --ntasks=${SLURM_NNODES} \
     --ntasks-per-node=1 \
     --cpus-per-task=4 \
     --gres=dcu:4 \
     --kill-on-bad-exit=1 \
     /bin/bash -c "
         cd '${BASELINE_DIR}' || exit 1
         '${TORCHRUN_BIN}' --standalone --nnodes=1 --nproc_per_node=4 \
             '${PROBE_SCRIPT}'
     "
echo "MHTPW_RUNTIME_GATE_PASSED=1"
echo "MHTPW shared runtime gate finished at: $(date)"
