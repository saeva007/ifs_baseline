#!/bin/bash
#SBATCH -J sf_argmax_eval
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

echo "Static-RNN source-full argmax evaluation started at: $(date)"
echo "Node: $(hostname)"
echo "Scenario: ${EVAL_SCENARIO:-figure1_all_sources}"

module purge
module load compiler/devtoolset/7.3.1
module load mpi/hpcx/2.11.0/gcc-7.3.1
source /public/home/xichen/ncydata/dtk/dtk-24.04.1/env.sh
export LD_LIBRARY_PATH=/public/home/xichen/ncydata/dtk/dtk-24.04.1/.hyhal/lib:$LD_LIBRARY_PATH
export CONDA_NO_PLUGINS=true
export CONDA_SOLVER=classic
source /public/home/putianshu/miniconda3/bin/activate
conda activate torch
export LD_LIBRARY_PATH=/public/home/xichen/.conda/envs/py310_ppy/openssl/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/public/home/xichen/ncydata/panpy_test_liud/hipnn/lib/release/:$LD_LIBRARY_PATH

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"

BASE="${BASE:-/public/home/putianshu/vis_mlp}"
BASELINE_DIR="${BASELINE_DIR:-${BASE}/ifs_baseline}"
EVAL_SCRIPT="${EVAL_SCRIPT:-${BASELINE_DIR}/test_PMST_overlap_forecast_source_s2.py}"
MERGE_SCRIPT="${MERGE_SCRIPT:-${BASELINE_DIR}/merge_overlap_source_eval_metrics.py}"
TRAIN_DIR="${TRAIN_DIR:-${BASE}/train}"
CKPT_DIR="${CKPT_DIR:-${BASELINE_DIR}/checkpoints}"
OUT_ROOT="${OUT_ROOT:-${BASE}/paper_eval_results_pm10_pm25_journal/best_effort_source_full_argmax}"

TIANJI_DATA_DIR="${TIANJI_DATA_DIR:-${BASELINE_DIR}/ml_dataset_overlap_tianji_12h_pm10_pm25_source_full}"
IFS_DATA_DIR="${IFS_DATA_DIR:-${BASELINE_DIR}/ml_dataset_overlap_ifs_12h_pm10_pm25_source_full}"
T2ND_DATA_DIR="${T2ND_DATA_DIR:-${BASELINE_DIR}/ml_dataset_overlap_tianji_12h_pm10_pm25_T2ND_rh2m_source_full}"
PANGU2025_DATA_DIR="${PANGU2025_DATA_DIR:-${BASELINE_DIR}/ml_dataset_overlap_pangu2025_12h_pm10_pm25_source_full}"
ERA5_2025_DATA_DIR="${ERA5_2025_DATA_DIR:-${BASELINE_DIR}/ml_dataset_overlap_era5_2025_12h_pm10_pm25_source_full}"

TIANJI_CKPT="${TIANJI_CKPT:-${CKPT_DIR}/exp_overlap_static_rnn_s2_tianji_source_full_pm10_pm25_S2_PhaseB_best_score.pt}"
IFS_CKPT="${IFS_CKPT:-${CKPT_DIR}/exp_overlap_static_rnn_s2_ifs_source_full_pm10_pm25_S2_PhaseB_best_score.pt}"
T2ND_CKPT="${T2ND_CKPT:-${CKPT_DIR}/exp_overlap_static_rnn_s2_T2ND_rh2m_source_full_pm10_pm25_S2_PhaseB_best_score.pt}"
PANGU2025_CKPT="${PANGU2025_CKPT:-${CKPT_DIR}/exp_overlap_static_rnn_s2_pangu2025_source_full_pm10_pm25_S2_PhaseB_best_score.pt}"
ERA5_2025_CKPT="${ERA5_2025_CKPT:-${CKPT_DIR}/exp_overlap_static_rnn_s2_era5_2025_source_full_pm10_pm25_S2_PhaseB_best_score.pt}"

DEVICE="${DEVICE:-cpu}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
NUM_WORKERS="${NUM_WORKERS:-0}"
THRESHOLD_MODE="${THRESHOLD_MODE:-argmax}"
THRESHOLD_SEARCH_POLICY="${THRESHOLD_SEARCH_POLICY:-response}"
SKIP_BOOTSTRAP="${SKIP_BOOTSTRAP:-1}"
NO_PER_SAMPLE_CSV="${NO_PER_SAMPLE_CSV:-1}"
SKIP_VALIDATION_INFERENCE="${SKIP_VALIDATION_INFERENCE:-1}"
LIMIT_SAMPLES="${LIMIT_SAMPLES:-}"
NO_FIGURES="${NO_FIGURES:-0}"
ALLOW_LEGACY_TIME_ALIGNMENT="${ALLOW_LEGACY_TIME_ALIGNMENT:-0}"
STRICT_META="${STRICT_META:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
EVAL_SCENARIO="${EVAL_SCENARIO:-figure1_all_sources}"
MERGE_ONLY=0

T2ND_SPEC="T2ND_rh2m_source_full=${T2ND_DATA_DIR}|${T2ND_CKPT}|AUTO|T2ND RH2M source-full"
PANGU2025_SPEC="pangu2025_source_full=${PANGU2025_DATA_DIR}|${PANGU2025_CKPT}|AUTO|Pangu-2025 source-full"
ERA5_2025_SPEC="era5_2025_source_full=${ERA5_2025_DATA_DIR}|${ERA5_2025_CKPT}|AUTO|ERA5-2025 source-full"

case "${EVAL_SCENARIO}" in
    figure1|figure1_all_sources|all_sources)
        SOURCE_SUBSET="${SOURCE_SUBSET:-tianji,ifs,T2ND_rh2m_source_full,pangu2025_source_full,era5_2025_source_full}"
        EXTRA_SOURCES="${EXTRA_SOURCES:-${T2ND_SPEC}; ${PANGU2025_SPEC}; ${ERA5_2025_SPEC}}"
        OUT_DIR="${OUT_DIR:-${OUT_ROOT}/figure1_all_sources}"
        SKIP_IFS_FORECAST_BASELINE="${SKIP_IFS_FORECAST_BASELINE:-1}"
        ;;
    figure2_pangu|figure2_pangu_ifs_empirical|pangu_ifs_empirical|pangu2025)
        SOURCE_SUBSET="${SOURCE_SUBSET:-pangu2025_source_full}"
        EXTRA_SOURCES="${EXTRA_SOURCES:-${PANGU2025_SPEC}}"
        OUT_DIR="${OUT_DIR:-${OUT_ROOT}/figure2_pangu_ifs_empirical_argmax}"
        SKIP_IFS_FORECAST_BASELINE="${SKIP_IFS_FORECAST_BASELINE:-0}"
        ;;
    figure2_merge|merge_figure2|figure2_pangu_ifs_ensemble)
        MERGE_ONLY=1
        INPUT_DIRS="${INPUT_DIRS:-${OUT_ROOT}/figure2_pangu_ifs_empirical_argmax,${OUT_ROOT}/figure2_tianji_ifs_mean_argmax}"
        OUT_DIR="${OUT_DIR:-${OUT_ROOT}/figure2_pangu_ifs_ensemble_argmax}"
        ;;
    *)
        echo "ERROR: unknown EVAL_SCENARIO=${EVAL_SCENARIO}" >&2
        echo "Use figure1_all_sources, figure2_pangu_ifs_empirical, or figure2_merge." >&2
        exit 2
        ;;
esac

mkdir -p "${BASELINE_DIR}/logs" "${OUT_DIR}"
cd "${BASELINE_DIR}"

python -c "import sys, torch; print('Python', sys.version.replace(chr(10), ' ')); print('Torch', torch.__version__)"

if [[ "${MERGE_ONLY}" == "1" ]]; then
    CMD=(python "${MERGE_SCRIPT}" --input_dirs "${INPUT_DIRS}" --out_dir "${OUT_DIR}")
    echo "Command: ${CMD[*]}"
    "${CMD[@]}"
    echo "Static-RNN source-full merge finished at: $(date)"
    exit 0
fi

CMD=(
    python "${EVAL_SCRIPT}"
    --model_arch static_rnn
    --static_rnn_train_dir "${TRAIN_DIR}"
    --independent_sources
    --threshold_mode "${THRESHOLD_MODE}"
    --threshold_search_policy "${THRESHOLD_SEARCH_POLICY}"
    --tianji_data_dir "${TIANJI_DATA_DIR}"
    --tianji_ckpt "${TIANJI_CKPT}"
    --ifs_data_dir "${IFS_DATA_DIR}"
    --ifs_ckpt "${IFS_CKPT}"
    --extra_sources "${EXTRA_SOURCES}"
    --source_subset "${SOURCE_SUBSET}"
    --out_dir "${OUT_DIR}"
    --device "${DEVICE}"
    --batch_size "${BATCH_SIZE}"
    --num_workers "${NUM_WORKERS}"
)

if [[ "${SKIP_BOOTSTRAP}" == "1" || "${SKIP_BOOTSTRAP}" == "true" ]]; then
    CMD+=(--skip_bootstrap)
fi
if [[ "${NO_PER_SAMPLE_CSV}" == "1" || "${NO_PER_SAMPLE_CSV}" == "true" ]]; then
    CMD+=(--no_per_sample_csv)
fi
if [[ "${SKIP_VALIDATION_INFERENCE}" == "1" && "${THRESHOLD_MODE}" != "val_search" ]]; then
    CMD+=(--skip_validation_inference)
fi
if [[ "${SKIP_IFS_FORECAST_BASELINE}" == "1" || "${SKIP_IFS_FORECAST_BASELINE}" == "true" ]]; then
    CMD+=(--skip_ifs_forecast_baseline)
fi
if [[ "${NO_FIGURES}" == "1" || "${NO_FIGURES}" == "true" ]]; then
    CMD+=(--no_figures)
fi
if [[ "${ALLOW_LEGACY_TIME_ALIGNMENT}" == "1" || "${ALLOW_LEGACY_TIME_ALIGNMENT}" == "true" ]]; then
    CMD+=(--allow_legacy_time_alignment)
fi
if [[ "${STRICT_META}" == "1" || "${STRICT_META}" == "true" ]]; then
    CMD+=(--strict_meta)
fi
if [[ -n "${LIMIT_SAMPLES}" ]]; then
    CMD+=(--limit_samples "${LIMIT_SAMPLES}")
fi
if [[ -n "${EXTRA_ARGS}" ]]; then
    # shellcheck disable=SC2206
    EXTRA_ARGV=(${EXTRA_ARGS})
    CMD+=("${EXTRA_ARGV[@]}")
fi

echo "Command: ${CMD[*]}"
"${CMD[@]}"

echo "Static-RNN source-full argmax evaluation finished at: $(date)"
