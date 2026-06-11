#!/usr/bin/env bash
set -euo pipefail

SCENARIO="${1:-figure1_all_sources}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
mkdir -p logs

SBATCH_SCRIPT="${SBATCH_SCRIPT:-${SCRIPT_DIR}/sub_static_rnn_source_full_argmax_eval.slurm}"

submit_scenario() {
    local scenario="$1"
    echo "Submitting source-full argmax eval: ${scenario}"
    sbatch --export=ALL,EVAL_SCENARIO="${scenario}" "${SBATCH_SCRIPT}"
}

case "${SCENARIO}" in
    figure1|figure1_all_sources|all_sources)
        submit_scenario "figure1_all_sources"
        ;;
    figure2_pangu|figure2_pangu_ifs_empirical|pangu_ifs_empirical|pangu2025)
        submit_scenario "figure2_pangu_ifs_empirical"
        ;;
    figure2_merge|merge_figure2|figure2_pangu_ifs_ensemble)
        submit_scenario "figure2_merge"
        ;;
    both)
        submit_scenario "figure1_all_sources"
        submit_scenario "figure2_pangu_ifs_empirical"
        ;;
    *)
        echo "ERROR: unknown scenario: ${SCENARIO}" >&2
        echo "Usage: bash submit_static_rnn_source_full_argmax_eval.sh [figure1_all_sources|figure2_pangu_ifs_empirical|figure2_merge|both]" >&2
        exit 2
        ;;
esac
