#!/bin/bash
# Experiment 2: Evaluate XARELLO models trained with decision noise
# Evaluates each trained model against a CLEAN victim (no noise at eval time)
#
# This tests: Did the noisy training corrupt XARELLO's learned policy?
# If successful, models trained with noise should have lower attack success rates.

set -e  # Exit on error

# Configuration - adjust these paths as needed
TASK="${TASK:-RD}"
TARGETED="${TARGETED:-true}"
VICTIM="${VICTIM:-BiLSTM}"
DATA_PATH="${DATA_PATH:-$HOME/data/BODEGA/$TASK}"
MODEL_PATH="${MODEL_PATH:-$HOME/data/BODEGA/$TASK/${VICTIM}-512.pth}"
OUT_DIR="${OUT_DIR:-$HOME/data/xarello/exp2_results}"

echo "========================================"
echo "Experiment 2: Evaluate Training Noise Defense"
echo "========================================"
echo "Task: $TASK"
echo "Victim: $VICTIM"
echo "Targeted: $TARGETED"
echo "Models: $OUT_DIR"
echo "========================================"

# Function to evaluate a single XARELLO model
eval_xarello() {
    local model_dir=$1
    local model_name=$2

    local qmodel_path="${model_dir}/xarello-qmodel.pth"
    local eval_dir="${model_dir}/eval_clean"
    mkdir -p "$eval_dir"

    echo ""
    echo "----------------------------------------"
    echo "Evaluating: $model_name"
    echo "Q-model: $qmodel_path"
    echo "Output: $eval_dir"
    echo "----------------------------------------"

    # Check if model exists
    if [ ! -f "$qmodel_path" ]; then
        echo "ERROR: Model not found at $qmodel_path"
        return 1
    fi

    # Check if evaluation already completed
    local results_file="${eval_dir}/results_${TASK}_${TARGETED}_XARELLO_${VICTIM}.txt"
    if [ -f "$results_file" ]; then
        echo "Evaluation already exists, skipping..."
        return 0
    fi

    python -u -m evaluation.attack \
        "$TASK" "$TARGETED" "XARELLO" "$VICTIM" \
        "$DATA_PATH" "$MODEL_PATH" "$eval_dir" \
        --qmodel_path "$qmodel_path"

    echo "Completed: $model_name"
}

# Move to xarello directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Add BODEGA to Python path
BODEGA_PATH="${BODEGA_PATH:-$HOME/Desktop/Repos/master/BODEGA}"
export PYTHONPATH="$BODEGA_PATH:$PYTHONPATH"

echo ""
echo "Starting evaluation experiments at $(date)"
echo ""

# Evaluate all trained models
for model_dir in "$OUT_DIR"/*/; do
    if [ -d "$model_dir" ]; then
        model_name=$(basename "$model_dir")
        eval_xarello "$model_dir" "$model_name"
    fi
done

echo ""
echo "========================================"
echo "All evaluations completed at $(date)"
echo "Results saved to: $OUT_DIR/*/eval_clean/"
echo "========================================"

# Summary of results
echo ""
echo "Results Summary:"
echo "========================================"
printf "%-25s | %-12s | %-12s | %-12s\n" "Model" "Success" "BODEGA" "Queries"
echo "----------------------------------------"
for model_dir in "$OUT_DIR"/*/; do
    if [ -d "$model_dir" ]; then
        model_name=$(basename "$model_dir")
        results_file="${model_dir}/eval_clean/results_${TASK}_${TARGETED}_XARELLO_${VICTIM}.txt"
        if [ -f "$results_file" ]; then
            success=$(grep "Success score:" "$results_file" | awk '{print $3}' || echo "N/A")
            bodega=$(grep "BODEGA score:" "$results_file" | awk '{print $3}' || echo "N/A")
            queries=$(grep "Queries per example:" "$results_file" | awk '{print $4}' || echo "N/A")
            printf "%-25s | %-12s | %-12s | %-12s\n" "$model_name" "$success" "$bodega" "$queries"
        else
            printf "%-25s | %-12s | %-12s | %-12s\n" "$model_name" "MISSING" "-" "-"
        fi
    fi
done

echo ""
echo "Interpretation:"
echo "- Baseline (none_0.0) shows attack success without defense"
echo "- Lower success rates for noisy training = defense is working"
echo "- The hypothesis is validated if noisy training produces weaker attacks"
