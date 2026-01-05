#!/bin/bash
# Experiment 2: Decision Noise During XARELLO Training
# Trains XARELLO with different noise configurations to corrupt Q-learning
#
# Hypothesis: Noisy responses during training will produce a corrupted policy
# that fails even when evaluated against a clean victim.

set -e  # Exit on error

# Configuration - adjust these paths as needed
TASK="${TASK:-RD}"
VICTIM="${VICTIM:-BiLSTM}"
DATA_PATH="${DATA_PATH:-$HOME/data/BODEGA/$TASK}"
MODEL_PATH="${MODEL_PATH:-$HOME/data/BODEGA/$TASK/${VICTIM}-512.pth}"
OUT_DIR="${OUT_DIR:-$HOME/data/xarello/exp2_results}"

# Create output directory
mkdir -p "$OUT_DIR"

echo "========================================"
echo "Experiment 2: Decision Noise During Training"
echo "========================================"
echo "Task: $TASK"
echo "Victim: $VICTIM"
echo "Output: $OUT_DIR"
echo "========================================"

# Function to train a single XARELLO variant
train_xarello() {
    local noise_type=$1
    local param=$2
    local desc=$3

    local out_subdir="${OUT_DIR}/${noise_type}_${param}"
    mkdir -p "$out_subdir"

    echo ""
    echo "----------------------------------------"
    echo "Training: $desc"
    echo "Noise type: $noise_type, Param: $param"
    echo "Output: $out_subdir"
    echo "----------------------------------------"

    # Check if training already completed
    if [ -f "$out_subdir/xarello-qmodel.pth" ]; then
        echo "Model already exists, skipping..."
        return 0
    fi

    python -u main-train-eval.py "$TASK" "$VICTIM" "$out_subdir" \
        "$noise_type" "$param" 42

    echo "Completed: $desc"
}

# Move to xarello directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Add BODEGA to Python path
BODEGA_PATH="${BODEGA_PATH:-$HOME/Desktop/Repos/master/BODEGA}"
export PYTHONPATH="$BODEGA_PATH:$PYTHONPATH"

echo ""
echo "Starting training experiments at $(date)"
echo ""

# 1. Train baseline (no noise)
train_xarello "none" "0.0" "Baseline (no noise)"

# 2. Train with label flip noise
for param in 0.05 0.1 0.15 0.2; do
    train_xarello "label_flip" "$param" "Label flip (epsilon=$param)"
done

# 3. Train with threshold noise
for param in 0.05 0.1 0.15 0.2; do
    train_xarello "threshold" "$param" "Threshold noise (sigma=$param)"
done

# 4. Train with confidence noise
for param in 0.05 0.1 0.15 0.2; do
    train_xarello "confidence" "$param" "Confidence noise (sigma=$param)"
done

echo ""
echo "========================================"
echo "All training completed at $(date)"
echo "Models saved to: $OUT_DIR"
echo "========================================"

# Summary of trained models
echo ""
echo "Trained Models Summary:"
echo "----------------------------------------"
for d in "$OUT_DIR"/*/; do
    if [ -d "$d" ]; then
        model_name=$(basename "$d")
        if [ -f "$d/xarello-qmodel.pth" ]; then
            echo "[OK] $model_name"
            if [ -f "$d/training_config.txt" ]; then
                grep -E "(final_eval_success|training_noise)" "$d/training_config.txt" || true
            fi
        else
            echo "[MISSING] $model_name - model not found"
        fi
    fi
done
