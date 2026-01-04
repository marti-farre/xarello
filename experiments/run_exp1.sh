#!/bin/bash
# Experiment 1: Input Preprocessing Defenses
# Runs baseline + all defense variants and collects results

set -e  # Exit on error

# Configuration - adjust these paths as needed
TASK="${TASK:-PR2}"
TARGETED="${TARGETED:-true}"
ATTACK="${ATTACK:-XARELLO}"
VICTIM="${VICTIM:-BERT}"
DATA_PATH="${DATA_PATH:-$HOME/data/BODEGA/$TASK}"
MODEL_PATH="${MODEL_PATH:-$HOME/data/BODEGA/$TASK/${VICTIM}-512.pth}"
OUT_DIR="${OUT_DIR:-$HOME/data/xarello/exp1_results}"

# Create output directory
mkdir -p "$OUT_DIR"

echo "========================================"
echo "Experiment 1: Input Preprocessing Defenses"
echo "========================================"
echo "Task: $TASK"
echo "Targeted: $TARGETED"
echo "Attack: $ATTACK"
echo "Victim: $VICTIM"
echo "Output: $OUT_DIR"
echo "========================================"

# Function to run a single experiment
run_experiment() {
    local defense=$1
    local param=$2
    local desc=$3

    echo ""
    echo "----------------------------------------"
    echo "Running: $desc"
    echo "Defense: $defense, Param: $param"
    echo "----------------------------------------"

    python -u -m evaluation.attack \
        "$TASK" "$TARGETED" "$ATTACK" "$VICTIM" \
        "$DATA_PATH" "$MODEL_PATH" "$OUT_DIR" \
        --defense "$defense" \
        --defense_param "$param" \
        --defense_seed 42

    echo "Completed: $desc"
}

# Move to xarello directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Add BODEGA to Python path
BODEGA_PATH="${BODEGA_PATH:-$HOME/Desktop/Repos/master/BODEGA}"
export PYTHONPATH="$BODEGA_PATH:$PYTHONPATH"

echo ""
echo "Starting experiments at $(date)"
echo ""

# 1. Spell check defense
run_experiment "spellcheck" 0.0 "Spell check defense"

# 3. Token dropout variants
for dropout in 0.05 0.1 0.15 0.2; do
    run_experiment "dropout" "$dropout" "Token dropout (p=$dropout)"
done

# 4. Embedding noise variants (character-level proxy)
for noise in 0.05 0.1 0.15 0.2; do
    run_experiment "noise" "$noise" "Embedding noise (std=$noise)"
done

echo ""
echo "========================================"
echo "All experiments completed at $(date)"
echo "Results saved to: $OUT_DIR"
echo "========================================"

# Summary of results
echo ""
echo "Results Summary:"
echo "----------------------------------------"
for f in "$OUT_DIR"/results_*.txt; do
    if [ -f "$f" ]; then
        echo ""
        echo "File: $(basename "$f")"
        grep -E "(Defense:|BODEGA score:|Success score:)" "$f" || true
    fi
done
