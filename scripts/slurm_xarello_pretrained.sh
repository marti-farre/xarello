#!/bin/bash
# SLURM array job: Evaluate pretrained XARELLO against static defenses
# 4 tasks x 3 defenses (none, DEFENSE_1, DEFENSE_2) = 12 jobs
#
# Prerequisites: Upload pretrained XARELLO weights to
#   ~/data/xarello/models/wide/<TASK>-BiLSTM-2/xarello-qmodel.pth
#
# Submit with: cd ~/xarello && sbatch scripts/slurm_xarello_pretrained.sh

#SBATCH -J xar_pre
#SBATCH -p high
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -c 4
#SBATCH --array=0-11
#SBATCH -o logs/xar_pre_%A_%a.out
#SBATCH -e logs/xar_pre_%A_%a.err

TASKS=(PR2 FC HN RD)
VICTIM="BiLSTM"

# ====================================================================
# Configure the two best static defenses here (from HPC results)
# Format: "defense_name:param"
# ====================================================================
BEST_DEF_1="spellcheck_mv:3"       # placeholder — update after HPC query
BEST_DEF_2="spellcheck_mv:7"       # placeholder — update after HPC query

DEFENSES=(
    none:0
    "$BEST_DEF_1"
    "$BEST_DEF_2"
)

i=$SLURM_ARRAY_TASK_ID
TASK=${TASKS[$((i / 3))]}
DEF_ENTRY=${DEFENSES[$((i % 3))]}
DEFENSE=${DEF_ENTRY%%:*}
PARAM=${DEF_ENTRY##*:}

DATA_PATH="$HOME/BODEGA/data/$TASK"
MODEL_PATH="$HOME/BODEGA/data/$TASK/${VICTIM}-512.pth"
OUT_DIR="results/xarello_vs_static"

module load Miniconda3
eval "$(conda shell.bash hook)"
conda activate bodega
export PYTHONPATH="$HOME/BODEGA:$PYTHONPATH"
export BODEGA_PATH="$HOME/BODEGA"
mkdir -p "$OUT_DIR" logs

echo "[$i] XARELLO pretrained | $TASK | $VICTIM | defense=$DEFENSE (param=$PARAM)"

if [ "$DEFENSE" = "none" ]; then
    python -m evaluation.attack \
        "$TASK" true XARELLO "$VICTIM" \
        "$DATA_PATH" "$MODEL_PATH" "$OUT_DIR" \
        --defense none --semantic_scorer BLEURT
else
    python -m evaluation.attack \
        "$TASK" true XARELLO "$VICTIM" \
        "$DATA_PATH" "$MODEL_PATH" "$OUT_DIR" \
        --defense "$DEFENSE" --defense_param "$PARAM" --defense_seed 42 \
        --semantic_scorer BLEURT
fi
