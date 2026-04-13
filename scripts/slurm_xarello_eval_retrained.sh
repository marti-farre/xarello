#!/bin/bash
# SLURM array job: Evaluate retrained XARELLO (trained against defense)
# 4 tasks x 2 configs (vs defense, vs no-defense) = 8 jobs
#
# Tests whether adaptive XARELLO can bypass the defense it was trained against
# AND whether it still works on undefended victims.
#
# Submit with: cd ~/xarello && sbatch scripts/slurm_xarello_eval_retrained.sh

#SBATCH -J xar_evr
#SBATCH -p medium
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -c 4
#SBATCH --array=0-7
#SBATCH -o logs/xar_evr_%A_%a.out
#SBATCH -e logs/xar_evr_%A_%a.err

TASKS=(PR2 FC HN RD)
VICTIM="BiLSTM"

# ====================================================================
# Must match slurm_xarello_train.sh settings
# ====================================================================
DEFENSE="spellcheck"       # placeholder — update after HPC query
DEFENSE_PARAM="0"          # placeholder — update after HPC query

EVAL_DEFENSES=(
    none:0
    "${DEFENSE}:${DEFENSE_PARAM}"
)

i=$SLURM_ARRAY_TASK_ID
TASK=${TASKS[$((i / 2))]}
DEF_ENTRY=${EVAL_DEFENSES[$((i % 2))]}
EVAL_DEFENSE=${DEF_ENTRY%%:*}
EVAL_PARAM=${DEF_ENTRY##*:}

TRAINED_MODEL="models/trained_vs_${DEFENSE}/${TASK}_${VICTIM}/xarello-qmodel.pth"
DATA_PATH="$HOME/BODEGA/data/$TASK"
MODEL_PATH="$HOME/BODEGA/data/$TASK/${VICTIM}-512.pth"
OUT_DIR="results/xarello_retrained_vs_${DEFENSE}"

module load Miniconda3
eval "$(conda shell.bash hook)"
conda activate bodega
export PYTHONPATH="$HOME/BODEGA:$PYTHONPATH"
export BODEGA_PATH="$HOME/BODEGA"
mkdir -p "$OUT_DIR" logs

echo "[$i] Retrained XARELLO | $TASK | $VICTIM | eval_defense=$EVAL_DEFENSE | trained_against=$DEFENSE"

if [ ! -f "$TRAINED_MODEL" ]; then
    echo "ERROR: Trained model not found at $TRAINED_MODEL"
    exit 1
fi

if [ "$EVAL_DEFENSE" = "none" ]; then
    python -m evaluation.attack \
        "$TASK" true XARELLO "$VICTIM" \
        "$DATA_PATH" "$MODEL_PATH" "$OUT_DIR" \
        --defense none --qmodel_path "$TRAINED_MODEL" \
        --semantic_scorer BLEURT
else
    python -m evaluation.attack \
        "$TASK" true XARELLO "$VICTIM" \
        "$DATA_PATH" "$MODEL_PATH" "$OUT_DIR" \
        --defense "$EVAL_DEFENSE" --defense_param "$EVAL_PARAM" --defense_seed 42 \
        --qmodel_path "$TRAINED_MODEL" \
        --semantic_scorer BLEURT
fi
