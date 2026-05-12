#!/bin/bash
# SLURM array job: Evaluate retrained XARELLO (BERT/GEMMA) on spellcheck_mv@3.
# 4 tasks per victim; defence pinned to the one we trained against.
#
# Submit with:
#   cd ~/xarello
#   sbatch --dependency=afterok:<BERT_TRAIN_JOBID>  --export=ALL,XARELLO_VICTIM=BERT  scripts/slurm_xarello_eval_retrained_bert_gemma.sh
#   sbatch --dependency=afterok:<GEMMA_TRAIN_JOBID> --export=ALL,XARELLO_VICTIM=GEMMA scripts/slurm_xarello_eval_retrained_bert_gemma.sh

#SBATCH -J xar_evr_bg
#SBATCH -p high
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -c 4
#SBATCH --array=0-3
#SBATCH -t 1-00:00:00
#SBATCH -o logs/xar_evr_bg_%A_%a.out
#SBATCH -e logs/xar_evr_bg_%A_%a.err

TASKS=(PR2 FC HN RD)
VICTIM="${XARELLO_VICTIM:-BERT}"

DEFENSE="spellcheck_mv"
DEFENSE_PARAM="3"

i=$SLURM_ARRAY_TASK_ID
TASK=${TASKS[$i]}

TRAINED_MODEL="models/trained_vs_${DEFENSE}/${TASK}_${VICTIM}/xarello-qmodel.pth"
DATA_PATH="$HOME/BODEGA/data/$TASK"
case "$VICTIM" in
    GEMMA) MODEL_PATH="$HOME/BODEGA/data/$TASK/GEMMA-512" ;;
    *)     MODEL_PATH="$HOME/BODEGA/data/$TASK/${VICTIM}-512.pth" ;;
esac
OUT_DIR="results/xarello_retrained_vs_${DEFENSE}"

# Robust env activation (see slurm_xarello_train_bert_gemma.sh).
if [ -f /soft/easybuild/x86_64/software/Miniconda3/22.11.1-1/etc/profile.d/conda.sh ]; then
    source /soft/easybuild/x86_64/software/Miniconda3/22.11.1-1/etc/profile.d/conda.sh
    conda activate bodega
else
    export PATH="$HOME/.conda/envs/bodega/bin:$PATH"
fi
export PYTHONPATH="$HOME/BODEGA:$PYTHONPATH"
export BODEGA_PATH="$HOME/BODEGA"
mkdir -p "$OUT_DIR" logs

echo "[$i] Retrained XARELLO | $TASK | $VICTIM | eval_defense=$DEFENSE"

if [ ! -f "$TRAINED_MODEL" ]; then
    echo "ERROR: Trained model not found at $TRAINED_MODEL"
    exit 1
fi

python -m evaluation.attack \
    "$TASK" true XARELLO "$VICTIM" \
    "$DATA_PATH" "$MODEL_PATH" "$OUT_DIR" \
    --defense "$DEFENSE" --defense_param "$DEFENSE_PARAM" --defense_seed 42 \
    --qmodel_path "$TRAINED_MODEL" \
    --semantic_scorer BLEURT
