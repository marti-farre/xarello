#!/bin/bash
# SLURM array job: Retrain XARELLO against spellcheck_mv@3 for BERT and GEMMA
# victims. Mirrors slurm_xarello_train.sh but parameterised by VICTIM via env.
#
# Submit with:
#   cd ~/xarello
#   sbatch --export=ALL,XARELLO_VICTIM=BERT  scripts/slurm_xarello_train_bert_gemma.sh
#   sbatch --export=ALL,XARELLO_VICTIM=GEMMA scripts/slurm_xarello_train_bert_gemma.sh
#
# OOM-resume a single failed cell from the latest saved checkpoint
# (xarello-qmodel.pth-N in the per-cell OUT_PATH):
#   sbatch --array=2 --mem=128G \
#          --export=ALL,XARELLO_VICTIM=BERT,XARELLO_STARTING_EPOCH=14 \
#          scripts/slurm_xarello_train_bert_gemma.sh

#SBATCH -J xar_trn_bg
#SBATCH -p high
#SBATCH --gres=gpu:1
#SBATCH --constraint=cuda
#SBATCH --mem=64G
#SBATCH -c 4
#SBATCH --array=0-3
#SBATCH -t 7-00:00:00
#SBATCH -o logs/xar_trn_bg_%A_%a.out
#SBATCH -e logs/xar_trn_bg_%A_%a.err

TASKS=(PR2 FC HN RD)
VICTIM="${XARELLO_VICTIM:-BERT}"

DEFENSE="spellcheck_mv"
DEFENSE_PARAM="3"

i=$SLURM_ARRAY_TASK_ID
TASK=${TASKS[$i]}

OUT_DIR="models/trained_vs_${DEFENSE}"
OUT_PATH="$OUT_DIR/${TASK}_${VICTIM}"

export XARELLO_STARTING_EPOCH="${XARELLO_STARTING_EPOCH:-0}"

# Robust env activation: try the easybuild conda first (present on most
# compute nodes); fall back to direct PATH manipulation on nodes that lack it.
if [ -f /soft/easybuild/x86_64/software/Miniconda3/22.11.1-1/etc/profile.d/conda.sh ]; then
    source /soft/easybuild/x86_64/software/Miniconda3/22.11.1-1/etc/profile.d/conda.sh
    conda activate bodega
else
    export PATH="$HOME/.conda/envs/bodega/bin:$PATH"
fi
export PYTHONPATH="$HOME/BODEGA:$PYTHONPATH"
mkdir -p "$OUT_PATH" logs

echo "[$i] Train XARELLO vs $DEFENSE | $TASK | $VICTIM"

if [ -f "$OUT_PATH/training_config.txt" ]; then
    echo "Final config marker found at $OUT_PATH, skipping..."
    exit 0
fi

python main-train-eval.py \
    "$TASK" "$VICTIM" "$OUT_PATH" \
    none 0.0 42 \
    "$DEFENSE" "$DEFENSE_PARAM" 42
