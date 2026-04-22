#!/bin/bash
# SLURM array job: Train XARELLO against the best static defense
# 4 tasks x BiLSTM = 4 jobs (each ~1-2 days)
#
# This trains an ADAPTIVE XARELLO that learns to attack despite the defense.
#
# Submit with: cd ~/xarello && sbatch scripts/slurm_xarello_train.sh

#SBATCH -J xar_trn
#SBATCH -p high
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -c 4
#SBATCH --array=0-3
#SBATCH -o logs/xar_trn_%A_%a.out
#SBATCH -e logs/xar_trn_%A_%a.err

TASKS=(PR2 FC HN RD)
VICTIM="BiLSTM"

# ====================================================================
# Configure the best static defense to train against (from HPC results)
# These use XARELLO defense names: spellcheck, noise, dropout, majority_vote, spellcheck_mv
# ====================================================================
DEFENSE="spellcheck_mv"    # placeholder — update after HPC query
DEFENSE_PARAM="3"          # placeholder — update after HPC query

i=$SLURM_ARRAY_TASK_ID
TASK=${TASKS[$i]}

OUT_DIR="models/trained_vs_${DEFENSE}"
OUT_PATH="$OUT_DIR/${TASK}_${VICTIM}"

module load Miniconda3
eval "$(conda shell.bash hook)"
conda activate bodega
export PYTHONPATH="$HOME/BODEGA:$PYTHONPATH"
mkdir -p "$OUT_PATH" logs

echo "[$i] Train XARELLO vs $DEFENSE | $TASK | $VICTIM"

# Check if already trained
if [ -f "$OUT_PATH/xarello-qmodel.pth" ]; then
    echo "Model already exists at $OUT_PATH, skipping..."
    exit 0
fi

# Training args: TASK VICTIM OUTPATH NOISE_TYPE NOISE_PARAM NOISE_SEED DEFENSE DEFENSE_PARAM DEFENSE_SEED
python main-train-eval.py \
    "$TASK" "$VICTIM" "$OUT_PATH" \
    none 0.0 42 \
    "$DEFENSE" "$DEFENSE_PARAM" 42
