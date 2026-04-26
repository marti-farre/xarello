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
#SBATCH --constraint=cuda
#SBATCH --mem=64G
#SBATCH -c 4
#SBATCH --array=2-3
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

# Resume from latest saved checkpoint (HN OOMed at ep 6, RD at ep 11)
case "$TASK" in
    HN) export XARELLO_STARTING_EPOCH=6 ;;
    RD) export XARELLO_STARTING_EPOCH=11 ;;
    *)  export XARELLO_STARTING_EPOCH=0 ;;
esac

source /soft/easybuild/x86_64/software/Miniconda3/22.11.1-1/etc/profile.d/conda.sh
conda activate bodega
export PYTHONPATH="$HOME/BODEGA:$PYTHONPATH"
mkdir -p "$OUT_PATH" logs

echo "[$i] Train XARELLO vs $DEFENSE | $TASK | $VICTIM"

# Skip only if a fully-trained final model marker exists.
# (We rely on XARELLO_STARTING_EPOCH for OOM-resume; do not skip on partial runs.)
if [ -f "$OUT_PATH/training_config.txt" ]; then
    echo "Final config marker found at $OUT_PATH, skipping..."
    exit 0
fi

# Training args: TASK VICTIM OUTPATH NOISE_TYPE NOISE_PARAM NOISE_SEED DEFENSE DEFENSE_PARAM DEFENSE_SEED
python main-train-eval.py \
    "$TASK" "$VICTIM" "$OUT_PATH" \
    none 0.0 42 \
    "$DEFENSE" "$DEFENSE_PARAM" 42
