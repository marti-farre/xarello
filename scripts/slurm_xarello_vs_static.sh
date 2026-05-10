#!/bin/bash
# SLURM array job: Evaluate pretrained XARELLO without any defense
# (true "no defence" baseline). 4 tasks per victim.
#
# Produces raw_<TASK>_True_XARELLO_<VICTIM>.tsv files that the paper's
# qualitative example extractor consumes as the middle column ("no defence").
#
# Prerequisites:
#   - Pretrained XARELLO weights at ~/data/xarello/models/wide/<TASK>-<VICTIM>-2/
#
# Submit with:
#   cd ~/xarello && sbatch                                  scripts/slurm_xarello_vs_static.sh   # BiLSTM
#   sbatch --export=ALL,XARELLO_VICTIM=BERT                 scripts/slurm_xarello_vs_static.sh
#   sbatch --export=ALL,XARELLO_VICTIM=GEMMA --array=0,1,3  scripts/slurm_xarello_vs_static.sh

#SBATCH -J xar_stat
#SBATCH -p high
#SBATCH --gres=gpu:1
#SBATCH --constraint=cuda
#SBATCH --mem=48G
#SBATCH -c 4
#SBATCH --array=0-3
#SBATCH -o logs/xar_stat_%A_%a.out
#SBATCH -e logs/xar_stat_%A_%a.err

TASKS=(PR2 FC HN RD)
VICTIM="${XARELLO_VICTIM:-BiLSTM}"

i=$SLURM_ARRAY_TASK_ID
TASK=${TASKS[$i]}

DATA_PATH="$HOME/BODEGA/data/$TASK"
case "$VICTIM" in
    GEMMA) MODEL_PATH="$HOME/BODEGA/data/$TASK/GEMMA-512" ;;
    *)     MODEL_PATH="$HOME/BODEGA/data/$TASK/${VICTIM}-512.pth" ;;
esac
OUT_DIR="results/xarello_vs_static"

source /soft/easybuild/x86_64/software/Miniconda3/22.11.1-1/etc/profile.d/conda.sh
conda activate bodega
export PYTHONPATH="$HOME/BODEGA:$PYTHONPATH"
export BODEGA_PATH="$HOME/BODEGA"
mkdir -p "$OUT_DIR" logs

echo "[$i] XARELLO vs no defence | $TASK | $VICTIM"

python -m evaluation.attack \
    "$TASK" true XARELLO "$VICTIM" \
    "$DATA_PATH" "$MODEL_PATH" "$OUT_DIR" \
    --defense none \
    --semantic_scorer BLEURT
