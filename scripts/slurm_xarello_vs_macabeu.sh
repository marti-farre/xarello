#!/bin/bash
# SLURM array job: Evaluate pretrained XARELLO against MACABEU defense
# 4 tasks x BiLSTM = 4 jobs
#
# Prerequisites:
#   - Pretrained XARELLO weights at ~/data/xarello/models/wide/<TASK>-BiLSTM-2/
#   - MACABEU policies at ~/macabeu/models/<TASK>_BiLSTM.pth
#
# Submit with: cd ~/xarello && sbatch scripts/slurm_xarello_vs_macabeu.sh

#SBATCH -J xar_mac
#SBATCH -p high
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -c 4
#SBATCH --array=0-3
#SBATCH -o logs/xar_mac_%A_%a.out
#SBATCH -e logs/xar_mac_%A_%a.err

TASKS=(PR2 FC HN RD)
VICTIM="BiLSTM"

i=$SLURM_ARRAY_TASK_ID
TASK=${TASKS[$i]}

DATA_PATH="$HOME/BODEGA/data/$TASK"
MODEL_PATH="$HOME/BODEGA/data/$TASK/${VICTIM}-512.pth"
MACABEU_POLICY="$HOME/macabeu/models/${TASK}_${VICTIM}.pth"
OUT_DIR="results/xarello_vs_macabeu"

module load Miniconda3
eval "$(conda shell.bash hook)"
conda activate bodega
export PYTHONPATH="$HOME/BODEGA:$PYTHONPATH"
export BODEGA_PATH="$HOME/BODEGA"
mkdir -p "$OUT_DIR" logs

echo "[$i] XARELLO vs MACABEU | $TASK | $VICTIM | policy=$MACABEU_POLICY"

python -m evaluation.attack \
    "$TASK" true XARELLO "$VICTIM" \
    "$DATA_PATH" "$MODEL_PATH" "$OUT_DIR" \
    --defense macabeu --defense_policy "$MACABEU_POLICY" \
    --semantic_scorer BLEURT
