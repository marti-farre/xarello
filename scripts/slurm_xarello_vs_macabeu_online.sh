#!/bin/bash
# SLURM array job: Evaluate pretrained XARELLO against ONLINE MACABEU defense.
# MACABEU's policy is updated after each attacked example (online RL) so it
# can adapt to XARELLO's adaptive strategy during evaluation.
# 4 tasks x BiLSTM = 4 jobs
#
# Prerequisites:
#   - Pretrained XARELLO weights at ~/data/xarello/models/wide/<TASK>-BiLSTM-2/
#   - (Optional warm-start) MACABEU offline policies at ~/macabeu/models/<TASK>_BiLSTM.pth
#
# Submit with: cd ~/xarello && sbatch scripts/slurm_xarello_vs_macabeu_online.sh

#SBATCH -J xar_mac_on
#SBATCH -p high
#SBATCH --gres=gpu:1
#SBATCH --constraint=cuda
#SBATCH --mem=48G
#SBATCH -c 4
#SBATCH --array=0-3
#SBATCH -o logs/xar_mac_on_%A_%a.out
#SBATCH -e logs/xar_mac_on_%A_%a.err

TASKS=(PR2 FC HN RD)
VICTIM="${XARELLO_VICTIM:-BiLSTM}"

i=$SLURM_ARRAY_TASK_ID
TASK=${TASKS[$i]}

DATA_PATH="$HOME/BODEGA/data/$TASK"
case "$VICTIM" in
    GEMMA) MODEL_PATH="$HOME/BODEGA/data/$TASK/GEMMA-512" ;;
    *)     MODEL_PATH="$HOME/BODEGA/data/$TASK/${VICTIM}-512.pth" ;;
esac
MACABEU_POLICY="$HOME/macabeu/models/${TASK}_${VICTIM}.pth"
OUT_DIR="results/xarello_vs_macabeu_online/${VICTIM}"

source /soft/easybuild/x86_64/software/Miniconda3/22.11.1-1/etc/profile.d/conda.sh
conda activate bodega
export PYTHONPATH="$HOME/BODEGA:$PYTHONPATH"
export BODEGA_PATH="$HOME/BODEGA"
mkdir -p "$OUT_DIR" logs

echo "[$i] XARELLO vs ONLINE MACABEU | $TASK | $VICTIM | warm_start=$MACABEU_POLICY"

python -m evaluation.attack \
    "$TASK" true XARELLO "$VICTIM" \
    "$DATA_PATH" "$MODEL_PATH" "$OUT_DIR" \
    --defense macabeu_online --defense_policy "$MACABEU_POLICY" \
    --semantic_scorer BLEURT
