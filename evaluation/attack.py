import argparse
import gc
import os
import pathlib
import sys
import time
import random
import numpy as np
from collections import Counter

import OpenAttack
import torch
from datasets import Dataset

from metrics.BODEGAScore import BODEGAScore
from utils.data_mappings import dataset_mapping, dataset_mapping_pairs, SEPARATOR_CHAR
from utils.no_ssl_verify import no_ssl_verify
from victims.transformer import VictimTransformer, PRETRAINED_BERT, PRETRAINED_GEMMA_2B, PRETRAINED_GEMMA_7B, \
    readfromfile_generator
from victims.bilstm import VictimBiLSTM
from victims.caching import VictimCache

from env.EnvAE import EnvAE
from evaluation.qattacker import QAttacker
from defenses.preprocessing import get_defense

# Attempt at determinism
random.seed(10)
torch.manual_seed(10)
np.random.seed(0)

# Running variables
print("Preparing the environment...")

# Default values
task = 'PR2'
targeted = True
victim_model_type = 'BERT'
attack_model_type = 'XARELLO'
attack_model_variant = 'wide'
out_dir = None
defense_type = 'none'
defense_param = 0.0
data_path = pathlib.Path.home() / 'data' / 'BODEGA' / task
victim_model_path = pathlib.Path.home() / 'data' / 'BODEGA' / task / (victim_model_type + '-512.pth')

# Parse arguments - support both legacy positional args and new named args
parser = argparse.ArgumentParser(description='XARELLO attack evaluation with optional defenses')
parser.add_argument('--task', type=str, default=None, help='Task name (PR2, HN, FC, C19, RD)')
parser.add_argument('--targeted', type=str, default=None, help='Targeted attack (true/false)')
parser.add_argument('--attack', type=str, default=None, help='Attack model type (XARELLO, random)')
parser.add_argument('--victim', type=str, default=None, help='Victim model type (BERT, BiLSTM, GEMMA)')
parser.add_argument('--data_path', type=str, default=None, help='Path to data directory')
parser.add_argument('--model_path', type=str, default=None, help='Path to victim model')
parser.add_argument('--out_dir', type=str, default=None, help='Output directory')
parser.add_argument('--defense', type=str, default='none',
                    help='Defense type: none, spellcheck, noise, dropout')
parser.add_argument('--defense_param', type=float, default=0.0,
                    help='Defense parameter (noise_std or dropout_prob)')
parser.add_argument('--defense_seed', type=int, default=42,
                    help='Random seed for defense (for reproducibility)')
parser.add_argument('--verbose', action='store_true',
                    help='Print defense modifications as they happen')
parser.add_argument('--qmodel_path', type=str, default=None,
                    help='Path to XARELLO Q-model (overrides default path)')
parser.add_argument('--semantic_scorer', type=str, default='BERTscore',
                    help='Semantic scorer: BERTscore or BLEURT')
parser.add_argument('--defense_policy', type=str, default=None,
                    help='Path to MACABEU defense policy (for --defense macabeu)')

# Check if using legacy positional args or new named args
if len(sys.argv) >= 7 and not sys.argv[1].startswith('--'):
    # Legacy positional argument parsing
    task = sys.argv[1]
    targeted = (sys.argv[2].lower() == 'true')
    attack_model_type = sys.argv[3]
    victim_model_type = sys.argv[4]
    data_path = pathlib.Path(sys.argv[5])
    victim_model_path = pathlib.Path(sys.argv[6])
    if len(sys.argv) >= 8 and not sys.argv[7].startswith('--'):
        out_dir = pathlib.Path(sys.argv[7])
    # Check for defense args after positional args
    remaining_args = sys.argv[8:] if len(sys.argv) >= 8 and not sys.argv[7].startswith('--') else sys.argv[7:]
    if remaining_args:
        args, _ = parser.parse_known_args(remaining_args)
        defense_type = args.defense
        defense_param = args.defense_param
        defense_seed = args.defense_seed
        verbose = args.verbose
        qmodel_path = args.qmodel_path
        semantic_scorer = args.semantic_scorer
        defense_policy = args.defense_policy
    else:
        defense_seed = 42
        verbose = False
        qmodel_path = None
        semantic_scorer = 'BERTscore'
        defense_policy = None
else:
    # Named argument parsing
    args = parser.parse_args()
    if args.task:
        task = args.task
    if args.targeted:
        targeted = (args.targeted.lower() == 'true')
    if args.attack:
        attack_model_type = args.attack
    if args.victim:
        victim_model_type = args.victim
    if args.data_path:
        data_path = pathlib.Path(args.data_path)
    if args.model_path:
        victim_model_path = pathlib.Path(args.model_path)
    if args.out_dir:
        out_dir = pathlib.Path(args.out_dir)
    defense_type = args.defense
    defense_param = args.defense_param
    defense_seed = args.defense_seed
    verbose = args.verbose
    qmodel_path = args.qmodel_path
    semantic_scorer = args.semantic_scorer
    defense_policy = args.defense_policy

# Build output filename including defense info
defense_suffix = ''
if defense_type != 'none':
    defense_suffix = f'_{defense_type}'
    if defense_param > 0:
        defense_suffix += f'_{defense_param}'

FILE_NAME = f'results_{task}_{targeted}_{attack_model_type}_{victim_model_type}{defense_suffix}.txt'
if out_dir and (out_dir / FILE_NAME).exists():
    print("Report found, exiting...")
    sys.exit()

# Prepare task data
with_pairs = (task == 'FC' or task == 'C19')

# Determine pretrained tokenizer name (depends only on victim type, not on weights)
if victim_model_type == 'BERT':
    pretrained_model_victim = PRETRAINED_BERT
elif victim_model_type == 'GEMMA':
    pretrained_model_victim = PRETRAINED_GEMMA_2B
elif victim_model_type == 'GEMMA7B':
    pretrained_model_victim = PRETRAINED_GEMMA_7B
elif victim_model_type == 'BiLSTM':
    pretrained_model_victim = 'bert-base-uncased'

# Load data BEFORE initialising MPS/CUDA device.
# Dataset.from_generator forks a subprocess internally; forking after MPS is
# initialised causes a mutex-lock crash on macOS (Abort trap 6).
print("Loading data...")
test_dataset = Dataset.from_generator(readfromfile_generator,
                                      gen_kwargs={'subset': 'attack', 'dir': data_path,
                                                  'pretrained_model': pretrained_model_victim, 'trim_text': True,
                                                  'with_pairs': with_pairs},
                                                  keep_in_memory = True)
if not with_pairs:
    dataset = test_dataset.map(function=dataset_mapping)
    dataset = dataset.remove_columns(["text"])
else:
    dataset = test_dataset.map(function=dataset_mapping_pairs)
    dataset = dataset.remove_columns(["text1", "text2"])

dataset = dataset.remove_columns(["fake"])

# Choose device (after dataset is loaded to avoid fork-after-MPS crash on macOS)
print("Setting up the device...")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:100"

if torch.cuda.is_available():
    victim_device = torch.device("cuda")
    attacker_device = torch.device("cuda")
elif torch.backends.mps.is_available():
   victim_device = torch.device('mps')
   attacker_device = torch.device('mps')
else:
    victim_device = torch.device("cpu")
    attacker_device = torch.device('cpu')

# Prepare victim
print("Loading up victim model...")
if victim_model_type == 'BERT':
    base_victim = VictimTransformer(victim_model_path, task, pretrained_model_victim, False, victim_device)
elif victim_model_type == 'GEMMA':
    base_victim = VictimTransformer(victim_model_path, task, pretrained_model_victim, True, victim_device)
elif victim_model_type == 'GEMMA7B':
    base_victim = VictimTransformer(victim_model_path, task, pretrained_model_victim, True, victim_device)
elif victim_model_type == 'BiLSTM':
    base_victim = VictimBiLSTM(victim_model_path, task, victim_device)

# Apply defense wrapper if specified
defended_victim = None
if defense_type == 'macabeu':
    from defenses.macabeu_defense import RLDefenseSelector
    macabeu_policy = defense_policy
    if not macabeu_policy:
        macabeu_policy = str(pathlib.Path.home() / 'macabeu' / 'models' /
                             f'{task}_{victim_model_type}.pth')
    print(f"Applying MACABEU defense (policy={macabeu_policy})...")
    defended_victim = RLDefenseSelector(base_victim, macabeu_policy, verbose=verbose)
    victim = defended_victim
elif defense_type != 'none':
    print(f"Applying {defense_type} defense (param={defense_param})...")
    defended_victim = get_defense(defense_type, base_victim, param=defense_param, seed=defense_seed, verbose=verbose)
    # Skip caching when defense is active - defense transforms input so cache would be invalid
    victim = defended_victim
else:
    victim = VictimCache(victim_model_path, base_victim)
# dataset = dataset.select(range(10))

# Filter data - save predictions from the filter pass to avoid re-querying a stochastic defense.
# Re-querying after filtering would give different results for stochastic defenses (e.g. MajorityVote),
# making the confusion matrix show wrong "before attack" accuracy.
y_pred_before = []
if targeted:
    # Preserve the original short-circuit: only query the model for Class 1 samples.
    # Querying Class 0 samples first would advance the RNG state and change which Class 1
    # samples the stochastic defense classifies correctly, altering the attacked subset.
    filtered = []
    for inst in dataset:
        if inst["y"] == 1:
            pred = victim.get_pred([inst["x"]])[0]
            if pred == inst["y"]:
                filtered.append(inst)
                y_pred_before.append(int(pred))
    dataset = filtered
else:
    print("Collecting original predictions for confusion matrix...")
    y_pred_before = [int(victim.get_pred([inst["x"]])[0]) for inst in dataset]
print("Subset size: " + str(len(dataset)))
# dataset = dataset [-10:]
attack_texts = [inst["x"] for inst in dataset]

y_true = [inst["y"] for inst in dataset]
class_distribution = Counter(y_true)
print(f"Class distribution: {dict(class_distribution)}")

# Prepare attack
print("Setting up the attacker...")
protected_tokens = ['~'] if task == 'FC' else []

# Use custom qmodel_path if provided, otherwise use default path
if qmodel_path:
    attack_model_path = pathlib.Path(qmodel_path)
    print(f"Using custom Q-model path: {attack_model_path}")
else:
    attack_model_path = pathlib.Path.home() / 'data' / 'xarello' / 'models' / attack_model_variant / (
            task + '-' + victim_model_type + '-2') / 'xarello-qmodel.pth'
pretrained_model_attacker = "bert-base-cased"
attack_env = EnvAE(pretrained_model_attacker, attack_texts, victim, attacker_device, static_embedding=True,
                   protected_tokens=protected_tokens)
if attack_model_type == 'XARELLO':
    attacker = QAttacker(attack_model_path, pretrained_model_attacker, attack_env, torch.device('cpu'), attacker_device,
                         long_text=(task in ['HN', 'RD']))
elif attack_model_type == 'random':
    attacker = QAttacker(None, pretrained_model_attacker, attack_env, torch.device('cpu'), attacker_device,
                         long_text=(task in ['HN', 'RD']))

# Run the attack
print("Evaluating the attack...")
# Include defense_suffix so each defense run keeps its own raw file (avoids overwriting)
RAW_FILE_NAME = f'raw_{task}_{targeted}_{attack_model_type}_{victim_model_type}{defense_suffix}.tsv'
raw_path = out_dir / RAW_FILE_NAME if out_dir else None
with no_ssl_verify():
    # PR2 is a single-sentence task: align_sentences=True would call LAMBO but achieves nothing
    align_sents = (task != 'PR2')
    scorer = BODEGAScore(victim_device, task, align_sentences=align_sents, semantic_scorer=semantic_scorer, raw_path=raw_path)
with no_ssl_verify():
    attack_eval = OpenAttack.AttackEval(attacker, victim, language='english', metrics=[
        scorer  # , OpenAttack.metric.EditDistance()
    ])
    start = time.time()
    summary = attack_eval.eval(dataset, visualize=True, progress_bar=False)
    end = time.time()
attack_time = end - start
attacker = None

# Save defense modifications if any
if defended_victim is not None and out_dir:
    modifications = defended_victim.get_modifications()
    if modifications:
        mod_file = out_dir / f'modifications_{task}_{targeted}_{attack_model_type}_{victim_model_type}{defense_suffix}.tsv'
        defended_victim.save_modifications(str(mod_file))
        print(f"Saved {len(modifications)} defense modifications to {mod_file}")

# Remove unused stuff
if hasattr(victim, 'finalise'):
    victim.finalise()
del victim
gc.collect()
torch.cuda.empty_cache()
if "TOKENIZERS_PARALLELISM" in os.environ:
    del os.environ["TOKENIZERS_PARALLELISM"]

# Evaluate (batch computation of semantic similarity and BODEGA score)
start = time.time()
score_success, score_semantic, score_character, score_BODEGA = scorer.compute()
end = time.time()
evaluate_time = end - start

# OpenAttack's summary table above shows 'nan' for BODEGA Score because BODEGAScore uses
# deferred (batch) computation. Now that we have the real value, inject it and reprint.
from OpenAttack.utils import result_visualizer
summary["Avg. BODEGA Score"] = score_BODEGA
print("\nCorrected summary (BODEGA Score computed after batch evaluation):")
result_visualizer(summary, sys.stdout.write)

# Print results
print("=" * 50)
print("EXPERIMENT CONFIGURATION")
print("=" * 50)
print(f"Task: {task}")
print(f"Targeted: {targeted}")
print(f"Attack: {attack_model_type}")
print(f"Victim: {victim_model_type}")
print(f"Defense: {defense_type}")
if defense_type == 'macabeu':
    print(f"Defense policy: {defense_policy}")
elif defense_type != 'none':
    print(f"Defense param: {defense_param}")
    print(f"Defense seed: {defense_seed}")
print(f"Semantic scorer: {semantic_scorer}")
print("=" * 50)
print("RESULTS")
print("=" * 50)
print("Subset size: " + str(len(dataset)))
print("Success score: " + str(score_success))
print("Semantic score: " + str(score_semantic))
print("Character score: " + str(score_character))
print("BODEGA score: " + str(score_BODEGA))
print("Queries per example: " + str(summary['Avg. Victim Model Queries']))
print("Total attack time: " + str(attack_time))
print("Time per example: " + str((attack_time) / len(dataset)))
print("Total evaluation time: " + str(evaluate_time))

# Compute and print confusion matrix
# y_pred_before was saved during the filter pass (not re-queried), so these numbers are accurate
# even for stochastic defenses.
print("")
print("=" * 50)
print("CONFUSION MATRIX (Before Attack)")
print("=" * 50)
correct_by_class = {}
total_by_class = {}
for yt, yp in zip(y_true, y_pred_before):
    total_by_class[yt] = total_by_class.get(yt, 0) + 1
    if yt == yp:
        correct_by_class[yt] = correct_by_class.get(yt, 0) + 1
for cls in sorted(total_by_class.keys()):
    corr = correct_by_class.get(cls, 0)
    tot = total_by_class[cls]
    print(f"Class {cls}: {corr}/{tot} correct ({100*corr/tot:.1f}%)")
total_corr = sum(correct_by_class.values())
total_tot = sum(total_by_class.values())
print(f"Overall: {total_corr}/{total_tot} correct ({100*total_corr/total_tot:.1f}%)")

# Attack success by class - computed directly from scorer.promises (promise.s2 is None for failures)
print("")
print("=" * 50)
print("ATTACK SUCCESS BY CLASS")
print("=" * 50)
asc = {}
for promise, y in zip(scorer.promises, y_true):
    if y not in asc:
        asc[y] = {"success": 0, "total": 0}
    asc[y]["total"] += 1
    if promise.s2 is not None:
        asc[y]["success"] += 1
for cls in sorted(asc.keys()):
    succ = asc[cls]["success"]
    total = asc[cls]["total"]
    print(f"Class {cls}: {succ}/{total} attacks succeeded ({100*succ/max(1,total):.1f}%)")

if out_dir:
    with open(out_dir / FILE_NAME, 'w') as f:
        f.write("# Experiment Configuration\n")
        f.write(f"Task: {task}\n")
        f.write(f"Targeted: {targeted}\n")
        f.write(f"Attack: {attack_model_type}\n")
        f.write(f"Victim: {victim_model_type}\n")
        f.write(f"Defense: {defense_type}\n")
        f.write(f"Defense param: {defense_param}\n")
        f.write(f"Defense seed: {defense_seed}\n")
        f.write("\n# Results\n")
        f.write("Subset size: " + str(len(dataset)) + '\n')
        f.write("Success score: " + str(score_success) + '\n')
        f.write("Semantic score: " + str(score_semantic) + '\n')
        f.write("Character score: " + str(score_character) + '\n')
        f.write("BODEGA score: " + str(score_BODEGA) + '\n')
        f.write("Queries per example: " + str(summary['Avg. Victim Model Queries']) + '\n')
        f.write("Total attack time: " + str(attack_time) + '\n')
        f.write("Time per example: " + str((attack_time) / len(dataset)) + '\n')
        f.write("Total evaluation time: " + str(evaluate_time) + '\n')
        
        # Write confusion matrix
        f.write("\n# Confusion Matrix (Before Attack)\n")
        for cls in sorted(total_by_class.keys()):
            corr = correct_by_class.get(cls, 0)
            tot = total_by_class[cls]
            f.write(f"Class {cls}: {corr}/{tot} correct ({100*corr/tot:.1f}%)\n")
        f.write(f"Overall: {total_corr}/{total_tot} correct ({100*total_corr/total_tot:.1f}%)\n")
        # Write attack success by class
        f.write("\n# Attack Success By Class\n")
        for cls in sorted(asc.keys()):
            succ = asc[cls]["success"]
            total = asc[cls]["total"]
            f.write(f"Class {cls}: {succ}/{total} attacks succeeded ({100*succ/max(1,total):.1f}%)\n")
