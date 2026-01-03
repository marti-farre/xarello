import gc
import os
import pathlib
import sys
import time
import random
import numpy as np

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

# Attempt at determinism
random.seed(10)
torch.manual_seed(10)
np.random.seed(0)

# Running variables
print("Preparing the environment...")
task = 'PR2'
targeted = True
victim_model_type = 'BERT'
attack_model_type = 'XARELLO'
attack_model_variant = 'wide'
out_dir = None
data_path = pathlib.Path.home() / 'data' / 'BODEGA' / task
victim_model_path = pathlib.Path.home() / 'data' / 'BODEGA' / task / (victim_model_type + '-512.pth')
if len(sys.argv) >= 7:
    task = sys.argv[1]
    targeted = (sys.argv[2].lower() == 'true')
    attack_model_type = sys.argv[3]
    victim_model_type = sys.argv[4]
    data_path = pathlib.Path(sys.argv[5])
    victim_model_path = pathlib.Path(sys.argv[6])
    if len(sys.argv) == 8:
        out_dir = pathlib.Path(sys.argv[7])

FILE_NAME = 'results_' + task + '_' + str(targeted) + '_' + attack_model_type + '_' + victim_model_type + '.txt'
if out_dir and (out_dir / FILE_NAME).exists():
    print("Report found, exiting...")
    sys.exit()

# Prepare task data
with_pairs = (task == 'FC' or task == 'C19')

# Choose device
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
    pretrained_model_victim = PRETRAINED_BERT
    victim = VictimCache(victim_model_path,
                         VictimTransformer(victim_model_path, task, pretrained_model_victim, False, victim_device))
elif victim_model_type == 'GEMMA':
    pretrained_model_victim = PRETRAINED_GEMMA_2B
    victim = VictimCache(victim_model_path,
                         VictimTransformer(victim_model_path, task, pretrained_model_victim, True, victim_device))
elif victim_model_type == 'GEMMA7B':
    pretrained_model_victim = PRETRAINED_GEMMA_7B
    victim = VictimCache(victim_model_path,
                         VictimTransformer(victim_model_path, task, pretrained_model_victim, True, victim_device))
elif victim_model_type == 'BiLSTM':
    pretrained_model_victim = 'bert-base-uncased'  # BiLSTM uses BERT tokenizer
    victim = VictimCache(victim_model_path, VictimBiLSTM(victim_model_path, task, victim_device))

# Load data
print("Loading data...")
test_dataset = Dataset.from_generator(readfromfile_generator,
                                      gen_kwargs={'subset': 'attack', 'dir': data_path,
                                                  'pretrained_model': pretrained_model_victim, 'trim_text': True,
                                                  'with_pairs': with_pairs})
if not with_pairs:
    dataset = test_dataset.map(function=dataset_mapping)
    dataset = dataset.remove_columns(["text"])
else:
    dataset = test_dataset.map(function=dataset_mapping_pairs)
    dataset = dataset.remove_columns(["text1", "text2"])

dataset = dataset.remove_columns(["fake"])
# dataset = dataset.select(range(10))

# Filter data
if targeted:
    dataset = [inst for inst in dataset if inst["y"] == 1 and victim.get_pred([inst["x"]])[0] == inst["y"]]
print("Subset size: " + str(len(dataset)))
# dataset = dataset [-10:]
attack_texts = [inst["x"] for inst in dataset]

# Prepare attack
print("Setting up the attacker...")
protected_tokens = ['~'] if task == 'FC' else []
attack_model_path = pathlib.Path.home() / 'data' / 'xarello' / 'models' / attack_model_variant / (
        task + '-' + victim_model_type) / 'xarello-qmodel.pth'
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
RAW_FILE_NAME = 'raw_' + task + '_' + str(targeted) + '_' + 'XARELLO' + '_' + victim_model_type + '.tsv'
raw_path = out_dir / RAW_FILE_NAME if out_dir else None
with no_ssl_verify():
    scorer = BODEGAScore(victim_device, task, align_sentences=True, semantic_scorer="BLEURT", raw_path=raw_path)
with no_ssl_verify():
    attack_eval = OpenAttack.AttackEval(attacker, victim, language='english', metrics=[
        scorer  # , OpenAttack.metric.EditDistance()
    ])
    start = time.time()
    summary = attack_eval.eval(dataset, visualize=True, progress_bar=False)
    end = time.time()
attack_time = end - start
attacker = None

# Remove unused stuff
victim.finalise()
del victim
gc.collect()
torch.cuda.empty_cache()
if "TOKENIZERS_PARALLELISM" in os.environ:
    del os.environ["TOKENIZERS_PARALLELISM"]

# Evaluate
start = time.time()
score_success, score_semantic, score_character, score_BODEGA = scorer.compute()
end = time.time()
evaluate_time = end - start

# Print results
print("Subset size: " + str(len(dataset)))
print("Success score: " + str(score_success))
print("Semantic score: " + str(score_semantic))
print("Character score: " + str(score_character))
print("BODEGA score: " + str(score_BODEGA))
print("Queries per example: " + str(summary['Avg. Victim Model Queries']))
print("Total attack time: " + str(attack_time))
print("Time per example: " + str((attack_time) / len(dataset)))
print("Total evaluation time: " + str(evaluate_time))

if out_dir:
    with open(out_dir / FILE_NAME, 'w') as f:
        f.write("Subset size: " + str(len(dataset)) + '\n')
        f.write("Success score: " + str(score_success) + '\n')
        f.write("Semantic score: " + str(score_semantic) + '\n')
        f.write("Character score: " + str(score_character) + '\n')
        f.write("BODEGA score: " + str(score_BODEGA) + '\n')
        f.write("Queries per example: " + str(summary['Avg. Victim Model Queries']) + '\n')
        f.write("Total attack time: " + str(end - start) + '\n')
        f.write("Time per example: " + str((end - start) / len(dataset)) + '\n')
        f.write("Total evaluation time: " + str(evaluate_time) + '\n')
