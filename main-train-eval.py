import random, time, torch

import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer
from victims.transformer import VictimTransformer, PRETRAINED_BERT, PRETRAINED_GEMMA_2B, PRETRAINED_GEMMA_7B
from victims.bilstm import VictimBiLSTM
import pathlib, sys

from agent.q_learning import Qlearner
from env.EnvAE import EnvAE, copy_observation
from ae_victims.openattack import OpenAttackVictimWrapper

random.seed(10)
torch.manual_seed(10)
np.random.seed(0)

task = sys.argv[1]
victim_model = sys.argv[2]
outpath_string = sys.argv[3]

pretrained_model = "bert-base-cased"
tokeniser = AutoTokenizer.from_pretrained(pretrained_model)

attacker_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
victim_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#task = 'PR2'
#victim_model = 'BiLSTM'
model_path = pathlib.Path.home() / 'data' / 'BODEGA' / task / (victim_model + '-512.pth')
plot_path = pathlib.Path(outpath_string) #pathlib.Path.home() / 'data' / 'xarello' / 'out'

if victim_model == 'BiLSTM':
    victim = OpenAttackVictimWrapper(VictimBiLSTM(model_path, task, victim_device), tokeniser)
elif victim_model == 'BERT':
    pretrained_model_here = PRETRAINED_BERT
    victim = OpenAttackVictimWrapper(VictimTransformer(model_path, task, pretrained_model_here, False, victim_device), tokeniser)
elif victim_model == 'GEMMA':
    pretrained_model_here = PRETRAINED_GEMMA_2B
    victim = OpenAttackVictimWrapper(VictimTransformer(model_path, task, pretrained_model_here, True, victim_device), tokeniser)
elif victim_model == 'GEMMA7B':
    pretrained_model_here = PRETRAINED_GEMMA_7B
    victim = OpenAttackVictimWrapper(VictimTransformer(model_path, task, pretrained_model_here, True, victim_device), tokeniser)

TRAIN_SIZE = 100
EVAL_SIZE = 20

if task == 'FC':
    all_texts = [
        line.split('\t')[2].strip().replace('\\n', '\n') + ' ~ ' + line.split('\t')[3].strip().replace('\\n', '\n')
        for line in open(pathlib.Path.home() / 'data' / 'BODEGA' / task / 'dev.tsv')]
else:
    all_texts = [line.split('\t')[2].strip().replace('\\n', '\n') for line in
                  open(pathlib.Path.home() / 'data' / 'BODEGA' / task / 'dev.tsv')]
eval_texts = all_texts[:EVAL_SIZE]
train_texts = all_texts[EVAL_SIZE:(EVAL_SIZE + TRAIN_SIZE)]
print("Using train set size: "+str(len(train_texts))+" and eval set size: "+str(len(eval_texts)))

TEXTS_IN_ROUND = len(train_texts)
MAX_EPOCHS = 5
protected_tokens = ['~'] if task == 'FC' else []

train_env = EnvAE(pretrained_model, train_texts, victim, attacker_device, static_embedding=True,
                  protected_tokens=protected_tokens)
eval_env = EnvAE(pretrained_model, eval_texts, victim, attacker_device, static_embedding=True,
                 protected_tokens=protected_tokens)

warmup_fraction = 0.3
print("Using warmup fraction of "+str(warmup_fraction))
warmup_episodes = int(warmup_fraction * len(train_texts) * MAX_EPOCHS * train_env.EPISODES_PER_TEXT)

save_best_model = True
save_all_models = True
save_path=pathlib.Path(outpath_string) / ('xarello-qmodel.pth')
#agent.load_from_path(save_path)

STARTING_EPOCH = 0
STARTING_EPISODES = 0
if STARTING_EPOCH>0:
    # load model
    agent = Qlearner(train_env, pretrained_model, warmup_episodes, attacker_device, longterm_memory=True,
                     starting_episodes=STARTING_EPISODES)
    print("Loading model from epoch "+str(STARTING_EPOCH))
    agent.load_from_path(str(save_path)+'-'+str(STARTING_EPOCH))
    print("Loaded.")
else:
    agent = Qlearner(train_env, pretrained_model, warmup_episodes, attacker_device, longterm_memory=True)



start = time.time()

train_observation, _ = train_env.reset(0)
train_rewards = []
train_successes = []
train_tries = []
eval_observation, _ = eval_env.reset(0)
eval_rewards = []
eval_successes = []
eval_tries = []
round_counter = 0
epoch_counter = STARTING_EPOCH
lowest_tries = float('inf')
while True:
    print('\nROUND ' + str(round_counter))
    if epoch_counter > 0:
        ################### ROUND OF TRAINING ######################
        agent.use_env(train_env)
        agent.only_greedy = False
        rewards_here = []
        successes_here = []
        tries_here = []
        while True:
            # Print info
            print(
                "Train E" + str(epoch_counter) + "-R" + str(round_counter) + "T" + str(
                    train_env.current_text) + "-E" + str(
                    train_env.episode) + "S" + str(
                    train_env.current_steps))
            if train_env.current_steps == 0:
                print(train_env.render())
            # Make an action
            action = agent.choose_action(train_observation)
            # Observe the result
            observation_old = copy_observation(train_observation)
            train_observation, reward, terminated, truncated, _ = train_env.step(action)
            print(train_env.render())
            rewards_here.append(reward)
            # Learn from the experience
            agent.learn(observation_old, action, train_observation, reward, terminated, truncated)
            # Finalise
            if terminated or truncated:
                print("END\n" if terminated else "RESET\n")
                train_observation, _ = train_env.reset()
                if train_env.current_steps == 0:
                    successes_here.append(1 if terminated else 0)
                    tries_here.append(train_env.previous_steps)
                # A round is after data repeats or N texts
                if train_env.current_steps == 0 and (train_env.current_text % TEXTS_IN_ROUND == 0):
                    break
        print("Mean train rewards: " + str(np.mean(rewards_here)))
        train_rewards.append(np.mean(rewards_here))
        print("Mean train success: " + str(np.mean(successes_here)))
        train_successes.append(np.mean(successes_here))
        print("Mean train tries: " + str(np.mean(tries_here)))
        train_tries.append(np.mean(tries_here))
    ################### EVALUATION ######################
    agent.use_env(eval_env)
    agent.only_greedy = True
    rewards_here = []
    successes_here = []
    tries_here = []
    while True:
        # Print info
        print(
            "Evaluation T" + str(eval_env.current_text) + "-E" + str(eval_env.episode) + "S" + str(
                eval_env.current_steps))
        if eval_env.current_steps == 0:
            print(eval_env.render())
        # Make an action
        action = agent.choose_action(eval_observation)
        # Observe the result
        train_observation, reward, terminated, truncated, _ = eval_env.step(action)
        print(eval_env.render())
        rewards_here.append(reward)
        # Finalise
        if terminated or truncated:
            print("END\n" if terminated else "RESET\n")
            eval_observation, _ = eval_env.reset()
            if eval_env.current_steps == 0:
                successes_here.append(1 if terminated else 0)
                tries_here.append(eval_env.previous_steps)
            if eval_env.current_steps == 0 and eval_env.current_text == 0:
                break
    print("Mean eval rewards: " + str(np.mean(rewards_here)))
    eval_rewards.append(np.mean(rewards_here))
    print("Mean eval success: " + str(np.mean(successes_here)))
    eval_successes.append(np.mean(successes_here))
    print("Mean eval tries: " + str(np.mean(tries_here)))
    eval_tries.append(np.mean(tries_here))
    #####################################################
    if train_env.current_text == 0 and train_env.episode == 0:
        # EPOCH end
        if save_best_model and epoch_counter>0 and np.mean(tries_here) < lowest_tries:
            print("Found best model at epoch "+str(epoch_counter))
            lowest_tries = np.mean(tries_here)
            agent.save_to_path(save_path)
        if save_all_models:
            print("Saving model at epoch "+str(epoch_counter)+" with episodes "+str(agent.episodes))
            agent.save_to_path(str(save_path)+'-'+str(epoch_counter))
        if epoch_counter == MAX_EPOCHS:
            print("THE END")
            break
        else:
            epoch_counter += 1
    else:
        round_counter += 1

train_env.close()

end = time.time()
processing_time = end - start

print("Final mean reward value: " + str(np.mean(train_rewards)))
print("Processing time: " + str(processing_time))

# outfile.close()

fig, ax = plt.subplots()
ax.plot(train_rewards)
# plt.show()
plt.savefig(plot_path / "train_reward.pdf", format="pdf", bbox_inches="tight")

fig, ax = plt.subplots()
ax.plot(train_successes)
# plt.show()
plt.savefig(plot_path / "train_success.pdf", format="pdf", bbox_inches="tight")

fig, ax = plt.subplots()
ax.plot(train_tries)
# plt.show()
plt.savefig(plot_path / "train_tries.pdf", format="pdf", bbox_inches="tight")

fig, ax = plt.subplots()
ax.plot(eval_rewards)
# plt.show()
plt.savefig(plot_path / "eval_reward.pdf", format="pdf", bbox_inches="tight")

fig, ax = plt.subplots()
ax.plot(eval_successes)
# plt.show()
plt.savefig(plot_path / "eval_success.pdf", format="pdf", bbox_inches="tight")

fig, ax = plt.subplots()
ax.plot(eval_tries)
# plt.show()
plt.savefig(plot_path / "eval_tries.pdf", format="pdf", bbox_inches="tight")
