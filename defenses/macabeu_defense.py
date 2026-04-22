"""Self-contained MACABEU RL defense selector for use within XARELLO.

Bundles the MACABEU inference components (TextFeatureExtractor, DefensePolicy,
RLDefenseSelector) to avoid namespace collisions with xarello/agent/.

MACABEU's action space uses BODEGA defense names. The BODEGA defenses module
is loaded explicitly from BODEGA_PATH to avoid clashing with xarello/defenses/.
"""

import importlib.util
import os
import random
import re
from collections import Counter
from typing import List, Tuple

import numpy as np
import OpenAttack
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# BODEGA defense loader (explicit path to avoid xarello/defenses/ collision)
# ---------------------------------------------------------------------------

def _load_bodega_get_defense():
    """Load BODEGA's get_defense from explicit path."""
    bodega_path = os.environ.get(
        'BODEGA_PATH', os.path.expanduser('~/BODEGA'))
    module_path = os.path.join(bodega_path, 'defenses', 'preprocessing.py')
    if not os.path.isfile(module_path):
        raise FileNotFoundError(
            f"BODEGA defenses not found at {module_path}. "
            f"Set BODEGA_PATH environment variable.")
    spec = importlib.util.spec_from_file_location(
        'bodega_preprocessing', module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_defense


# ---------------------------------------------------------------------------
# Text feature extractor (from macabeu/agent/features.py)
# ---------------------------------------------------------------------------

class TextFeatureExtractor:
    FEATURE_NAMES = [
        'text_length', 'word_count', 'avg_word_length',
        'oov_ratio', 'non_ascii_ratio', 'uppercase_ratio',
        'punctuation_ratio', 'digit_ratio', 'repeated_char_ratio',
        'char_entropy'
    ]
    NUM_FEATURES = len(FEATURE_NAMES)

    def __init__(self):
        self._vocab = None

    @property
    def vocab(self):
        if self._vocab is None:
            try:
                from symspellpy import SymSpell
                import importlib.resources
                ss = SymSpell()
                dict_path = (importlib.resources.files("symspellpy")
                             / "frequency_dictionary_en_82_765.txt")
                ss.load_dictionary(str(dict_path), term_index=0, count_index=1)
                self._vocab = set(ss.words.keys())
            except (ImportError, Exception):
                self._vocab = set()
        return self._vocab

    def extract(self, text: str) -> np.ndarray:
        chars = list(text)
        words = text.split()
        n_chars = max(len(chars), 1)
        n_words = max(len(words), 1)
        features = np.zeros(self.NUM_FEATURES, dtype=np.float32)
        features[0] = min(n_chars / 2000.0, 1.0)
        features[1] = min(n_words / 400.0, 1.0)
        features[2] = (np.mean([len(w) for w in words]) / 20.0
                       if words else 0.0)
        word_cores = [re.sub(r'[^a-zA-Z]', '', w.lower()) for w in words]
        word_cores = [w for w in word_cores if len(w) > 0]
        if word_cores and self.vocab:
            oov = sum(1 for w in word_cores if w not in self.vocab)
            features[3] = oov / len(word_cores)
        features[4] = sum(1 for c in chars if ord(c) > 127) / n_chars
        features[5] = sum(1 for c in chars if c.isupper()) / n_chars
        features[6] = sum(
            1 for c in chars if c in '.,!?;:"\'-()[]{}') / n_chars
        features[7] = sum(1 for c in chars if c.isdigit()) / n_chars
        repeated = sum(
            1 for i in range(1, len(chars)) if chars[i] == chars[i - 1])
        features[8] = repeated / n_chars
        counter = Counter(chars)
        probs = np.array(
            list(counter.values()), dtype=np.float32) / n_chars
        features[9] = -np.sum(probs * np.log2(probs + 1e-10)) / 8.0
        return features


# ---------------------------------------------------------------------------
# Q-network and policy (from macabeu/agent/q_network.py)
# ---------------------------------------------------------------------------

class DefenseQNetwork(nn.Module):
    def __init__(self, n_features: int, n_actions: int,
                 hidden_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_actions)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


class DefensePolicy:
    def __init__(self, n_features: int, n_actions: int,
                 action_names: List[str],
                 lr: float = 1e-3,
                 max_eps: float = 1.0,
                 min_eps: float = 0.05,
                 warmup_steps: int = 500,
                 device: torch.device = None):
        self.n_actions = n_actions
        self.action_names = action_names
        self.device = device or torch.device('cpu')
        self.q_net = DefenseQNetwork(n_features, n_actions).to(self.device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.eps = max_eps
        self.max_eps = max_eps
        self.min_eps = min_eps
        self.warmup_steps = warmup_steps
        self.step_count = 0

    def select_action(self, features: np.ndarray,
                      greedy: bool = False) -> int:
        if not greedy and random.random() < self.eps:
            return random.randint(0, self.n_actions - 1)
        with torch.no_grad():
            feat_t = torch.tensor(
                features, dtype=torch.float32
            ).unsqueeze(0).to(self.device)
            q_values = self.q_net(feat_t)
            return q_values.argmax(dim=1).item()

    def update(self, features_batch: np.ndarray, actions_batch: np.ndarray,
               rewards_batch: np.ndarray) -> float:
        feat_t = torch.tensor(features_batch, dtype=torch.float32).to(self.device)
        act_t = torch.tensor(actions_batch, dtype=torch.long).to(self.device)
        rew_t = torch.tensor(rewards_batch, dtype=torch.float32).to(self.device)
        q_all = self.q_net(feat_t)
        q_selected = q_all.gather(1, act_t.unsqueeze(1)).squeeze(1)
        loss = self.loss_fn(q_selected, rew_t)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.step_count += 1
        if self.step_count < self.warmup_steps:
            self.eps = self.max_eps - (self.max_eps - self.min_eps) * self.step_count / self.warmup_steps
        else:
            self.eps = self.min_eps
        return loss.item()

    def save(self, path: str):
        torch.save({
            'q_net': self.q_net.state_dict(),
            'step_count': self.step_count,
            'eps': self.eps,
            'action_names': self.action_names,
            'n_features': self.q_net.network[0].in_features,
            'n_actions': self.q_net.network[-1].out_features,
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(
            path, map_location=self.device, weights_only=False)
        self.q_net.load_state_dict(checkpoint['q_net'])
        self.step_count = checkpoint.get('step_count', 0)
        self.eps = checkpoint.get('eps', self.min_eps)


# ---------------------------------------------------------------------------
# Action space (from macabeu/agent/defense_env.py)
# ---------------------------------------------------------------------------

DEFAULT_ACTION_SPACE = [
    ('none', 0.0),
    ('spellcheck', 0.0),
    ('unicode', 0.0),
    ('majority_vote', 3),
    ('majority_vote', 7),
    ('discretize', 0.0),
    ('spellcheck_mv', 3),
    ('char_noise', 0.10),
]

DEFENSE_COSTS = {
    0: 0.0,
    1: 0.0,
    2: 0.0,
    3: 0.05,
    4: 0.10,
    5: 0.0,
    6: 0.05,
    7: 0.0,
}


class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer = []
        self.index = 0

    def push(self, features: np.ndarray, action: int, reward: float):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.index] = (features.copy(), action, reward)
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        features = np.array([b[0] for b in batch])
        actions = np.array([b[1] for b in batch])
        rewards = np.array([b[2] for b in batch])
        return features, actions, rewards

    def __len__(self):
        return len(self.buffer)


def get_action_names(action_space=None):
    action_space = action_space or DEFAULT_ACTION_SPACE
    names = []
    for name, param in action_space:
        if param == 0.0 or name == 'none':
            names.append(name)
        else:
            names.append(f"{name}@{param:g}")
    return names


# ---------------------------------------------------------------------------
# RLDefenseSelector (from macabeu/agent/defense_selector.py)
# ---------------------------------------------------------------------------

class RLDefenseSelector(OpenAttack.Classifier):
    """Adaptive defense selector using MACABEU's trained Q-network."""

    def __init__(self, victim, policy_path: str,
                 action_space=None, seed: int = 42,
                 verbose: bool = False):
        self.victim = victim
        self.action_space = action_space or DEFAULT_ACTION_SPACE
        self.verbose = verbose
        self.feature_extractor = TextFeatureExtractor()

        n_features = TextFeatureExtractor.NUM_FEATURES
        n_actions = len(self.action_space)
        action_names = get_action_names(self.action_space)

        self.policy = DefensePolicy(
            n_features=n_features, n_actions=n_actions,
            action_names=action_names)
        self.policy.load(policy_path)
        self.policy.q_net.eval()

        # Load BODEGA's get_defense (not xarello's)
        bodega_get_defense = _load_bodega_get_defense()

        self.defenses = {}
        for i, (defense_name, param) in enumerate(self.action_space):
            if defense_name == 'none':
                self.defenses[i] = victim
            else:
                self.defenses[i] = bodega_get_defense(
                    defense_name, victim, param=param,
                    seed=seed, verbose=verbose)

        self.action_counts = np.zeros(n_actions, dtype=int)

    def get_pred(self, input_: List[str]) -> np.ndarray:
        return self.get_prob(input_).argmax(axis=1)

    def get_prob(self, input_: List[str]) -> np.ndarray:
        all_probs = []
        for text in input_:
            features = self.feature_extractor.extract(text)
            action = self.policy.select_action(features, greedy=True)
            self.action_counts[action] += 1
            if self.verbose:
                print(f"[MACABEU] {self.policy.action_names[action]}"
                      f" | {text[:60]}...")
            prob = self.defenses[action].get_prob([text])
            all_probs.append(prob[0])
        return np.array(all_probs)

    def get_action_statistics(self) -> dict:
        total = max(self.action_counts.sum(), 1)
        stats = {}
        for i, name in enumerate(self.policy.action_names):
            count = int(self.action_counts[i])
            stats[name] = {
                'count': count,
                'pct': round(count / total * 100, 1)
            }
        return stats

    def get_modifications(self):
        return []

    def save_modifications(self, path: str):
        pass

    def clear_modifications(self):
        pass

    def finalise(self):
        if hasattr(self.victim, 'finalise'):
            self.victim.finalise()


# ---------------------------------------------------------------------------
# OnlineRLDefenseSelector (from macabeu/agent/online_selector.py)
# ---------------------------------------------------------------------------

class OnlineRLDefenseSelector(OpenAttack.Classifier):
    """Adaptive defense selector that learns online during attacks.

    Each call to get_prob/get_pred extracts features, picks a defense via
    epsilon-greedy, and tracks the last (features, action) for later reward
    assignment. Call `observe_result(true_label, pred_after_attack)` after
    each attacked example to inject the reward and update the Q-network.
    """

    def __init__(self, victim, action_space=None, seed: int = 42,
                 lr: float = 1e-3, max_eps: float = 1.0,
                 min_eps: float = 0.05, warmup_examples: int = 50,
                 batch_size: int = 16, buffer_size: int = 5000,
                 pretrained_path: str = None, verbose: bool = False):
        self.victim = victim
        self.action_space = action_space or DEFAULT_ACTION_SPACE
        self.verbose = verbose
        self.batch_size = batch_size
        self.feature_extractor = TextFeatureExtractor()

        n_features = TextFeatureExtractor.NUM_FEATURES
        n_actions = len(self.action_space)
        action_names = get_action_names(self.action_space)

        self.policy = DefensePolicy(
            n_features=n_features, n_actions=n_actions,
            action_names=action_names, lr=lr,
            max_eps=max_eps, min_eps=min_eps,
            warmup_steps=warmup_examples)

        if pretrained_path:
            self.policy.load(pretrained_path)
            self.policy.eps = max_eps
            self.policy.step_count = 0
            self.policy.warmup_steps = warmup_examples

        self.replay_buffer = ReplayBuffer(capacity=buffer_size)

        bodega_get_defense = _load_bodega_get_defense()
        self.defenses = {}
        for i, (defense_name, param) in enumerate(self.action_space):
            if defense_name == 'none':
                self.defenses[i] = victim
            else:
                self.defenses[i] = bodega_get_defense(
                    defense_name, victim, param=param,
                    seed=seed, verbose=False)

        self._current_features = None
        self._current_action = None
        self._current_text = None

        self.action_counts = np.zeros(n_actions, dtype=int)
        self.example_count = 0
        self.total_reward = 0.0
        self.reward_history = []

    def get_pred(self, input_: List[str]) -> np.ndarray:
        return self.get_prob(input_).argmax(axis=1)

    def get_prob(self, input_: List[str]) -> np.ndarray:
        all_probs = []
        for text in input_:
            features = self.feature_extractor.extract(text)
            action = self.policy.select_action(features, greedy=False)
            self.action_counts[action] += 1

            self._current_features = features
            self._current_action = action
            self._current_text = text

            if self.verbose:
                print(f"[ONLINE_RL] eps={self.policy.eps:.3f} "
                      f"action={self.policy.action_names[action]} "
                      f"| {text[:50]}...")

            prob = self.defenses[action].get_prob([text])
            all_probs.append(prob[0])
        return np.array(all_probs)

    def observe_result(self, true_label: int, prediction_after_attack: int):
        if self._current_features is None:
            return

        correct = 1.0 if prediction_after_attack == true_label else -1.0
        cost = DEFENSE_COSTS.get(self._current_action, 0.0)
        reward = correct - cost

        self.replay_buffer.push(
            self._current_features, self._current_action, reward)

        self.total_reward += reward
        self.example_count += 1
        self.reward_history.append(reward)

        if len(self.replay_buffer) >= self.batch_size:
            f_batch, a_batch, r_batch = self.replay_buffer.sample(self.batch_size)
            loss = self.policy.update(f_batch, a_batch, r_batch)
            if self.verbose and self.example_count % 20 == 0:
                avg_recent = np.mean(self.reward_history[-20:])
                print(f"[ONLINE_RL] Example {self.example_count} | "
                      f"loss={loss:.4f} | eps={self.policy.eps:.3f} | "
                      f"avg_reward(20)={avg_recent:.3f}")

        self._current_features = None
        self._current_action = None

    def get_action_statistics(self) -> dict:
        total = max(self.action_counts.sum(), 1)
        stats = {}
        for i, name in enumerate(self.policy.action_names):
            count = int(self.action_counts[i])
            stats[name] = {'count': count, 'pct': round(count / total * 100, 1)}
        return stats

    def get_learning_curve(self, window: int = 20) -> List[float]:
        if len(self.reward_history) < window:
            return list(self.reward_history)
        curve = []
        for i in range(window, len(self.reward_history) + 1):
            curve.append(float(np.mean(self.reward_history[i - window:i])))
        return curve

    def get_modifications(self):
        return []

    def save_modifications(self, path: str):
        pass

    def clear_modifications(self):
        pass

    def finalise(self):
        if hasattr(self.victim, 'finalise'):
            self.victim.finalise()

    def save(self, path: str):
        self.policy.save(path)
