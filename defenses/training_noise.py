"""
Training-time noise wrappers for XARELLO.

These wrappers add noise to the victim's responses during XARELLO training
to corrupt the Q-learning reward signal. The hypothesis: noisy responses
during training will produce a corrupted policy that fails even when
evaluated against a clean victim.

This is different from eval-time defenses (in preprocessing.py) which
modify the input text. Training noise operates on the OUTPUT (predictions
and rewards) to corrupt XARELLO's learning.

Usage:
    victim = OpenAttackVictimWrapper(VictimBiLSTM(...), tokeniser)
    noisy_victim = get_training_noise_wrapper('label_flip', victim, epsilon=0.1, seed=42)
    train_env = EnvAE(..., victim=noisy_victim, ...)
"""

import random
from typing import Optional

import numpy as np

from ae_victims.victim import Victim


class NoisyVictimWrapper(Victim):
    """Base class for training noise wrappers.

    Wraps an OpenAttackVictimWrapper and adds noise to its outputs.
    Implements the same Victim interface (pred, pred_changed).
    """

    def __init__(self, victim: Victim, seed: Optional[int] = None):
        self.victim = victim
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

    def pred(self, tokens_original):
        return self.victim.pred(tokens_original)

    def pred_changed(self, tokens_original, tokens_modified, tokens_previous):
        return self.victim.pred_changed(tokens_original, tokens_modified, tokens_previous)


class LabelFlipWrapper(NoisyVictimWrapper):
    """Exp 2a: With probability epsilon, return flipped prediction/reward.

    This simulates a "random oracle" defense where the victim occasionally
    lies about its predictions. XARELLO will learn from incorrect feedback.

    Args:
        victim: The underlying victim (OpenAttackVictimWrapper)
        epsilon: Probability of flipping the response (0.0 to 1.0)
        seed: Random seed for reproducibility
    """

    def __init__(self, victim: Victim, epsilon: float = 0.1, seed: Optional[int] = None):
        super().__init__(victim, seed)
        self.epsilon = epsilon

    def pred(self, tokens_original):
        pred = self.victim.pred(tokens_original)
        if self.rng.random() < self.epsilon:
            return 1 - pred  # Flip 0↔1
        return pred

    def pred_changed(self, tokens_original, tokens_modified, tokens_previous):
        result = self.victim.pred_changed(tokens_original, tokens_modified, tokens_previous)
        if self.rng.random() < self.epsilon:
            if result == 1.0:
                # Success → failure (hide successful attack)
                return 0.0
            elif result == 0.0:
                # Complete failure → spurious success (wrong termination signal)
                return 1.0
            else:
                # Partial reward → negate it (mislead gradient direction)
                return -result
        return result


class ThresholdNoiseWrapper(NoisyVictimWrapper):
    """Exp 2b: Simulate noisy decision threshold by randomly flipping decisions.

    This simulates uncertainty in the victim's decision boundary. Near-threshold
    samples would naturally be uncertain, so we model this by probabilistically
    flipping outcomes with probability proportional to sigma.

    Args:
        victim: The underlying victim (OpenAttackVictimWrapper)
        sigma: Noise level (probability of flipping, 0.0 to 1.0)
        seed: Random seed for reproducibility
    """

    def __init__(self, victim: Victim, sigma: float = 0.1, seed: Optional[int] = None):
        super().__init__(victim, seed)
        self.sigma = sigma

    def pred(self, tokens_original):
        pred = self.victim.pred(tokens_original)
        # Simulate threshold noise: with prob proportional to sigma, flip prediction
        if self.rng.random() < self.sigma:
            return 1 - pred
        return pred

    def pred_changed(self, tokens_original, tokens_modified, tokens_previous):
        result = self.victim.pred_changed(tokens_original, tokens_modified, tokens_previous)

        # For terminal reward (1.0 = success): occasionally report false success/failure
        if result == 1.0:
            if self.rng.random() < self.sigma:
                return 0.0  # False negative: hide actual success
        elif result <= 0.0:
            if self.rng.random() < self.sigma * 0.5:
                return 1.0  # False positive: report spurious success (rare)

        # For partial rewards: add Gaussian noise scaled by sigma
        if result != 1.0:
            noise = self.np_rng.normal(0, self.sigma)
            result = float(np.clip(result + noise, -1.0, 0.99))

        return result


class ConfidenceNoiseWrapper(NoisyVictimWrapper):
    """Exp 2c: Add Gaussian noise to reward values.

    This is a simplified version that adds noise to the reward signal
    without accessing the underlying model's probabilities. It corrupts
    the Q-learning gradient signal by making partial rewards unreliable.

    Args:
        victim: The underlying victim (OpenAttackVictimWrapper)
        sigma: Standard deviation of Gaussian noise (0.0 to 1.0)
        seed: Random seed for reproducibility
    """

    def __init__(self, victim: Victim, sigma: float = 0.1, seed: Optional[int] = None):
        super().__init__(victim, seed)
        self.sigma = sigma

    def pred(self, tokens_original):
        pred = self.victim.pred(tokens_original)
        # Small chance of flip proportional to sigma
        if self.rng.random() < self.sigma * 0.5:
            return 1 - pred
        return pred

    def pred_changed(self, tokens_original, tokens_modified, tokens_previous):
        result = self.victim.pred_changed(tokens_original, tokens_modified, tokens_previous)

        # Don't modify terminal success signal (1.0) - that would break episode termination
        # Only add noise to partial rewards (the probability deltas)
        if result != 1.0:
            noise = self.np_rng.normal(0, self.sigma)
            result = float(np.clip(result + noise, -1.0, 0.99))

        return result


def get_training_noise_wrapper(
    noise_type: str,
    victim: Victim,
    param: float = 0.1,
    seed: Optional[int] = None
) -> Victim:
    """Factory function to create a training noise wrapper.

    Args:
        noise_type: Type of noise ('label_flip', 'threshold', 'confidence', or 'none')
        victim: The underlying victim (OpenAttackVictimWrapper)
        param: Noise parameter (epsilon for label_flip, sigma for threshold/confidence)
        seed: Random seed for reproducibility

    Returns:
        A Victim wrapper that adds noise to responses during training

    Example:
        victim = OpenAttackVictimWrapper(VictimBiLSTM(...), tokeniser)
        noisy_victim = get_training_noise_wrapper('label_flip', victim, 0.1, 42)
    """
    noise_type = noise_type.lower()

    if noise_type == 'none' or noise_type == '':
        return victim
    elif noise_type == 'label_flip':
        return LabelFlipWrapper(victim, epsilon=param, seed=seed)
    elif noise_type == 'threshold':
        return ThresholdNoiseWrapper(victim, sigma=param, seed=seed)
    elif noise_type == 'confidence':
        return ConfidenceNoiseWrapper(victim, sigma=param, seed=seed)
    else:
        raise ValueError(
            f"Unknown training noise type: {noise_type}. "
            f"Available: none, label_flip, threshold, confidence"
        )
