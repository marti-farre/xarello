"""
Input preprocessing defenses for adversarial robustness evaluation.

These defenses modify input text before passing to the victim classifier,
aiming to neutralize character-level and word-level perturbations.
"""

import random
from abc import ABC, abstractmethod
from typing import List, Callable, Optional

import numpy as np
import OpenAttack

# SEPARATOR used for text pairs in FC/C19 tasks (from BODEGA)
SEPARATOR = ' ~ '


class DefenseWrapper(OpenAttack.Classifier, ABC):
    """
    Base class for defense wrappers that preprocess input before classification.

    Wraps any OpenAttack.Classifier and applies a defense strategy to inputs
    before delegating to the underlying victim model.
    """

    def __init__(self, victim: OpenAttack.Classifier):
        """
        Args:
            victim: The underlying classifier to wrap
        """
        self.victim = victim

    def get_pred(self, input_: List[str]) -> np.ndarray:
        """Get hard predictions after applying defense."""
        return self.get_prob(input_).argmax(axis=1)

    def get_prob(self, input_: List[str]) -> np.ndarray:
        """Get probability distributions after applying defense."""
        defended_input = self.apply_defense(input_)
        return self.victim.get_prob(defended_input)

    def apply_defense(self, input_: List[str]) -> List[str]:
        """
        Apply defense to a batch of inputs.

        Handles text pairs (separated by SEPARATOR) by applying defense
        to each part independently.
        """
        result = []
        for text in input_:
            if SEPARATOR in text:
                # Handle text pairs (FC/C19 tasks)
                parts = text.split(SEPARATOR)
                defended_parts = [self.defend_single(p) for p in parts]
                result.append(SEPARATOR.join(defended_parts))
            else:
                result.append(self.defend_single(text))
        return result

    @abstractmethod
    def defend_single(self, text: str) -> str:
        """Apply defense to a single text. Must be implemented by subclasses."""
        pass

    def finalise(self):
        """Delegate finalise call to victim (for caching support)."""
        if hasattr(self.victim, 'finalise'):
            self.victim.finalise()


class SpellCheckDefense(DefenseWrapper):
    """
    Defense that corrects spelling errors to counter character-level attacks.

    XARELLO uses character swaps, insertions, and deletions which often
    result in misspellings. This defense attempts to correct them.
    """

    def __init__(self, victim: OpenAttack.Classifier, language: str = 'en'):
        """
        Args:
            victim: The underlying classifier to wrap
            language: Language for spell checking (default: 'en')
        """
        super().__init__(victim)
        self.language = language
        self._spellchecker = None

    @property
    def spellchecker(self):
        """Lazy load spellchecker to avoid import overhead."""
        if self._spellchecker is None:
            try:
                from spellchecker import SpellChecker
                self._spellchecker = SpellChecker(language=self.language)
            except ImportError:
                raise ImportError(
                    "SpellCheckDefense requires pyspellchecker. "
                    "Install with: pip install pyspellchecker"
                )
        return self._spellchecker

    def defend_single(self, text: str) -> str:
        """Correct spelling errors in text."""
        words = text.split()
        corrected = []

        for word in words:
            # Preserve punctuation attached to words
            prefix = ''
            suffix = ''
            core = word

            # Strip leading punctuation
            while core and not core[0].isalnum():
                prefix += core[0]
                core = core[1:]

            # Strip trailing punctuation
            while core and not core[-1].isalnum():
                suffix = core[-1] + suffix
                core = core[:-1]

            if core:
                # Check if word needs correction
                correction = self.spellchecker.correction(core.lower())
                if correction and correction != core.lower():
                    # Preserve original case pattern
                    if core.isupper():
                        correction = correction.upper()
                    elif core[0].isupper():
                        correction = correction.capitalize()
                    corrected.append(prefix + correction + suffix)
                else:
                    corrected.append(word)
            else:
                corrected.append(word)

        return ' '.join(corrected)


class EmbeddingNoiseDefense(DefenseWrapper):
    """
    Defense that adds Gaussian noise to word embeddings.

    This defense is applied at the model level, not the text level.
    It requires access to the victim's embedding layer.

    Note: This is a placeholder implementation that operates at the token level
    by randomly perturbing characters. For true embedding noise, we would need
    to modify the victim model's forward pass directly.
    """

    def __init__(
        self,
        victim: OpenAttack.Classifier,
        noise_std: float = 0.1,
        seed: Optional[int] = None
    ):
        """
        Args:
            victim: The underlying classifier to wrap
            noise_std: Standard deviation of Gaussian noise (0.0 = no noise)
            seed: Random seed for reproducibility
        """
        super().__init__(victim)
        self.noise_std = noise_std
        self.rng = random.Random(seed)

    def defend_single(self, text: str) -> str:
        """
        Apply character-level perturbation as a proxy for embedding noise.

        Since we can't easily access embeddings at inference time without
        modifying the model, we apply random character substitutions with
        visually similar characters.
        """
        if self.noise_std == 0.0:
            return text

        # Character substitution maps (visually similar chars)
        similar_chars = {
            'a': ['@', 'à', 'á', 'ä'],
            'e': ['è', 'é', 'ë', '3'],
            'i': ['1', 'í', 'ì', '!'],
            'o': ['0', 'ò', 'ó', 'ö'],
            'u': ['ù', 'ú', 'ü'],
            's': ['$', '5'],
            'l': ['1', '|'],
            't': ['+', '7'],
        }

        result = []
        for char in text:
            # Probability of perturbation scales with noise_std
            if self.rng.random() < self.noise_std and char.lower() in similar_chars:
                alternatives = similar_chars[char.lower()]
                replacement = self.rng.choice(alternatives)
                if char.isupper():
                    replacement = replacement.upper()
                result.append(replacement)
            else:
                result.append(char)

        return ''.join(result)


class TokenDropoutDefense(DefenseWrapper):
    """
    Defense that randomly drops tokens from input.

    By randomly removing words, this defense can disrupt adversarial
    perturbations that rely on specific token positions or sequences.
    """

    def __init__(
        self,
        victim: OpenAttack.Classifier,
        dropout_prob: float = 0.1,
        seed: Optional[int] = None,
        min_tokens: int = 3
    ):
        """
        Args:
            victim: The underlying classifier to wrap
            dropout_prob: Probability of dropping each token (0.0 - 1.0)
            seed: Random seed for reproducibility
            min_tokens: Minimum number of tokens to keep
        """
        super().__init__(victim)
        self.dropout_prob = dropout_prob
        self.min_tokens = min_tokens
        self.rng = random.Random(seed)

    def defend_single(self, text: str) -> str:
        """Randomly drop tokens from text."""
        if self.dropout_prob == 0.0:
            return text

        words = text.split()
        if len(words) <= self.min_tokens:
            return text

        # Calculate how many tokens we can drop
        max_drop = len(words) - self.min_tokens

        # Decide which tokens to keep
        kept = []
        dropped_count = 0

        for word in words:
            if self.rng.random() >= self.dropout_prob or dropped_count >= max_drop:
                kept.append(word)
            else:
                dropped_count += 1

        # Ensure we have minimum tokens
        if len(kept) < self.min_tokens:
            # Randomly add back some dropped tokens
            kept = words[:self.min_tokens]

        return ' '.join(kept)


class IdentityDefense(DefenseWrapper):
    """No-op defense that passes input unchanged. Useful for baseline comparisons."""

    def defend_single(self, text: str) -> str:
        return text


def get_defense(
    defense_name: str,
    victim: OpenAttack.Classifier,
    param: float = 0.0,
    seed: Optional[int] = None
) -> OpenAttack.Classifier:
    """
    Factory function to create defense wrappers.

    Args:
        defense_name: Name of defense ('none', 'spellcheck', 'noise', 'dropout')
        victim: The victim classifier to wrap
        param: Defense-specific parameter (noise_std or dropout_prob)
        seed: Random seed for reproducibility

    Returns:
        Wrapped victim classifier with defense applied
    """
    defense_name = defense_name.lower()

    if defense_name == 'none' or defense_name == '':
        return victim
    elif defense_name == 'spellcheck':
        return SpellCheckDefense(victim)
    elif defense_name == 'noise':
        return EmbeddingNoiseDefense(victim, noise_std=param, seed=seed)
    elif defense_name == 'dropout':
        return TokenDropoutDefense(victim, dropout_prob=param, seed=seed)
    elif defense_name == 'identity':
        return IdentityDefense(victim)
    else:
        raise ValueError(f"Unknown defense: {defense_name}. "
                        f"Available: none, spellcheck, noise, dropout")
