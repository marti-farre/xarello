import random
from abc import ABC, abstractmethod
from typing import List, Callable, Optional

import numpy as np
import OpenAttack

# SEPARATOR used for text pairs in FC/C19 tasks (from BODEGA)
SEPARATOR = ' ~ '


class DefenseWrapper(OpenAttack.Classifier, ABC):
    def __init__(self, victim: OpenAttack.Classifier, verbose: bool = False):
        self.victim = victim
        self.verbose = verbose
        self.modifications = []

    def get_pred(self, input_: List[str]) -> np.ndarray:
        return self.get_prob(input_).argmax(axis=1)

    def get_prob(self, input_: List[str]) -> np.ndarray:
        defended_input = self.apply_defense(input_)
        return self.victim.get_prob(defended_input)

    def apply_defense(self, input_: List[str]) -> List[str]:
        result = []
        for text in input_:
            if SEPARATOR in text:
                # Handle text pairs (FC/C19 tasks)
                parts = text.split(SEPARATOR)
                defended_parts = [self.defend_single(p) for p in parts]
                defended_text = SEPARATOR.join(defended_parts)
            else:
                defended_text = self.defend_single(text)

            # Track and log modifications
            if text != defended_text:
                self.modifications.append((text, defended_text))
                if self.verbose:
                    print(f"\n[DEFENSE] Text modified:")
                    print(f"  Original: {text[:100]}{'...' if len(text) > 100 else ''}")
                    print(f"  Defended: {defended_text[:100]}{'...' if len(defended_text) > 100 else ''}")

            result.append(defended_text)
        return result

    def get_modifications(self) -> List[tuple]:
        return self.modifications

    def clear_modifications(self):
        self.modifications = []

    def save_modifications(self, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            f.write("original\tdefended\n")
            for orig, defended in self.modifications:
                # Escape tabs and newlines for TSV format
                orig_escaped = orig.replace('\t', '\\t').replace('\n', '\\n')
                defended_escaped = defended.replace('\t', '\\t').replace('\n', '\\n')
                f.write(f"{orig_escaped}\t{defended_escaped}\n")

    @abstractmethod
    def defend_single(self, text: str) -> str:
        pass

    def finalise(self):
        if hasattr(self.victim, 'finalise'):
            self.victim.finalise()


class SpellCheckDefense(DefenseWrapper):
    """
    Defense that corrects spelling errors to counter character-level attacks.

    XARELLO uses character swaps, insertions, and deletions which often
    result in misspellings. This defense attempts to correct them.
    """

    def __init__(self, victim: OpenAttack.Classifier, language: str = 'en', verbose: bool = False):
        super().__init__(victim, verbose)
        self.language = language
        self._spellchecker = None

    @property
    def spellchecker(self):
        if self._spellchecker is None:
            try:
                from symspellpy import SymSpell
                import importlib.resources
            except ImportError:
                raise ImportError(
                    "SpellCheckDefense requires symspellpy. "
                    "Install with: pip install symspellpy"
                )

            self._spellchecker = SymSpell(
                max_dictionary_edit_distance=2,
                prefix_length=7
            )

            # Load built-in English dictionary
            dictionary_path = importlib.resources.files("symspellpy") / "frequency_dictionary_en_82_765.txt"
            if not self._spellchecker.load_dictionary(dictionary_path, term_index=0, count_index=1):
                raise RuntimeError("Failed to load symspellpy English dictionary")
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
                from symspellpy import Verbosity
                suggestions = self.spellchecker.lookup(core.lower(), Verbosity.TOP, max_edit_distance=2)
                correction = suggestions[0].term if suggestions else None
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
        seed: Optional[int] = None,
        verbose: bool = False
    ):
        super().__init__(victim, verbose)
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
        min_tokens: int = 3,
        verbose: bool = False
    ):
        super().__init__(victim, verbose)
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

    def __init__(self, victim: OpenAttack.Classifier, verbose: bool = False):
        super().__init__(victim, verbose)

    def defend_single(self, text: str) -> str:
        return text


class MajorityVoteDefense(DefenseWrapper):
    """
    Defense that creates perturbed copies of input and uses majority voting.

    For each input text:
    1. Create N perturbed copies using random character-level perturbations
    2. Classify each copy with the victim model
    3. Return the majority vote as the predicted class

    get_prob() returns NOISY vote-fraction probabilities (stochastic oracle).
    This confuses gradient-based word-level attacks that rely on consistent
    probability signals to find effective substitutions.

    Ported from BODEGA defenses/preprocessing.py (experiment-3/combine-voting-check).
    """

    def __init__(
        self,
        victim: OpenAttack.Classifier,
        num_copies: int = 7,
        perturbation_prob: float = 0.1,
        aggregation: str = 'hard',
        seed: Optional[int] = None,
        verbose: bool = False
    ):
        super().__init__(victim, verbose)
        self.num_copies = num_copies
        self.perturbation_prob = perturbation_prob
        self.aggregation = aggregation
        self.rng = random.Random(seed)

    def get_prob(self, input_: List[str]) -> np.ndarray:
        """
        Return vote-fraction probabilities from N perturbed copies (noisy oracle).

        The stochastic oracle is intentional: each call returns different probabilities
        because each call re-runs N fresh random perturbations. This confuses gradient-based
        word-level attacks. OpenAttack "Check attacker result failed" errors are expected
        and harmless — they indicate samples where the stochastic oracle blocked the attack.
        """
        all_probs = []
        for _ in range(self.num_copies):
            perturbed_input = self.apply_defense(input_)
            probs = self.victim.get_prob(perturbed_input)
            all_probs.append(probs)

        all_probs = np.stack(all_probs, axis=0)  # (num_copies, batch_size, num_classes)

        if self.aggregation == 'hard':
            predictions = all_probs.argmax(axis=2)  # (num_copies, batch_size)
            batch_size = predictions.shape[1]
            num_classes = all_probs.shape[2]
            result = np.zeros((batch_size, num_classes))
            for sample_idx in range(batch_size):
                votes = predictions[:, sample_idx]
                for class_idx in range(num_classes):
                    result[sample_idx, class_idx] = np.sum(votes == class_idx) / self.num_copies
            return result
        else:  # soft voting
            return np.mean(all_probs, axis=0)

    def defend_single(self, text: str) -> str:
        """Apply random character-level perturbation (delete/swap/insert)."""
        if self.perturbation_prob == 0.0:
            return text

        result = []
        i = 0
        while i < len(text):
            char = text[i]
            if char.isspace():
                result.append(char)
                i += 1
                continue
            if self.rng.random() < self.perturbation_prob:
                perturbation = self.rng.choice(['delete', 'swap', 'insert'])
                if perturbation == 'delete':
                    i += 1
                    continue
                elif perturbation == 'swap' and i < len(text) - 1 and not text[i + 1].isspace():
                    result.append(text[i + 1])
                    result.append(char)
                    i += 2
                    continue
                elif perturbation == 'insert':
                    random_char = chr(self.rng.randint(ord('a'), ord('z')))
                    result.append(random_char)
                    result.append(char)
                    i += 1
                    continue
            result.append(char)
            i += 1

        result_str = ''.join(result)
        return result_str if result_str.strip() else text


class SpellCheckMVDefense(MajorityVoteDefense):
    """
    Combined defense: SpellCheck first, then MajorityVote.

    Pipeline: adversarial input → SpellCheck → N noisy MV copies → victim × N → majority vote

    - SpellCheck handles character-level attacks (XARELLO char swaps, inserts, deletes)
    - MajorityVote handles word-level attacks by confusing the RL agent's reward signal

    Ported from BODEGA defenses/preprocessing.py (experiment-3/combine-voting-check).
    """

    def __init__(
        self,
        victim: OpenAttack.Classifier,
        num_copies: int = 7,
        seed: Optional[int] = None,
        verbose: bool = False
    ):
        super().__init__(victim, num_copies=num_copies, seed=seed, verbose=verbose)
        self._spellcheck = SpellCheckDefense(victim, verbose=verbose)

    def get_prob(self, input_: List[str]) -> np.ndarray:
        """SpellCheck input, then return noisy MV vote-fraction probabilities."""
        spellchecked = self._spellcheck.apply_defense(input_)
        return super().get_prob(spellchecked)


def get_defense(
    defense_name: str,
    victim: OpenAttack.Classifier,
    param: float = 0.0,
    seed: Optional[int] = None,
    verbose: bool = False
) -> OpenAttack.Classifier:
    defense_name = defense_name.lower()

    if defense_name == 'none' or defense_name == '':
        return victim
    elif defense_name == 'spellcheck':
        return SpellCheckDefense(victim, verbose=verbose)
    elif defense_name == 'noise':
        return EmbeddingNoiseDefense(victim, noise_std=param, seed=seed, verbose=verbose)
    elif defense_name == 'dropout':
        return TokenDropoutDefense(victim, dropout_prob=param, seed=seed, verbose=verbose)
    elif defense_name == 'identity':
        return IdentityDefense(victim, verbose=verbose)
    elif defense_name == 'majority_vote' or defense_name == 'vote':
        num_copies = int(param) if param > 0 else 7
        return MajorityVoteDefense(victim, num_copies=num_copies, seed=seed, verbose=verbose)
    elif defense_name == 'spellcheck_mv' or defense_name == 'sc_mv':
        num_copies = int(param) if param > 0 else 7
        return SpellCheckMVDefense(victim, num_copies=num_copies, seed=seed, verbose=verbose)
    else:
        raise ValueError(f"Unknown defense: {defense_name}. "
                        f"Available: none, spellcheck, noise, dropout, identity, "
                        f"majority_vote, spellcheck_mv")
