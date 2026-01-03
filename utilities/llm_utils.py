import os
import urllib.request
import zipfile
import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoModel, AutoConfig
import fasttext
from huggingface_hub import hf_hub_download

# Embedding type selection: 'glove' (~400MB, allows more training data) or 'fasttext' (~7GB, better quality)
EMBEDDING_TYPE = 'glove'

_fasttext_model_cache = None


def prepare_for_llm(observations, tokeniser):
    input_ids = []
    attention_mask = []
    token_type_ids = []
    for observation in observations:
        input_ids_here = [tokeniser.pad_token_id] * len(observation)
        attention_mask_here = [0] * len(observation)
        token_type_ids_here = [0] * len(observation)
        # Can be optimised to avoid the loop
        for i in range(len(observation)):
            if observation[i] != tokeniser.pad_token_id and observation[i] != tokeniser.sep_token_id:
                input_ids_here[i] = observation[i]
                attention_mask_here[i] = 1
            else:
                input_ids_here[i] = tokeniser.sep_token_id
                break
        input_ids.append(input_ids_here)
        attention_mask.append(attention_mask_here)
        token_type_ids.append(token_type_ids_here)
    return torch.tensor(input_ids), torch.tensor(attention_mask), torch.tensor(token_type_ids)


def get_best_candidates(orig_texts, orig_tokens, BEST_K, tokeniser, pretrained_model, device, protected_tokens,
                        with_masking=False):
    BATCH_SIZE = 16
    model = AutoModelForMaskedLM.from_pretrained(pretrained_model).to(device)
    candidates = np.zeros(orig_tokens.shape + (BEST_K,), dtype=int)
    for i, text in enumerate(orig_texts):
        if i % 50 == 0:
            print("Preparing candidates for text " + str(i))
        original = orig_tokens[i]
        variants = []
        if with_masking:
            for j in range(len(original)):
                if original[j] == tokeniser.pad_token_id:
                    break
                variant = original.copy()
                variant[j] = tokeniser.mask_token_id
                variants.append(variant)
        else:
            variants.append(original.copy())
        batches = [variants[i:i + BATCH_SIZE] for i in range(0, len(variants), BATCH_SIZE)]
        outputs = []
        for batch in batches:
            input = prepare_for_llm(batch, tokeniser)
            input = (x.to(device) for x in input)
            with torch.no_grad():
                output = model(*input)['logits']
            outputs.append(output)
        outputs = torch.cat(outputs).to(torch.device('cpu')).numpy()
        for j in range(len(original)):
            variant_idx = j if with_masking else 0
            if original[j] == tokeniser.pad_token_id:
                break
            if tokeniser.convert_ids_to_tokens([original[j]])[0] in protected_tokens:
                candidates[i][j] = [original[j]] * BEST_K
                continue
            candidates_here = []  # [tokeniser.unk_token_id][original[j]]
            outputs[variant_idx][j][original[j]] = -float('inf')
            outputs[variant_idx][j][tokeniser.pad_token_id] = -float('inf')
            outputs[variant_idx][j][tokeniser.sep_token_id] = -float('inf')
            outputs[variant_idx][j][tokeniser.unk_token_id] = -float('inf')
            outputs[variant_idx][j][tokeniser.cls_token_id] = -float('inf')
            for k in range(BEST_K):
                candidate = outputs[variant_idx][j].argmax(-1)
                candidates_here.append(candidate)
                outputs[variant_idx][j][candidate] = -float('inf')
            candidates[i][j] = candidates_here
            # print("SENTENCE: " + text)
            # print("REPLACE: " + str(self.tokeniser.convert_ids_to_tokens(
            #    [original[j]])) + " -> " + ' '.join(self.tokeniser.convert_ids_to_tokens(candidates_here)))
    return candidates


def get_candidate_embeddings_llm(orig_texts, orig_tokens, replacement_tokens, BEST_K, tokeniser, pretrained_model,
                                 device):
    BATCH_SIZE = 16
    model = AutoModel.from_pretrained(pretrained_model).to(device)
    hidden_size = AutoConfig.from_pretrained(pretrained_model).hidden_size
    embeddings = np.zeros(orig_tokens.shape + (BEST_K, hidden_size))
    for i, text in enumerate(orig_texts):
        if i % 50 == 0:
            print("Generating embeddings for candidates in text " + str(i))
        original = orig_tokens[i]
        variants = []
        for j in range(len(original)):
            if original[j] == tokeniser.pad_token_id:
                break
            for replacement in replacement_tokens[i][j]:
                variant = original.copy()
                variant[j] = replacement
                variants.append(variant)
        batches = [variants[i:i + BATCH_SIZE] for i in range(0, len(variants), BATCH_SIZE)]
        outputs = []
        for batch in batches:
            input = prepare_for_llm(batch, tokeniser)
            input = (x.to(device) for x in input)
            with torch.no_grad():
                output = model(*input)['last_hidden_state']
            outputs.append(output)
        outputs = torch.cat(outputs).to(torch.device('cpu')).numpy()
        counter = 0
        for j in range(len(original)):
            if original[j] == tokeniser.pad_token_id:
                break
            for k in range(BEST_K):
                embeddings[i][j][k] = outputs[counter][j]
                counter = counter + 1
    return embeddings


def load_fastText_vectors():
    """Load FastText vectors (~7GB download, ~7GB in memory)"""
    global _fasttext_model_cache
    if _fasttext_model_cache is not None:
        print("Using cached FastText model...")
        return _fasttext_model_cache
    model_path = hf_hub_download(repo_id="facebook/fasttext-en-vectors", filename='model.bin')
    model = fasttext.load_model(model_path)
    _fasttext_model_cache = model
    return model


def load_glove_vectors(glove_path=None):
    """Load GloVe 6B 300d vectors (~862MB download, ~400MB in memory)"""
    if glove_path is None:
        glove_dir = os.path.expanduser('~/.cache/glove')
        glove_path = os.path.join(glove_dir, 'glove.6B.300d.txt')

    if not os.path.exists(glove_path):
        print("Downloading GloVe vectors (862MB)...")
        os.makedirs(os.path.dirname(glove_path), exist_ok=True)
        zip_path = os.path.join(os.path.dirname(glove_path), 'glove.6B.zip')

        url = "https://nlp.stanford.edu/data/glove.6B.zip"
        urllib.request.urlretrieve(url, zip_path)

        print("Extracting GloVe vectors...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extract('glove.6B.300d.txt', os.path.dirname(glove_path))

        os.remove(zip_path)
        print("GloVe vectors ready.")

    print("Loading GloVe vectors into memory...")
    embeddings = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings[word] = vector

    print(f"Loaded {len(embeddings)} word vectors.")
    return embeddings


def get_candidate_embeddings_static(replacement_tokens, tokeniser):
    """Get embeddings for candidate replacement tokens using configured embedding type."""
    hidden_size = 300
    embeddings = np.zeros(replacement_tokens.shape + (hidden_size,), dtype=np.float16)

    if EMBEDDING_TYPE == 'fasttext':
        print("Reading static embedding dictionary (FastText)...")
        model = load_fastText_vectors()
        print("Obtaining candidate embeddings...")
        for i in range(replacement_tokens.shape[0]):
            for j in range(replacement_tokens.shape[1]):
                strings = tokeniser.batch_decode(replacement_tokens[i][j])
                for k, string in enumerate(strings):
                    normalised = string.replace('##', '')
                    embeddings[i, j, k] = model.get_word_vector(normalised)
    else:  # glove
        print("Reading static embedding dictionary (GloVe)...")
        emb_dict = load_glove_vectors()
        print("Obtaining candidate embeddings...")
        for i in range(replacement_tokens.shape[0]):
            for j in range(replacement_tokens.shape[1]):
                strings = tokeniser.batch_decode(replacement_tokens[i][j])
                for k, string in enumerate(strings):
                    normalised = string.replace('##', '').lower()  # GloVe is lowercase
                    if normalised in emb_dict:
                        embeddings[i, j, k] = emb_dict[normalised]

    return embeddings
