import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoModel, AutoConfig
import fasttext
from huggingface_hub import hf_hub_download


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
    model_path = hf_hub_download(repo_id="facebook/fasttext-en-vectors", filename='model.bin')
    model = fasttext.load_model(model_path)
    return model


def get_candidate_embeddings_static(replacement_tokens, tokeniser):
    hidden_size = 300
    embeddings = np.zeros(replacement_tokens.shape + (hidden_size,))
    print("Reading static embedding dictionary (FastText)...")
    model = load_fastText_vectors()
    print("Obtaining candidate embeddings...")
    for i in range(replacement_tokens.shape[0]):
        for j in range(replacement_tokens.shape[1]):
            strings = tokeniser.batch_decode(replacement_tokens[i][j])
            for k, string in enumerate(strings):
                normalised = string.replace('##', '')
                embeddings[i, j, k] = model.get_word_vector(normalised)
    return embeddings
