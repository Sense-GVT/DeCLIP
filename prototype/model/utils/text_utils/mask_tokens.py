import torch
from typing import Tuple, List


def mask_tokens(inputs, special_tokens, mask_token, tokenizer_length, mlm_probability=0.15, special_tokens_mask=None) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, mlm_probability)
    if special_tokens_mask is None:
        special_tokens_mask = [1 if val in special_tokens else 0 for val in labels.tolist()]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    # if tokenizer._pad_token is not None:
    #     padding_mask = labels.eq(tokenizer.pad_token_id)
    #     probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = mask_token

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(tokenizer_length, labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


def MaskTokens(tokens, mask_type, mask_token, special_tokens=None, tokenizer_length=None, sepcial_tokens_mask=None, special_tokens_mask=None):
    if mask_type == 'MLM':
        tokens, labels = mask_tokens(inputs=tokens, special_tokens=special_tokens, mask_token=mask_token, tokenizer_length=tokenizer_length, special_tokens_mask=special_tokens_mask)
    else:
        raise NotImplementedError(mask_type)
    return tokens, labels
