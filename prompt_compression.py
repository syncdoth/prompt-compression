from typing import Any

import torch
import torch.nn.functional as F

from transformers import PretrainedModel


def prompt_compress_loss(prompt_embed: torch.Tensor,
                         model: PretrainedModel,
                         target_ids: torch.LongTensor,
                         right_context_ids: torch.LongTensor,
                         left_context_ids: torch.LongTensor = None,
                         left_context_emb: torch.Tensor = None,
                         device: Any[str, torch.device] = 'cpu',
                         is_encoder_decoder: bool = False):
    """
    left_context_ids / target_ids / prompt_embed: should not have eos
    right_context_ids: should have eos

    prompt_embed: the soft prompt to tune.
    target_ids: the "target" text you want to compress into prompt.
    left_context_ids: the context that comes before the "target".
    left_context_emb: the embedding of left context. The left context might also
        be some compressed prompts / tuned prompts / or just embeddings.
    right_context_ids: the context that comes after the "target".
    """

    assert left_context_ids.shape[0] == right_context_ids.shape[0] == prompt_embed.shape[0]

    if left_context_ids is not None:
        assert left_context_emb is None
        left_context_emb = model.get_input_embeddings()(left_context_ids).detach()
        # for prompt embed
        input_embed = torch.cat([left_context_emb, prompt_embed], dim=1)
        # for original
        target_embs = model.get_input_embeddings()(target_ids).detach()
        original_input_emb = torch.cat([left_context_emb, target_embs], dim=1)

    if left_context_emb is not None:
        assert left_context_ids is None
        # for prompt embed
        input_embed = torch.cat([left_context_emb, prompt_embed], dim=1)
        # for original
        target_embs = model.get_input_embeddings()(target_ids).detach()
        original_input_emb = torch.cat([left_context_emb, target_embs], dim=1)

    if is_encoder_decoder:
        # NOTE: if T5, the final token of input ids should be eos.
        # TODO: this might not be true for other encoder-decoders.
        _, eos_token_embed = get_eos_token_embed(model, device=device)
        input_embed = torch.cat([input_embed, eos_token_embed], dim=1)
        original_input_emb = torch.cat([original_input_emb, eos_token_embed], dim=1)

        decoder_input_ids = model._shift_right(right_context_ids)
        prompt_logits = model(inputs_embeds=input_embed, decoder_input_ids=decoder_input_ids).logits
        original_logits = model(inputs_embeds=original_input_emb,
                                decoder_input_ids=decoder_input_ids).logits
    else:
        right_context_emb = model.get_input_embeddings()(right_context_ids).detach()
        all_input_embed = torch.cat([input_embed, right_context_emb], dim=1)
        prompt_logits = model(inputs_embeds=all_input_embed).logits

        target_input_emb = torch.cat([original_input_emb, right_context_emb], dim=1)
        original_logits = model(inputs_embeds=target_input_emb).logits

        # for non-encoder-decoder model, need to trim the logits to match the length;
        # only the right context part.
        p_length = input_embed.shape[1]  # left context + prompt length
        prompt_logits = prompt_logits[:, p_length - 1:, :]

        original_p_length = original_input_emb.shape[1]  # left context + target context
        original_logits = original_logits[:, original_p_length - 1:, :]

    loss = F.kl_div(prompt_logits, original_logits)
    return loss


def init_prompt(prompt_length, embed_size):
    # TODO: better init
    return torch.rand([1, prompt_length, embed_size])


def get_eos_token_embed(model, device='cpu'):
    eos_token_ids = torch.LongTensor([[model.config.eos_token_id]]).to(device)
    eos_embed = model.get_input_embeddings()(eos_token_ids).detach()
    return eos_token_ids, eos_embed
