from argparse import Namespace

import torch
from transformers import get_linear_schedule_with_warmup

from prompt_compression import prompt_compress_loss
from hyperparameters import HyperParameters


def train_context_prompt(model,
                         prompt_embed,
                         dataloader,
                         hp: HyperParameters,
                         args: Namespace,
                         device='cpu',
                         is_encoder_decoder=False):

    optimizer = torch.optim.Adam([prompt_embed], lr=hp.lr)
    num_training_steps = min(hp.epochs * len(dataloader), hp.max_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, hp.num_warmup_steps, num_training_steps)

    gs = 0  # global step
    for epoch in range(hp.epochs):
        for i, batch in enumerate(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            bsize = batch['left_context_ids'].shape[0]
            batch_embed = prompt_embed.repeat(bsize, 1)

            loss = prompt_compress_loss(batch_embed,
                                        model,
                                        batch['target_ids'],
                                        batch['right_context_ids'],
                                        left_context_ids=batch['left_context_ids'],
                                        left_context_emb=batch['left_context_emb'],
                                        device=device,
                                        is_encoder_decoder=is_encoder_decoder)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            gs += 1

            if gs >= hp.max_steps:
                return prompt_embed

    return prompt_embed


def online_dialog_compress(model,
                           prompt_embed: torch.Tensor,
                           hp: HyperParameters,
                           args: Namespace,
                           target_ids=None,
                           right_context_ids=None,
                           left_context_emb=None,
                           device: str = 'cpu',
                           is_encoder_decoder: bool = False):
    optimizer = torch.optim.Adam([prompt_embed], lr=hp.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, hp.num_warmup_steps, hp.max_steps)

    for step in range(hp.max_steps):
        loss = prompt_compress_loss(prompt_embed,
                                    model,
                                    target_ids,
                                    right_context_ids,
                                    left_context_emb=left_context_emb,
                                    device=device,
                                    is_encoder_decoder=is_encoder_decoder)
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return prompt_embed
