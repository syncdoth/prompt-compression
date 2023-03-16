"""
The main file to run experiments.
"""
import logging
import os
import time

import torch
from model import load_transformer_LM_tokenizer
import wandb

from data import get_dataloaders
from prompt_compression import init_prompt
from train import train_context_prompt
from utils import set_random_seeds
from hyperparameters import PromptHyperParameters

import fire


def main(
        model_name_or_path='roberta-base',
        wandb_project='project',
        checkpoint_dir='ckpt',
        wandb_runname=None,
        seed=100,
        **hyperparameters,  # see hyperparameters.py
):
    """
    For the arguments for hyperparameters, check the hyperparameters.py
    """
    hp = PromptHyperParameters(**hyperparameters)
    set_random_seeds(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    base_path = os.path.dirname(checkpoint_dir)
    os.makedirs(base_path, exist_ok=True)

    logging.basicConfig(handlers=[
        logging.FileHandler(os.path.join(base_path, 'train_log.log'), mode='a'),
        logging.StreamHandler(),
    ],
                        format='%(asctime)s:%(msecs)d|%(name)s|%(levelname)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)
    logging.info('Start Training!')

    # load model, tokenizer
    model, tokenizer = load_transformer_LM_tokenizer(model_name_or_path)
    model = model.to(device)

    # load data
    # TODO
    data_kwargs = {'tokenizer': tokenizer}
    dataloaders = get_dataloaders(batch_size=hp.batch_size,
                                  eval_batch_size=hp.eval_batch_size,
                                  **data_kwargs)

    if not wandb_runname:
        wandb_runname = str(round(time.time() * 1000))
    experiment = wandb.init(project=wandb_project, name=wandb_runname, config=hp)

    prompt_embed = init_prompt(hp.prompt_length, model.config.hidden_size)
    train_context_prompt(
        prompt_embed,
        model,
        dataloaders['train'],
        hp,
        device=device,
        wandb_experiment=experiment,
        is_encoder_decoder=model.config.is_encoder_decoder,
    )


if __name__ == '__main__':
    fire.Fire(main)
