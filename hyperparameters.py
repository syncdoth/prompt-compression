from argparse import Namespace
from dataclasses import dataclass
import logging


@dataclass
class PromptHyperParameters:
    max_steps: int = 1
    epochs: int = 1
    lr: float = 1e-3
    num_warmup_steps: int = 0
    weight_decay: float = 0.01
    batch_size: int = 16
    eval_batch_size: int = 16
    max_seq_length: int = 32
    prompt_length: int = 16

    @classmethod
    def from_args(cls, args: Namespace):
        hp = cls()
        dict_repr = cls.__dict__
        for k, v in args._get_kwargs():
            if k not in dict_repr:
                continue
            hp.__setattr__(k, v)
        return hp

    @classmethod
    def from_dict(cls, kwargs: dict):
        hp = cls()
        dict_repr = cls.__dict__
        unused_args = []
        for k, v in kwargs.items():
            if k not in dict_repr:
                unused_args.append(k)
                continue
            hp.__setattr__(k, v)

        logging.warning(f'some of the args are not recognized by {hp.__class__}'
                        f' and were ignored. The ignored keys are: {unused_args}')
        return hp
