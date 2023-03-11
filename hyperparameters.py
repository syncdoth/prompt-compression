from argparse import Namespace
from dataclasses import dataclass


@dataclass
class HyperParameters:
    max_steps: int = 1
    epochs: int = 1
    lr: float = 0.1
    num_warmup_steps: int = 0

    @classmethod
    def from_args(cls, args: Namespace):
        hp = cls()
        dict_repr = cls.__dict__
        for k, v in args._get_kwargs():
            if k not in dict_repr:
                continue
            hp.__setattr__(k, v)
        return hp
