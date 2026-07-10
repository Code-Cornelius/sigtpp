import typing
from enum import Enum

import numpy as np

from src.generators import hp


class NoiseGenType(Enum):
    # Keep the enum name in uppercase, and the generator name in lowercase. The only tricky part is below with `upper`.
    NORMAL = (
        'normal',
        lambda batch_size, length_seqs: np.random.normal(0.0, 1.0, (batch_size, length_seqs, 1)),
    )
    NORMAL_LARGE_VAR = (
        'normal_large_var',
        lambda batch_size, length_seqs: np.random.normal(0.0, 2.5, (batch_size, length_seqs, 1)),
    )
    EXP = ('exp', lambda batch_size, length_seqs: hp.gen(batch_size, 1.0, None, length_seqs))
    EXP_CTR = ('exp_ctr', lambda batch_size, length_seqs: hp.gen(batch_size, 1.0, None, length_seqs) - 1.0)
    EXP_TWO = ('exp_two', lambda batch_size, length_seqs: hp.gen(batch_size, 2.0, None, length_seqs))
    LOG_EXP = (
        'log_exp',
        lambda batch_size, length_seqs: (np.log(hp.gen(batch_size, 1.0, None, length_seqs)) + 0.5798) / 1.29,
    )
    LOG_EXP_SPREAD = (
        'log_exp_spread',
        lambda batch_size, length_seqs: (np.log(hp.gen(batch_size, 1.0, None, length_seqs)) + 0.5798) * 2.0,
    )
    LOG_EXP_NOT_STD = (
        'log_exp_not_std',
        lambda batch_size, length_seqs: np.log(hp.gen(batch_size, 1.0, None, length_seqs)),
    )

    @classmethod
    def list_all_names(cls):
        return [member._name for member in cls]

    def __init__(self, name: str, generator: typing.Callable[[int, int], np.ndarray]):
        self._name = name
        self._generator = generator

    def generate(self, batch_size: int, length_seqs: int) -> np.ndarray:
        return self._generator(batch_size, length_seqs)


def get_noise_gen(noise_gen_name: str):
    try:
        noise_gen_type = NoiseGenType[noise_gen_name.upper()]
        return noise_gen_type.generate
    except KeyError:
        raise NotImplementedError(f"Requested noise generator {noise_gen_name} not implemented.")
