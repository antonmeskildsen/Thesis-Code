from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import reduce

from itertools import product
from typing import *
from typing_extensions import Protocol
import random
from itertools import cycle

import numpy as np

T = TypeVar('T')


def samples_step(start, stop, step=1, *, stratified=True, clip=True):
    """Sample with step intervals. It is functionally comparable to np.arange.

    Args:
        start:
        stop:
        step:
        stratified:
        clip:

    Returns:

    """
    nums = np.arange(start, stop, step)
    if stratified:
        nums = nums + (np.random.random(len(nums)) * step - step * 0.5)
    return nums


def samples_num(start, stop, num, *, stratified=True, clip=True):
    """Create sample of specific size. It is functionally comparable to np.linspace.

    Args:
        start:
        stop:
        num:
        stratified:
        clip:

    Returns:

    """
    nums = np.linspace(start, stop, num)
    step = 1 if num == 0 else (stop - start) / num
    if stratified:
        nums = nums + (np.random.random(len(nums)) * step - step * 0.5)
    return nums.clip(start, stop)


@dataclass
class Sampler(ABC):
    @abstractmethod
    def __len__(self):
        ...

    @abstractmethod
    def __iter__(self):
        ...

    params: List[str]
    generators: List[np.ndarray]


class GridSearch(Sampler):
    """Perform a search over the cartesian product of the generator values. The number of search values
    rise exponentially with the number of generators.
    """

    def __len__(self):
        return int(reduce(lambda a, b: a*b, map(len, self.generators)))

    def __iter__(self):
        for values in product(*self.generators):
            yield dict(zip(self.params, values))


class UniformSampler(Sampler):
    """Generate a uniform sampling for searching. The number of search values is equal to the length of
    the longest generator.
    """

    def __len__(self):
        return max(map(len, self.generators))

    def __iter__(self):
        for gen in self.generators:
            np.random.shuffle(gen)
        max_len = len(max(self.generators, key=len))
        generators = [cycle(gen) if len(gen) < max_len else gen for gen in self.generators]
        for values in zip(*generators):
            yield dict(zip(self.params, values))

# class RandomSampler(Strategy):
#
#     def __len__(self):
#         pass
#
#     def __iter__(self):
#         while True:
#             yield {param: gen for param, gen in zip(self.params, self.generators)}
