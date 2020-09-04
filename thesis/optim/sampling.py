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
        nums = nums + (np.random.random(len(nums))*step - step*0.5)
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
    step = 1 if num == 0 else (stop-start)/num
    if stratified:
        nums = nums + (np.random.random(len(nums)) * step - step * 0.5)
    return nums


class Strategy(Protocol):
    def __call__(
            self, params: List[str],
            generators: List[Any]) -> Generator[Dict[str, T], None, None]:
        """

        Args:
            params: Parameter names
            generators: Generator list

        Yields:
            dict: Dictionary of parameter/value pairs.
        """
        ...


def grid_search(params: List[str], generators: List[np.ndarray]):
    """Perform a search over the cartesian product of the generator values. The number of search values
    rise exponentially with the number of generators.
    """
    for values in product(*generators):
        yield dict(zip(params, values))


def uniform_sampler(params: List[str], generators: List[np.ndarray]):
    """Generate a uniform sampling for searching. The number of search values is equal to the length of
    the longest generator.
    """
    for gen in generators:
        np.random.shuffle(gen)
    max_len = len(max(generators, key=len))
    generators = [cycle(gen) if len(gen) < max_len else gen for gen in generators]
    for values in zip(*generators):
        yield dict(zip(params, values))


def random_generator(params, generators):
    while True:
        yield {param: gen() for param, gen in zip(params, generators)}
