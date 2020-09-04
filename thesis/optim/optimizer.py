from itertools import product
from dataclasses import dataclass
from typing import *
from tqdm import tqdm

import numpy as np
from torch.utils.data import Dataset
from typing_extensions import Protocol

from thesis.optim.loss import LossFunction
from thesis.optim.sampling import Strategy

from thesis.data import SegmentationDataset, GazeDataset

T = TypeVar('T')


@dataclass
class Experiment:
    filter: Any
    metrics: list
    iris_datasets: List[SegmentationDataset]
    gaze_datasets: List[GazeDataset]
    strategy: Generator[Dict[str, T], None, None]

    def run(self):
        pass

    def save(self):
        pass


def optimize(model, loss: LossFunction, strategy: Generator[Dict[str, T], None, None], dataset: Dataset,
             iterations: Optional[int] = None):
    avg_loss = []
    for i, hyper_params in enumerate(strategy):
        if iterations is not None and i >= iterations:
            break

        losses = []
        for img in tqdm(dataset,
                        desc=f'Epoch: {i:2d}',
                        postfix=hyper_params):
            output = model(img, **hyper_params)
            l = loss(p=output, t=img, **hyper_params)
            losses.append(l)
        avg_loss.append((hyper_params, np.mean(losses)))
        # print(avg_loss[-1])

    return avg_loss
