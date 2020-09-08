from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Callable, Iterator
from tqdm import tqdm

import numpy as np

from thesis.data import SegmentationDataset, GazeDataset, SegmentationSample, GazeImage
from thesis.tracking.gaze import GazeModel
from thesis.optim.objective_terms import gradient_entropy, GazeAbsoluteAccuracy


class MultiObjectiveOptimizer:
    @abstractmethod
    def run(self, *, status=False) -> Iterator:
        ...

    @abstractmethod
    def metrics(self):
        ...

    @abstractmethod
    def pareto_frontier(self):
        ...


@dataclass
class Term:
    model: None

    def __call__(self, data_point, filtered_image):
        ...


class Objective(ABC):

    @abstractmethod
    def eval(self, params):
        ...

    @abstractmethod
    def output_dimensions(self):
        ...


@dataclass
class ObfuscationObjective(Objective):
    filter: Callable
    iris_datasets: List[SegmentationDataset]
    gaze_datasets: List[GazeDataset]

    def eval(self, params):
        entropy = []
        gaze = []

        for dataset in self.iris_datasets:
            for sample in dataset.samples:
                output = self.filter(sample.image.image, **params)
                entropy.append(gradient_entropy(output, sample.image.image))

        for dataset in self.gaze_datasets:
            model = GazeAbsoluteAccuracy(dataset.model)
            for sample in dataset.test_samples:
                output = self.filter(sample.image, **params)
                gaze.append(model(sample, output))

        return np.mean(entropy), np.mean(gaze)

    def output_dimensions(self):
        return 2


def dominates(y, y_mark):
    return np.all(y <= y_mark) and np.any(y < y_mark)


@dataclass
class NaiveMultiObjectiveOptimizer(MultiObjectiveOptimizer):
    def __init__(self, objective: Objective, sampler: Iterator):
        self.objective = objective
        self.sampler = sampler

        self.results = []

    def run(self, *, status=False):
        if status:
            iterator = tqdm(self.sampler)
        else:
            iterator = self.sampler

        for params in iterator:
            output = self.objective.eval(params)
            self.results.append((params, output))

    def metrics(self):
        return np.array(self.results)

    def pareto_frontier(self):
        pareto = []
        for params, output in self.results:
            if not any([dominates(output, output_mark) for _, output_mark in self.results]):
                pareto.append((params, output))
        return pareto
