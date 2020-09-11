from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Callable, Iterator
from tqdm import tqdm

import numpy as np

from thesis.data import SegmentationDataset, GazeDataset, SegmentationSample, GazeImage
from thesis.information.entropy import gradient_histogram, histogram
from thesis.optim.sampling import Sampler
from thesis.optim.objective_terms import gradient_entropy, GazeAbsoluteAccuracy, AbsoluteEntropy


class MultiObjectiveOptimizer:
    @abstractmethod
    def run(self, *, wrapper=None):
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
        gradient_entropies = []
        intensity_entropies = []
        gaze = []

        grad_func = AbsoluteEntropy(gradient_histogram)
        intensity_func = AbsoluteEntropy(histogram)

        for dataset in self.iris_datasets:
            for sample in dataset.samples:
                output = self.filter(sample.image.image, **params)
                gradient_entropies.append(grad_func(sample, output))
                intensity_entropies.append(intensity_func(sample, output))

        for dataset in self.gaze_datasets:
            model = GazeAbsoluteAccuracy(dataset.model)
            for sample in dataset.test_samples:
                output = self.filter(sample.image, **params)
                gaze.append(model(sample, output))

        return {
            'gradient_entropy': np.mean(gradient_entropies),
            'gaze': np.mean(gaze)
            # 'intensity_entropy': np.mean(intensity_entropies)
        }

    def output_dimensions(self):
        return 2


def dominates(y, y_mark):
    y = np.array(y)
    y_mark = np.array(y_mark)
    return np.all(y <= y_mark) and np.any(y < y_mark)


@dataclass
class PopulationMultiObjectiveOptimizer(MultiObjectiveOptimizer, ABC):
    ...


@dataclass
class NaiveMultiObjectiveOptimizer(MultiObjectiveOptimizer):
    def __init__(self, objective: Objective, sampler: Sampler):
        self.objective = objective
        self.sampler = sampler

        self.results = []

    def run(self, *, wrapper=None):
        if wrapper is not None:
            iterator = wrapper(self.sampler, len(self.sampler))
        else:
            iterator = self.sampler

        for params in iterator:
            output = self.objective.eval(params)
            self.results.append((params, output))

    def metrics(self):
        return self.results

    def pareto_frontier(self):
        pareto = []
        for i, (params, output) in enumerate(self.results):
            if not any([dominates(tuple(output_mark.values()), tuple(output.values())) for _, output_mark in
                        self.results]):
                # pareto.append((params, output))
                pareto.append(i)

        # d = len(self.results)
        # domination_matrix = np.zeros((d, d))
        # for i, (params, output) in enumerate(self.results):
        #     for j, (params_mark, output_mark) in enumerate(self.results):
        #         if dominates(tuple(output.values()), tuple(output_mark.values())):
        #             domination_matrix[i, j] = 1
        # print(domination_matrix)
        return pareto
