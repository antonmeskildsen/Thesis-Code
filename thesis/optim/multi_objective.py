from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Callable, Optional, Type
import random
from itertools import product, repeat

import numpy as np
import cv2 as cv

import eyeinfo

from thesis.segmentation import IrisImage
from thesis.data import SegmentationDataset, GazeDataset, PupilDataset
from thesis.optim.pareto import dominates
from thesis.optim.sampling import Sampler, PopulationInitializer
from thesis.optim.objective_terms import GazeTerm, SegmentationTerm, PupilTerm, GradientHistogramTerm
from thesis.optim.population import SelectionMethod, MutationMethod, CrossoverMethod
from thesis.entropy import joint_gradient_histogram, entropy, mutual_information_grad, joint_gabor_histogram
from thesis.optim.status import ProgressBar

from multiprocessing import Pool, Queue, Lock


class FastReadCounter:
    def __init__(self):
        self.value = 0
        self._lock = Lock()

    def increment(self):
        with self._lock:
            self.value += 1


class Objective(ABC):

    @abstractmethod
    def metrics(self) -> List[str]:
        ...

    @abstractmethod
    def eval(self, params, parallel=False) -> List[float]:
        ...

    @abstractmethod
    def output_dimensions(self):
        ...


@dataclass
class ObfuscationObjective(Objective):
    filter: Callable
    iris_datasets: List[SegmentationDataset]
    gaze_datasets: List[GazeDataset]
    pupil_datasets: List[PupilDataset]

    iris_terms: List[SegmentationTerm]
    histogram_terms: List[GradientHistogramTerm]
    gabor_terms: List[GaborTerm]
    gaze_terms: List[GazeTerm]
    pupil_terms: List[PupilTerm]

    iris_samples: int
    gaze_samples: int
    pupil_samples: int

    def metrics(self) -> List[str]:
        return list(map(lambda x: type(x).__name__, self.iris_terms)) + \
               list(map(lambda x: type(x).__name__, self.histogram_terms)) + \
               list(map(lambda x: type(x).__name__, self.gaze_terms)) + \
               list(map(lambda x: type(x).__name__, self.pupil_terms))

    # @profile
    def eval(self, params):
        iris_results = [[] for _ in range(len(self.iris_terms))]
        histogram_results = [[] for _ in range(len(self.histogram_terms))]
        gaze_results = [[] for _ in range(len(self.gaze_terms))]
        pupil_results = [[] for _ in range(len(self.pupil_terms))]

        samples_per_set = self.iris_samples // len(self.iris_datasets)
        for dataset in self.iris_datasets:
            for sample in random.sample(dataset.samples, samples_per_set):
                output = self.filter(sample.image.image, **params)
                for i, term in enumerate(self.iris_terms):
                    iris_results[i].append(term(sample, output))

                angular, radial = 200, 20  # Should be: 1000, 100

                polar, mask = sample.image.to_polar(angular, radial)
                ii = IrisImage(sample.image.segmentation, output)
                polar_filtered, _ = ii.to_polar(angular, radial)

                divisions = 16

                pyramid = [(polar, polar_filtered, mask)]
                scales = 6
                num_angles = 6
                angles = np.linspace(0, np.pi, num_angles)
                for _ in range(scales):
                    for theta in angles:
                        hist_source, hist_filtered, hist_joint = joint_gabor_histogram(polar, polar_filtered, mask, theta, divisions)
                    polar = cv.pyrDown(polar)
                    polar_filtered = cv.pyrDown(polar_filtered)
                    mask = cv.resize(mask, polar.shape, interpolation=cv.INTER_NEAREST)
                    pyramid.append((polar, polar_filtered, mask))





                hist_source, hist_filtered, hist_joint = joint_gradient_histogram(polar, polar_filtered, mask,
                                                                                  divisions)
                entropy_source = entropy(hist_source)
                entropy_filtered = entropy(hist_filtered)
                mutual_information = mutual_information_grad(hist_source, hist_filtered, hist_joint)

                for i, term in enumerate(self.histogram_terms):
                    histogram_results[i].append(term(entropy_source, entropy_filtered, mutual_information))

        samples_per_set = self.gaze_samples // len(self.gaze_datasets)
        for dataset in self.gaze_datasets:
            for sample in random.sample(dataset.test_samples, samples_per_set):
                output = self.filter(sample.image, **params)
                for i, term in enumerate(self.gaze_terms):
                    gaze_results[i].append(term(dataset.model, sample, output))

        samples_per_set = self.pupil_samples // len(self.pupil_datasets)
        for dataset in self.pupil_datasets:
            for sample in random.sample(dataset.samples, samples_per_set):
                output = self.filter(sample.image, **params)
                for i, term in enumerate(self.pupil_terms):
                    pupil_results[i].append(term(sample, output))

        return list(map(np.mean, iris_results)) + list(map(np.mean, histogram_results)) + list(
            map(np.mean, gaze_results)) + list(map(np.mean, pupil_results))

    def output_dimensions(self):
        return len(self.iris_terms) + len(self.gaze_terms)


def id_wrap(iterator, total):
    return iterator


@dataclass
class MultiObjectiveOptimizer:
    results: List
    objective: Objective

    @abstractmethod
    def run(self, wrapper=id_wrap, parallel=False):
        ...

    def metrics(self):
        return self.results

    def pareto_frontier(self, k=0):
        pareto = []
        for i, (params, output, _) in enumerate(self.results):
            if not any([dominates(tuple(output_mark.values()), tuple(output.values())) for _, output_mark, km in
                        self.results if km == k]):
                pareto.append(i)
        return pareto


@dataclass
class PopulationMultiObjectiveOptimizer(MultiObjectiveOptimizer):
    selection_method: SelectionMethod
    crossover_method: CrossoverMethod
    mutation_method: MutationMethod
    iterations: int
    initial_population: PopulationInitializer

    def run(self, wrapper=id_wrap, parallel=False):
        pop = list(self.initial_population)
        params = list(pop[0].keys())

        m = self.objective.output_dimensions()
        m_pop = len(pop)
        m_sub_pop = m_pop // m + 1  # TODO: Check for correct solution (is it important to get len(parents)==m_pop?

        iterator = range(self.iterations)

        self.results = []
        for k in wrapper(iterator, self.iterations):
            ys = [self.objective.eval(x, parallel) for x in pop]
            self.results.extend(list(zip(pop, [dict(zip(self.objective.metrics(), y)) for y in ys], [k] * len(pop))))

            parents = []
            for i in range(m):
                selected = self.selection_method.select([y[i] for y in ys])
                parents.extend(selected[:m_sub_pop])

            p = np.random.choice(m_pop * 2, m_pop * 2, False)

            def p_ind(idx):
                return parents[p[idx] % m_pop][p[idx] // m_pop]

            parents = [(p_ind(i), p_ind(i + 1)) for i in range(0, 2 * m_pop, 2)]
            pop_values = [list(p.values()) for p in pop]
            children = [self.crossover_method.crossover(pop_values[p[0]], pop_values[p[1]]) for p in parents]
            pop_values = [self.mutation_method.mutate(c) for c in children]
            pop_values = [np.clip(v, 0, None) for v in pop_values]
            pop = [dict(zip(params, p)) for p in pop_values]


def for_each(args):
    params, objective = args
    output = objective.eval(params)
    return params, dict(zip(objective.metrics(), output)), 0


from tqdm import tqdm


@dataclass
class NaiveMultiObjectiveOptimizer(MultiObjectiveOptimizer):
    sampler: Sampler

    def run(self, wrapper=id, parallel=False):
        # bar = bar(len(self.sampler))
        self.results = []

        args = zip(self.sampler, repeat(self.objective))

        if parallel:
            pool = Pool(processes=8)
            self.results = list(wrapper(pool.imap(for_each, args), total=len(self.sampler)))
            pool.close()
        else:
            self.results = list(wrapper(map(for_each, args), total=len(self.sampler)))
