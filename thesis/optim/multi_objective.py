from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Callable

import numpy as np

from thesis.data import SegmentationDataset, GazeDataset
from thesis.optim.sampling import Sampler, PopulationInitializer
from thesis.optim.objective_terms import GazeTerm, SegmentationTerm
from thesis.optim.population import SelectionMethod, MutationMethod, CrossoverMethod


class Objective(ABC):

    @abstractmethod
    def metrics(self) -> List[str]:
        ...

    @abstractmethod
    def eval(self, params) -> List[float]:
        ...

    @abstractmethod
    def output_dimensions(self):
        ...


@dataclass
class ObfuscationObjective(Objective):
    filter: Callable
    iris_datasets: List[SegmentationDataset]
    gaze_datasets: List[GazeDataset]

    iris_terms: List[SegmentationTerm]
    gaze_terms: List[GazeTerm]

    def metrics(self) -> List[str]:
        return list(map(lambda x: type(x).__name__, self.iris_terms)) + list(
            map(lambda x: type(x).__name__, self.gaze_terms))

    def eval(self, params):
        iris_results = [[] for _ in range(len(self.iris_terms))]
        gaze_results = [[] for _ in range(len(self.gaze_terms))]

        for dataset in self.iris_datasets:
            for sample in dataset.samples:
                output = self.filter(sample.image.image, **params)
                for i, term in enumerate(self.iris_terms):
                    iris_results[i].append(term(sample, output))

        for dataset in self.gaze_datasets:
            for sample in dataset.test_samples:
                output = self.filter(sample.image, **params)
                for i, term in enumerate(self.gaze_terms):
                    gaze_results[i].append(term(dataset.model, sample, output))

        a = list(map(np.mean, iris_results))
        b = list(map(np.mean, gaze_results))
        return list(map(np.mean, iris_results)) + list(map(np.mean, gaze_results))

    def output_dimensions(self):
        return len(self.iris_terms) + len(self.gaze_terms)


@dataclass
class MultiObjectiveOptimizer:
    results: List
    objective: Objective

    @abstractmethod
    def run(self, *, wrapper=None):
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


def dominates(y, y_mark):
    y = np.array(y)
    y_mark = np.array(y_mark)
    return np.all(y <= y_mark) and np.any(y < y_mark)


@dataclass
class PopulationMultiObjectiveOptimizer(MultiObjectiveOptimizer):
    selection_method: SelectionMethod
    crossover_method: CrossoverMethod
    mutation_method: MutationMethod
    iterations: int
    initial_population: PopulationInitializer

    def run(self, *, wrapper=None):
        pop = list(self.initial_population)
        params = list(pop[0].keys())

        m = self.objective.output_dimensions()
        m_pop = len(pop)
        m_subpop = m_pop // m + 1  # TODO: Check for correct solution (is it important to get len(parents)==m_pop?

        iterator = range(self.iterations)
        if wrapper is not None:
            iterator = wrapper(range(self.iterations), self.iterations)

        self.results = []
        for k in iterator:
            ys = [self.objective.eval(x) for x in pop]
            self.results.extend(list(zip(pop, [dict(zip(self.objective.metrics(), y)) for y in ys], [k] * len(pop))))

            parents = []
            for i in range(m):
                selected = self.selection_method.select([y[i] for y in ys])
                parents.extend(selected[:m_subpop])

            p = np.random.choice(m_pop * 2, m_pop * 2, False)

            def p_ind(i):
                return parents[p[i] % m_pop][p[i] // m_pop]

            parents = [(p_ind(i), p_ind(i + 1)) for i in range(0, 2 * m_pop, 2)]
            pop_values = [list(p.values()) for p in pop]
            children = [self.crossover_method.crossover(pop_values[p[0]], pop_values[p[1]]) for p in parents]
            pop_values = [self.mutation_method.mutate(c) for c in children]
            pop_values = [np.clip(v, 0, None) for v in pop_values]
            pop = [dict(zip(params, p)) for p in pop_values]


@dataclass
class NaiveMultiObjectiveOptimizer(MultiObjectiveOptimizer):
    sampler: Sampler

    def run(self, *, wrapper=None):
        self.results = []
        if wrapper is not None:
            iterator = wrapper(self.sampler, len(self.sampler))
        else:
            iterator = self.sampler

        for params in iterator:
            output = self.objective.eval(params)
            self.results.append((params, dict(zip(self.objective.metrics(), output)), 0))
