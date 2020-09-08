from typing import List

import click
import yaml

from thesis.data import GazeDataset, SegmentationDataset
from thesis.optim.multi_objective import ObfuscationObjective, NaiveMultiObjectiveOptimizer
from thesis.optim.sampling import uniform_sampler, samples_step, samples_num
from thesis.optim.filters import bfilter


@click.group()
def experiment():
    ...


@experiment.command()
@click.argument('config')
def run(config):
    """
    CONFIG: Path to configuration yaml file.
    """
    with open(config) as config:
        config = yaml.safe_load(config)
        gaze_data = load_gaze_data(config['gaze_data'])
        iris_data = load_iris_data(config['iris_data'])

        optimizer = setup_optimizer(iris_data, gaze_data)
        optimizer.run(status=True)


def load_gaze_data(datasets: List[str]) -> List[GazeDataset]:
    return list(map(GazeDataset.from_path, datasets))


def load_iris_data(datasets: List[str]) -> List[SegmentationDataset]:
    return list(map(SegmentationDataset.from_path, datasets))


def setup_optimizer(iris_data, gaze_data):
    objective = ObfuscationObjective(bfilter, iris_data, gaze_data)
    sampler = uniform_sampler(
        ['ksize', 'sigma_c', 'sigma_s'],
        [
            samples_step(3, 11, 2, stratified=False),
            samples_num(5, 15, 10),
            samples_num(5, 15, 10),
        ]
    )
    return NaiveMultiObjectiveOptimizer(objective, sampler)


def run_optimization():
    pass


def export_data():
    pass


if __name__ == '__main__':
    experiment()
