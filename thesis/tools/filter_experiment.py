from typing import List

import click
import yaml

from thesis.data import GazeDataset, SegmentationDataset


@click.command()
@click.argument('config')
def run_experiment(config):
    """
    CONFIG: Path to configuration yaml file.
    """
    with open(config) as config:
        config = yaml.safe_load(config)
        gaze_data = load_gaze_data(config['gaze_data'])
        iris_data = load_iris_data(config['iris_data'])


def load_gaze_data(datasets: List[str]) -> List[GazeDataset]:
    return list(map(GazeDataset.from_path, datasets))


def load_iris_data(datasets: List[str]) -> List[SegmentationDataset]:
    return list(map(SegmentationDataset.from_path, datasets))


def setup_optimizer():
    pass


def run_optimization():
    pass


def export_data():
    pass