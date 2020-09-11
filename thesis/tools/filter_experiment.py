import click
import yaml

import matplotlib.pyplot as plt
import matplotlib
from util.utilities import load_gaze_data, load_iris_data

matplotlib.use('TkAgg')

from thesis.optim.multi_objective import ObfuscationObjective, NaiveMultiObjectiveOptimizer
from thesis.optim.sampling import GridSearch, samples_num
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
        metrics = optimizer.metrics()
        # pprint.pprint(metrics)
        pareto = optimizer.pareto_frontier()
        # print(pareto)

        x = [e[1]['gaze'] for e in metrics]
        y = [e[1]['gradient_entropy'] for e in metrics]
        plt.scatter(x, y)
        x = [e[1]['gaze'] for e in pareto]
        y = [e[1]['gradient_entropy'] for e in pareto]
        plt.plot(x, y)
        plt.show()


def setup_optimizer(iris_data, gaze_data):
    objective = ObfuscationObjective(bfilter, iris_data, gaze_data)
    sampler = GridSearch(
        ['sigma_c', 'sigma_s'],
        [
            samples_num(1, 100, 5),
            samples_num(5, 15, 5),
        ]
    )
    # objective = ObfuscationObjective(gfilter, iris_data, gaze_data)
    # sampler = UniformSampler(
    #     ['sigma'],
    #     [
    #         samples_num(1, 50, 5),
    #     ]
    # )
    return NaiveMultiObjectiveOptimizer(objective, sampler)


def run_optimization():
    pass


def export_data():
    pass


if __name__ == '__main__':
    experiment()
