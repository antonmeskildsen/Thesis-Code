from typing import Dict

import click
import os
import yaml

from tqdm import tqdm

import pandas as pd
import json

from thesis.optim.sampling import GridSearch, Sampler
from thesis.optim.multi_objective import MultiObjectiveOptimizer, NaiveMultiObjectiveOptimizer, ObfuscationObjective, \
    PopulationMultiObjectiveOptimizer
from thesis.optim import multi_objective
from thesis.optim import filters
from thesis.optim import objective_terms
from thesis.tools.cli.utilities import load_gaze_data, load_iris_data, load_pupil_data
from thesis.tools.st_utils import json_to_strategy

from memory_profiler import profile


def progress_tqdm(iterator, total):
    for v in tqdm(iterator, total=total):
        yield v


@click.command()
@click.argument('config')
@click.argument('name')
def main(config, name):
    with open(os.path.join('configs/filter_experiment/', config)) as file:
        data = yaml.safe_load(file)

        with open(data['data_config']) as data_config_file, open(data['strategy_config']) as strategy_config_file:
            data_config = yaml.safe_load(data_config_file)
            strategy_config = yaml.safe_load(strategy_config_file)

        def load_data():
            return load_gaze_data(data_config['gaze_data']), load_iris_data(data_config['iris_data']), \
                   load_pupil_data(data_config['pupil_data'])

        loaded = load_data()
        gaze_data, iris_data, pupil_data = loaded

        metrics = data['metrics']

        def do(x):
            return getattr(objective_terms, x)()

        iris_terms = list(map(do, metrics['iris_metrics']))
        histogram_terms = list(map(do, metrics['histogram_metrics']))
        gaze_terms = list(map(do, metrics['gaze_metrics']))
        pupil_terms = list(map(do, metrics['pupil_metrics']))

        optimizers: Dict[str, MultiObjectiveOptimizer] = {}
        projected_iterations = 0

        def make_strategy(data, num):
            parameters, generators = [], []
            for k, v in data.items():
                parameters.append(k)
                generators.append(getattr(sampling, v['type'])(**v['params'], num=num))
            return parameters, generators

        samples = data['samples']
        iris_samples = samples['iris']
        gaze_samples = samples['gaze']
        pupil_samples = samples['pupil']

        method = getattr(multi_objective, data['method'])

        params = {}
        if method == NaiveMultiObjectiveOptimizer:
            params['configuration'] = strategy_config
            sampling = GridSearch

            for f in map(lambda f: getattr(filters, f), data['filters']):
                objective = ObfuscationObjective(f, iris_data, gaze_data, pupil_data, iris_terms, histogram_terms,
                                                 gaze_terms,
                                                 pupil_terms,
                                                 iris_samples, gaze_samples, pupil_samples)
                params, generators = json_to_strategy(strategy_config[f.__name__])
                sampler: Sampler = sampling(params, generators)
                projected_iterations += len(sampler)
                optimizers[f.__name__] = method([], objective, sampler)
        else:
            raise NotImplementedError("Only NaiveMultiObjectiveOptimizer currently supported.")
        # elif method == PopulationMultiObjectiveOptimizer:
        #     params['configuration'] = strategy_config
        #     generations = st.sidebar.number_input('Generations (K)', 0, 10, 5)
        #     iterations = st.sidebar.number_input('Iterations', 1, 100, 2)
        #     selection = st.sidebar.selectbox('Selection technique', (TruncationSelection, TournamentSelection),
        #                                      format_func=type_name)
        #     crossover = st.sidebar.selectbox('Crossover technique', (UniformCrossover,), format_func=type_name)
        #     # mutation = st.sidebar.selectbox('Mutation technique', (GaussianMutation,), format_func=type_name)
        #
        #     pop_num = st.number_input('Population', 1, 1000, 10)
        #
        #     projected_iterations = iterations * pop_num * len(filters)
        #
        #     for f in filters:
        #         objective = ObfuscationObjective(f, iris_data, gaze_data, pupil_data, iris_terms, histogram_terms,
        #                                          gaze_terms,
        #                                          pupil_terms,
        #                                          iris_samples, gaze_samples, pupil_samples)
        #         init = PopulationInitializer(*make_strategy(config[f.__name__], pop_num))
        #
        #         sigmas = []
        #         means = []
        #         for param in config[f.__name__].values():
        #             sigmas.append(param['mutation']['sigma'])
        #             means.append(param['mutation']['mean'])
        #         mutation = GaussianMutation(np.array(sigmas), np.array(means))
        #
        #         optimizers[f.__name__] = PopulationMultiObjectiveOptimizer([], objective, selection(generations),
        #                                                                    crossover(),
        #                                                                    mutation, iterations, init)

        results = []

        for filter_name, o in optimizers.items():
            f'Running optimizer for {filter_name}'
            o.run(wrapper=progress_tqdm)

        'Results computed!'

        for filter_name, o in optimizers.items():
            metrics = o.metrics()
            pareto = [o.pareto_frontier(k) for k in range(max([m[2] for m in metrics]) + 1)]

            metrics_df = [{**a, **b, 'pareto': i in pareto[generations], 'filter': filter_name, 'k': generations} for
                          i, (a, b, generations) in
                          enumerate(metrics)]
            results.extend(metrics_df)
            # results[filter_name] = metrics_df
            metrics = pd.DataFrame(metrics_df)

        with open(os.path.join('results', f'{name}.json'), 'w') as f2:
            json.dump({
                'name': name,
                'optimizer': {
                    'method': method.__name__,
                    'params': params
                },
                # 'metrics': metrics,
                'results': results
            }, f2)


if __name__ == '__main__':
    main()
