from typing import List, Dict

import streamlit as st

import os
import re
import yaml
import json
from glob2 import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt

from optim.multi_objective import MultiObjectiveOptimizer, NaiveMultiObjectiveOptimizer, \
    PopulationMultiObjectiveOptimizer, ObfuscationObjective

from thesis.util.st_utils import file_select, type_name, json_to_strategy, progress, file_select_sidebar
from thesis.util.utilities import load_iris_data, load_gaze_data
from thesis.optim.sampling import GridSearch, UniformSampler, Sampler, PopulationInitializer
from thesis.optim import sampling
from thesis.optim.filters import bfilter, gfilter
from thesis.optim.objective_terms import AbsoluteGradientEntropy, RelativeGradientEntropy, GazeAbsoluteAccuracy, \
    GazeRelativeAccuracy
from thesis.optim.population import TruncationSelection, TournamentSelection, UniformCrossover, GaussianMutation

st.title('Obfuscation Experiment')
"""
This experiment aims to test the impact of various image obfuscation methods on eye tracking utility (gaze, 
feature detection) and iris recognition (accuracy, image entropy, iris code distortion). 

Since there are multiple objectives, the optimisation is focused on finding a pareto frontier defining 
optimal trade-offs for each obfuscation method applied. Comparing these frontiers makes it possible to 
compare the methods 
"""

"""
## Data configuration
"""

config_file = file_select('Data configuration file', 'configs/data/*.yaml')

with open(config_file) as config_file:
    config = yaml.safe_load(config_file)

gaze_data = load_gaze_data(config['gaze_data'])
iris_data = load_iris_data(config['iris_data'])

'**Gaze data:**'
f = [(g.name, len(g.test_samples), len(g.calibration_samples)) for g in gaze_data]
f = pd.DataFrame(f, columns=['Name', 'Test samples', 'Calibration samples'])
st.write(f)

'**Iris data:**'
f = [(g.name, len(g.samples)) for g in iris_data]
f = pd.DataFrame(f, columns=['Name', 'Samples'])
st.write(f)

st.sidebar.write("""
## Metrics and results
""")

iris_metrics = st.sidebar.multiselect('Iris metrics', (AbsoluteGradientEntropy, RelativeGradientEntropy),
                                      default=(AbsoluteGradientEntropy,), format_func=type_name)
gaze_metrics = st.sidebar.multiselect('Gaze metrics', (GazeAbsoluteAccuracy, GazeRelativeAccuracy),
                                      default=(GazeAbsoluteAccuracy,), format_func=type_name)

iris_terms = list(map(lambda x: x(), iris_metrics))
gaze_terms = list(map(lambda x: x(), gaze_metrics))

filters = st.sidebar.multiselect('Filter types', (gfilter, bfilter), default=(gfilter,), format_func=type_name)

st.sidebar.write(
    """
    ## Optimizer setup
    """)
method = st.sidebar.selectbox('Type', (NaiveMultiObjectiveOptimizer, PopulationMultiObjectiveOptimizer),
                              format_func=lambda x: x.__name__, index=1)

optimizers: Dict[str, MultiObjectiveOptimizer] = {}
projected_iterations = 0


def make_strategy(data, num):
    params, generators = [], []
    for k, v in data.items():
        params.append(k)
        generators.append(getattr(sampling, v['type'])(**v['params'], num=num))
    return params, generators


params = {}
if method == NaiveMultiObjectiveOptimizer:
    config_file = file_select_sidebar('Strategy file', 'configs/strategies/*.yaml')
    with open(config_file) as config_file:
        config = yaml.safe_load(config_file)
    st.write(config)
    params['configuration'] = config
    sampling = st.sidebar.selectbox('Sampling technique', (GridSearch, UniformSampler), format_func=type_name)

    for f in filters:
        objective = ObfuscationObjective(f, iris_data, gaze_data, iris_terms, gaze_terms)
        sampler: Sampler = sampling(*json_to_strategy(config[f.__name__]))
        projected_iterations += len(sampler)
        optimizers[f.__name__] = method([], objective, sampler)

elif method == PopulationMultiObjectiveOptimizer:
    config_file = file_select_sidebar('Strategy file', 'configs/population/*.yaml')
    with open(config_file) as config_file:
        config = yaml.safe_load(config_file)
    st.write(config)
    params['configuration'] = config
    k = st.sidebar.number_input('K', 0, 10, 5)
    iterations = st.sidebar.number_input('Iterations', 1, 100, 2)
    selection = st.sidebar.selectbox('Selection technique', (TruncationSelection, TournamentSelection),
                                     format_func=type_name)
    crossover = st.sidebar.selectbox('Crossover technique', (UniformCrossover,), format_func=type_name)
    # mutation = st.sidebar.selectbox('Mutation technique', (GaussianMutation,), format_func=type_name)

    pop_num = st.number_input('Population', 1, 1000, 10)

    for f in filters:
        objective = ObfuscationObjective(f, iris_data, gaze_data, iris_terms, gaze_terms)
        init = PopulationInitializer(*make_strategy(config[f.__name__], pop_num))

        sigmas = []
        means = []
        for param in config[f.__name__].values():
            sigmas.append(param['mutation']['sigma'])
            means.append(param['mutation']['mean'])
        mutation = GaussianMutation(np.array(sigmas), np.array(means))

        optimizers[f.__name__] = PopulationMultiObjectiveOptimizer([], objective, selection(k), crossover(), mutation,
                                                                   iterations, init)

"### Summary"
f'Expected number of iterations: {projected_iterations}'

st.sidebar.write("""
## Export
""")
path = 'results'
should_export = st.sidebar.checkbox('Export results')
if should_export:
    name = st.sidebar.text_input('Experiment name')
    description = st.sidebar.text_area('Description')

    existing = list(glob(os.path.join(path, f'{name}-*.json')))
    matches = [re.search('([0-9]+)', s) for s in existing]
    numbers = [int(match.group(0)) for match in matches]

    if len(numbers) == 0:
        next_num = 0
    else:
        next_num = max(numbers) + 1
    new_path = os.path.join(path, f'{name}-{next_num}.json')

    f'**Experiment {name}, no {next_num}**'

"""
## Run
"""

results = []

if st.button('Start experiment'):
    for filter_name, o in optimizers.items():
        f'Running optimizer for {filter_name}'
        o.run(wrapper=progress)

    'Results computed!'

    for filter_name, o in optimizers.items():
        metrics = o.metrics()
        pareto = [o.pareto_frontier(k) for k in range(max([m[2] for m in metrics]) + 1)]

        metrics_df = [{**a, **b, 'pareto': i in pareto[k], 'filter': filter_name, 'k': k} for i, (a, b, k) in
                      enumerate(metrics)]
        results.extend(metrics_df)
        # results[filter_name] = metrics_df
        metrics = pd.DataFrame(metrics_df)
        st.write(metrics)

        c = alt.Chart(metrics).mark_point().encode(
            x=gaze_metrics[0].__name__,
            y=iris_metrics[0].__name__,
            color='k:Q'
        ).interactive()

        c = c + alt.Chart(metrics).mark_line().encode(
            x=gaze_metrics[0].__name__,
            y=iris_metrics[0].__name__,
            color='k:Q'
        ).transform_filter(alt.datum.pareto).interactive()
        st.altair_chart(c, use_container_width=True)

    if should_export:
        with open(new_path, 'w') as file:
            json.dump({
                'name': name,
                'description': description,
                'optimizer': {
                    'method': method.__name__,
                    'params': params
                },
                # 'metrics': metrics,
                'results': results
            }, file)

        f'Successfully exported data at: {new_path}'
