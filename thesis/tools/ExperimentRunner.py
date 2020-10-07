import copy
from typing import Dict

import streamlit as st

import os
import re
import yaml
import json
from glob2 import glob

import numpy as np
import pandas as pd
import altair as alt

from data import GazeDataset, PupilDataset, SegmentationDataset
from optim.multi_objective import MultiObjectiveOptimizer, NaiveMultiObjectiveOptimizer, \
    PopulationMultiObjectiveOptimizer, ObfuscationObjective

from tools.st_utils import file_select, type_name, json_to_strategy, progress, file_select_sidebar
from tools.cli.utilities import load_iris_data, load_gaze_data, load_pupil_data
from thesis.optim.sampling import GridSearch, UniformSampler, Sampler, PopulationInitializer
from thesis.optim import sampling
from thesis.optim.filters import bilateral_filter, gaussian_filter, uniform_noise, gaussian_noise, salt_and_pepper, \
    cauchy_noise
from thesis.optim.objective_terms import AbsoluteGradientEntropy, RelativeGradientEntropy, AbsoluteMutualInformation, \
    RelativeMutualInformation, AbsoluteGazeAccuracy, RelativeGazeAccuracy, IrisCodeSimilarity, ImageSimilarity, \
    AbsolutePupilDistanceBaseAlgorithm, AbsolutePupilDistanceElse, AbsolutePupilDistanceExcuse, \
    RelativePupilDistanceBaseAlgorithm, RelativePupilDistanceElse, RelativePupilDistanceExcuse, \
    AbsoluteOriginalGradientEntropy
from thesis.optim.population import TruncationSelection, TournamentSelection, UniformCrossover, GaussianMutation

st.title('Obfuscation Experiment')
"""
This experiment aims to test the impact of various image obfuscation methods on eye tracking utility (gaze, 
feature detection) and iris recognition (accuracy, image entropy, iris code distortion). 

Since there are multiple objectives, the optimisation is focused on finding a pareto frontier defining 
optimal trade-offs for each obfuscation method applied. Comparing these frontiers makes it possible to 
compare the methods.
"""

"""
## Data configuration
"""

config_file = file_select('Data configuration file', 'configs/data/*.yaml')

with open(config_file) as config_file:
    config = yaml.safe_load(config_file)


@st.cache(hash_funcs={
    GazeDataset: lambda _: None,
    PupilDataset: lambda _: None,
    SegmentationDataset: lambda _: None
}, allow_output_mutation=True)
def load_data():
    return load_gaze_data(config['gaze_data']), load_iris_data(config['iris_data']), \
           load_pupil_data(config['pupil_data'])


loaded = copy.deepcopy(load_data())
gaze_data, iris_data, pupil_data = loaded

'**Gaze data:**'
f = [(g.name, len(g.test_samples), len(g.calibration_samples)) for g in gaze_data]
f = pd.DataFrame(f, columns=['Name', 'Test samples', 'Calibration samples'])
st.write(f)

'**Iris data:**'
f = [(g.name, len(g.samples)) for g in iris_data]
f = pd.DataFrame(f, columns=['Name', 'Samples'])
st.write(f)

'**Pupil data:**'
f = [(g.name, len(g.samples)) for g in pupil_data]
f = pd.DataFrame(f, columns=['Name', 'Samples'])
st.write(f)

st.sidebar.write("""
## Metrics and results
""")

iris_metrics = st.sidebar.multiselect('Iris metrics',
                                      (IrisCodeSimilarity,),
                                      default=(IrisCodeSimilarity,), format_func=type_name)
histogram_metrics = st.sidebar.multiselect('Histogram metrics',
                                           (
                                               AbsoluteGradientEntropy, AbsoluteOriginalGradientEntropy,
                                               RelativeGradientEntropy, AbsoluteMutualInformation,
                                               RelativeMutualInformation),
                                           default=(RelativeMutualInformation,), format_func=type_name)
gaze_metrics = st.sidebar.multiselect('Gaze metrics', (AbsoluteGazeAccuracy, RelativeGazeAccuracy),
                                      default=(AbsoluteGazeAccuracy,), format_func=type_name)
pupil_metrics = st.sidebar.multiselect('Pupil metrics',
                                       (AbsolutePupilDistanceBaseAlgorithm, AbsolutePupilDistanceElse,
                                        AbsolutePupilDistanceExcuse, RelativePupilDistanceBaseAlgorithm,
                                        RelativePupilDistanceElse, RelativePupilDistanceExcuse),
                                       default=(AbsolutePupilDistanceElse,), format_func=type_name)

iris_terms = list(map(lambda x: x(), iris_metrics))
histogram_terms = list(map(lambda x: x(), histogram_metrics))
gaze_terms = list(map(lambda x: x(), gaze_metrics))
pupil_terms = list(map(lambda x: x(), pupil_metrics))

filters = st.sidebar.multiselect('Filter types',
                                 (gaussian_filter, bilateral_filter, gaussian_noise, uniform_noise, salt_and_pepper,
                                  cauchy_noise),
                                 default=(gaussian_filter,),
                                 format_func=type_name)

st.sidebar.write(
    """
    ## Optimizer setup
    """)
method = st.sidebar.selectbox('Type', (NaiveMultiObjectiveOptimizer, PopulationMultiObjectiveOptimizer),
                              format_func=lambda x: x.__name__, index=0)

optimizers: Dict[str, MultiObjectiveOptimizer] = {}
projected_iterations = 0


def make_strategy(data, num):
    parameters, generators = [], []
    for k, v in data.items():
        parameters.append(k)
        generators.append(getattr(sampling, v['type'])(**v['params'], num=num))
    return parameters, generators


iris_samples = st.sidebar.number_input('Iris Samples', 1, min(map(len, iris_data)), 50)
gaze_samples = st.sidebar.number_input('Gaze Samples', 1, min(map(len, gaze_data)), 50)
pupil_samples = st.sidebar.number_input('Pupil Samples', 1, min(map(len, pupil_data)), 50)

st.sidebar.markdown('### Optimizer parameters')

params = {}
if method == NaiveMultiObjectiveOptimizer:
    config_file = file_select_sidebar('Strategy file', 'configs/strategies/*.yaml')
    with open(config_file) as config_file:
        config = yaml.safe_load(config_file)
    # st.write(config)
    params['configuration'] = config
    sampling = st.sidebar.selectbox('Sampling technique', (GridSearch, UniformSampler), format_func=type_name)

    for f in filters:
        objective = ObfuscationObjective(f, iris_data, gaze_data, pupil_data, iris_terms, histogram_terms, gaze_terms,
                                         pupil_terms,
                                         iris_samples, gaze_samples, pupil_samples)
        params, generators = json_to_strategy(config[f.__name__])
        for p, g in zip(params, generators):
            f'Param: {p}'
            st.write(g)
        sampler: Sampler = sampling(params, generators)
        projected_iterations += len(sampler)
        optimizers[f.__name__] = method([], objective, sampler)

elif method == PopulationMultiObjectiveOptimizer:
    config_file = file_select_sidebar('Strategy file', 'configs/population/*.yaml')
    with open(config_file) as config_file:
        config = yaml.safe_load(config_file)
    st.write(config)
    params['configuration'] = config
    generations = st.sidebar.number_input('Generations (K)', 0, 10, 5)
    iterations = st.sidebar.number_input('Iterations', 1, 100, 2)
    selection = st.sidebar.selectbox('Selection technique', (TruncationSelection, TournamentSelection),
                                     format_func=type_name)
    crossover = st.sidebar.selectbox('Crossover technique', (UniformCrossover,), format_func=type_name)
    # mutation = st.sidebar.selectbox('Mutation technique', (GaussianMutation,), format_func=type_name)

    pop_num = st.number_input('Population', 1, 1000, 10)

    projected_iterations = iterations * pop_num * len(filters)

    for f in filters:
        objective = ObfuscationObjective(f, iris_data, gaze_data, pupil_data, iris_terms, histogram_terms, gaze_terms,
                                         pupil_terms,
                                         iris_samples, gaze_samples, pupil_samples)
        init = PopulationInitializer(*make_strategy(config[f.__name__], pop_num))

        sigmas = []
        means = []
        for param in config[f.__name__].values():
            sigmas.append(param['mutation']['sigma'])
            means.append(param['mutation']['mean'])
        mutation = GaussianMutation(np.array(sigmas), np.array(means))

        optimizers[f.__name__] = PopulationMultiObjectiveOptimizer([], objective, selection(generations), crossover(),
                                                                   mutation, iterations, init)

"### Summary"
f'Expected number of iterations: {projected_iterations}'

st.sidebar.write("""
## Export
""")
path = 'results'
new_path = ''
name = ''
description = ''
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

        metrics_df = [{**a, **b, 'pareto': i in pareto[generations], 'filter': filter_name, 'k': generations} for
                      i, (a, b, generations) in
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
