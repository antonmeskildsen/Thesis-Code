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

from thesis.util.st_utils import file_select, type_name, json_to_strategy, progress
from thesis.util.utilities import load_iris_data, load_gaze_data
from thesis.optim.sampling import GridSearch, UniformSampler, Sampler
from thesis.optim.filters import bfilter, gfilter
from thesis.optim.objective_terms import Term, AbsoluteEntropy, RelativeEntropy, GazeAbsoluteAccuracy, \
    GazeRelativeAccuracy

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

"""
## Optimizer setup
"""
method = st.selectbox('Type', (NaiveMultiObjectiveOptimizer, PopulationMultiObjectiveOptimizer),
                      format_func=lambda x: x.__name__)

filters = st.multiselect('Filter types', (gfilter, bfilter), format_func=type_name)
optimizers: Dict[str, MultiObjectiveOptimizer] = {}
projected_iterations = 0

params = {}
if method == NaiveMultiObjectiveOptimizer:
    config_file = file_select('Strategy file', 'configs/strategies/*.yaml')
    with open(config_file) as config_file:
        config = yaml.safe_load(config_file)
    st.write(config)
    params['configuration'] = config
    sampling = st.selectbox('Sampling technique', (GridSearch, UniformSampler), format_func=type_name)

    for f in filters:
        objective = ObfuscationObjective(f, iris_data, gaze_data)
        sampler: Sampler = sampling(*json_to_strategy(config[f.__name__]))
        projected_iterations += len(sampler)
        optimizers[f.__name__] = method(objective, sampler)

elif method == PopulationMultiObjectiveOptimizer:
    st.warning('Not yet implemented')

"### Summary"
f'Expected number of iterations: {projected_iterations}'

"""
## Metrics and results
"""

metrics = st.multiselect('Metrics', (AbsoluteEntropy, RelativeEntropy, GazeAbsoluteAccuracy, GazeRelativeAccuracy),
                         default=(AbsoluteEntropy, GazeAbsoluteAccuracy),
                         format_func=type_name)

"""
## Export setup
"""
path = 'results'
should_export = st.checkbox('Export results')
if should_export:
    name = st.text_input('Experiment name')
    description = st.text_area('Description')

    existing = list(glob(os.path.join(path, f'{name}-*.json')))
    matches = [re.search('([0-9]+)', s) for s in existing]
    numbers = [int(match.group(0)) for match in matches]

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
        pareto = o.pareto_frontier()
        metrics_df = [{**a, **b, 'pareto': i in pareto, 'filter': filter_name} for i, (a, b) in enumerate(metrics)]
        results.extend(metrics_df)
        # results[filter_name] = metrics_df
        metrics = pd.DataFrame(metrics_df)

        c = alt.Chart(metrics).mark_point().encode(
            x='gaze',
            y='gradient_entropy',
        ).interactive()

        c = c + alt.Chart(metrics).mark_line().encode(
            x='gaze',
            y='gradient_entropy',
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