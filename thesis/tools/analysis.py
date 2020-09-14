from typing import List, Dict

import streamlit as st

import os
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
from thesis.optim.objective_terms import Term, AbsoluteGradientEntropy, RelativeGradientEntropy, GazeAbsoluteAccuracy, \
    GazeRelativeAccuracy

st.title("Obfuscation result analysis")
"""
This tool provides simple visualisations for the analysis of the obfuscation experimental results.
"""

file_path = file_select('Result file', os.path.join('results', '*.json'))
with open(file_path) as file:
    results = json.load(file)


frame = pd.DataFrame(results['results'])
st.write(frame)
# for filter_name, r in metrics.items():
#     frame = pd.DataFrame(r)
#     st.write(frame)

dom = st.slider('x-axis domain', 0.05, 1., 0.5)

x = st.selectbox('X-axis', frame.columns, index=0)
y = st.selectbox('Y-axis', frame.columns, index=1)

# alt.X(x, scale=alt.Scale(domain=(0, dom)))
c = alt.Chart(frame).mark_point().encode(
    x=x,
    y=y,
    color='filter'
).interactive()

c = c + alt.Chart(frame).mark_line().encode(
    x=x,
    y=y,
    color='filter'
).transform_filter(alt.datum.pareto).interactive()
st.altair_chart(c, use_container_width=True)



# c = alt.Chart(frame).mark_point().encode(
#     x=x,
#     y=y,
#     color='filter'
# ).interactive()
# st.altair_chart(c, use_container_width=True)