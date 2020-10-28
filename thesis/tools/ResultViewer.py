import streamlit as st

import os
import json

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict

from tools.st_utils import file_select

st.title("Obfuscation result analysis")
"""
This tool provides simple visualisations for the analysis of the obfuscation experimental results.
"""

file_path = file_select('Result file', os.path.join('results', '*.json'))
with open(file_path) as file:
    results = json.load(file)

frame = pd.DataFrame(results['results'])
frame['gaze_relative_error'] = frame['gaze_angle_error_filtered'] / frame['gaze_angle_error_source']
# frame['pupil_relative_error'] = frame[]
d = {c: 'mean' for c in frame.columns}
d['filter'] = 'first'
grouped = frame.groupby('group').agg(d)

st.write(grouped)

st.sidebar.write("# Display settings")
x = st.sidebar.selectbox('X-axis', frame.columns, index=0)
y = st.sidebar.selectbox('Y-axis', frame.columns, index=1)

do_pareto = st.sidebar.checkbox('Enable pareto')

if results['optimizer']['method'] == 'PopulationMultiObjectiveOptimizer':
    k = st.sidebar.number_input('Choose iteration', 0, max(frame['k']), 0)
else:
    k = 0

fig, ax = plt.subplots()
sns.scatterplot(x=x, y=y, hue='filter', data=grouped, ax=ax)
st.pyplot(fig)

'## Performance analysis'
selected_filter = st.sidebar.selectbox('Filter', pd.unique(frame['filter']))
threshold = st.number_input('Maximum relative gaze error', 0.0, 100.0, 1.0)
filtered = frame[frame['filter'] == selected_filter]

fig, ax = plt.subplots()
sns.kdeplot(x='iris_code_similarity', y='gradient_mutual_information', data=filtered, ax=ax)
st.pyplot(fig)
#
# '# 3D Stuff'
# fig = plt.figure()
# ax = Axes3D(fig)
# filt = frame[frame['filter'] == 'bilateral_filter']
# ax.scatter(xs=filt['sigma_s'], ys=filt['sigma_c'], zs=filt['gaze_angle_error_filtered'])
# ax.set_xlabel('Sigma spatial')
# ax.set_ylabel('Sigma color')
# ax.set_zlabel('Gaze angle error')
# st.pyplot(fig)
#
# z_axis = st.selectbox('Z axis metric', filt.columns)
#
# fig = plt.figure()
# ax = Axes3D(fig)
# ax.scatter(xs=filt['sigma_s'], ys=filt['sigma_c'], zs=filt[z_axis])
# ax.set_xlabel('Sigma spatial')
# ax.set_ylabel('Sigma color')
# ax.set_zlabel('Iris code similarity')
# st.pyplot(fig)
#
# vars = frame[
#     ['filter', 'gaze_relative_error', 'iris_code_similarity', 'gradient_mutual_information']]
# corr = vars.corr()
#
# fig, ax = plt.subplots()
# g = sns.pairplot(vars, hue='filter', diag_kind=None)
# st.pyplot(g)
#
# # Generate a mask for the upper triangle
# mask = np.triu(np.ones_like(corr, dtype=bool))
# cmap = sns.diverging_palette(80, 150, as_cmap=True)
#
# fig, ax = plt.subplots(figsize=(5, 5))
# sns.heatmap(corr, cmap=cmap, ax=ax, vmin=-1, vmax=1)
# st.pyplot(fig)
#
# vars = frame[
#     ['iris_code_similarity', 'gabor_mutual_information_1.0x', 'gabor_mutual_information_0.5x',
#      'gabor_mutual_information_0.25x', 'gabor_mutual_information_0.125x', 'gabor_mutual_information_0.0625x']]
# corr = vars.corr()
#
#
#
# # Generate a mask for the upper triangle
# mask = np.triu(np.ones_like(corr, dtype=bool))
# cmap = sns.diverging_palette(230, 20, as_cmap=True)
#
# fig, ax = plt.subplots(figsize=(5, 5))
# sns.heatmap(corr, cmap=cmap, ax=ax, vmin=-1, vmax=1)
# st.pyplot(fig)
