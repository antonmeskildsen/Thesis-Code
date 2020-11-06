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
from thesis.optim.pareto import pareto_frontier, pareto_set

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

def aggregate(df, method: str):
    d = {c: method for c in df.columns}
    d['filter'] = 'first'
    return df.groupby(['filter', 'group']).agg(d)


method = st.selectbox('Aggregate method', ('mean', 'min', 'max', 'std', 'median'))

st.sidebar.write("# Display settings")
x = st.sidebar.selectbox('X-axis', frame.columns, index=0)
y = st.sidebar.selectbox('Y-axis', frame.columns, index=1)

do_pareto = st.sidebar.checkbox('Enable pareto')

gaze_t = st.number_input('Relative gaze error threshold', 0.0, value=1.0)

frame = frame[frame['gaze_relative_error'] < gaze_t]

agg = aggregate(frame, method)
st.write(agg)


if results['optimizer']['method'] == 'PopulationMultiObjectiveOptimizer':
    k = st.sidebar.number_input('Choose iteration', 0, max(frame['k']), 0)
else:
    k = 0

fig, ax = plt.subplots()
sns.scatterplot(x=x, y=y, hue='filter', data=agg, ax=ax)
st.pyplot(fig)

st.write(len(agg))
fig, ax = plt.subplots()
rows = []
for f in agg['filter'].unique():
    bfilter = agg[agg['filter'] == f]
    s = pareto_set(np.array(bfilter[[x, y]]))
    row = bfilter.iloc[s]
    rows.append(row)
    sns.lineplot(x=x, y=y, data=row, ax=ax, marker='o')
st.pyplot(fig)

pareto_optimal = pd.concat(rows)
# st.write(pareto_optimal)


"# Optimal values"
agg.reset_index(drop=True, inplace=True)
m = agg.loc[agg.groupby('filter')['iris_code_similarity'].idxmin()]

params = {
    'bilateral_filter': ['sigma_c', 'sigma_s'],
    'non_local_means': ['h'],
    'gaussian_filter': ['sigma'],
    'mean_filter': ['size'],
    'median_filter': ['size'],
    'uniform_noise': ['intensity'],
    'gaussian_noise': ['loc', 'scale'],
    'cauchy_noise': ['scale'],
    'salt_and_pepper': ['intensity', 'density']
}

out_params = {}
info = []
for f, p in params.items():
    info.append(m[m['filter'] == f].iloc[0])
    out_params[f] = {p_name: m[m['filter'] == f].iloc[0].at[p_name] for p_name in p}
st.write(out_params)
info = pd.concat(info, axis=1).T
st.write(info[['filter', 'gaze_relative_error']])

# fname = st.text_input('File name')
# if st.checkbox('Export'):
#     with open('configs/')

# '## Performance analysis'
# selected_filter = st.sidebar.selectbox('Filter', pd.unique(frame['filter']))
# threshold = st.number_input('Maximum relative gaze error', 0.0, 100.0, 1.0)
# filtered = frame[frame['filter'] == selected_filter]

# fig, ax = plt.subplots()
# sns.kdeplot(x='iris_code_similarity', y='gradient_mutual_information', data=filtered, ax=ax)
# st.pyplot(fig)
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
