import streamlit as st

import os
from collections import defaultdict
import json
import numpy as np
import cv2 as cv
from glob2 import glob
from itertools import product
from scipy import stats

import seaborn as sns
import matplotlib.pyplot as plt

from thesis.segmentation import IrisImage, IrisSegmentation, IrisCodeEncoder, IrisCode
from thesis.tools.st_utils import file_select

"""
# Iris Recognition analysis
"""

file_path = file_select('Result file', os.path.join('results', 'recognition', '*.json'))
with open(file_path) as file:
    data = json.load(file)

st.write(data['parameters'])

intra_distances = np.array(data['results']['intra_distances'])
inter_distances = np.array(data['results']['inter_distances'])

clip = st.slider('Clip margin', 0.0, 0.5, 0.1)

intra_distances = intra_distances[intra_distances < 1-clip]
intra_distances = intra_distances[intra_distances > clip]
inter_distances = inter_distances[inter_distances < 1-clip]
inter_distances = inter_distances[inter_distances > clip]

intra_distances.sort()
inter_distances.sort()

threshold = st.slider('Threshold', 0.3, 0.5, 0.35)

false_accepts = (inter_distances < threshold).sum() / len(inter_distances)
false_rejects = (intra_distances > threshold).sum() / len(intra_distances)

start = st.sidebar.slider('Minimum', 0.2, 0.5, 0.33)
stop = st.sidebar.slider('Maximum', 0.2, 0.7, 0.42)


"""## Distribution and properties."""

inter_mean = np.nanmean(inter_distances)
inter_std = np.nanstd(inter_distances)
intra_mean = np.nanmean(intra_distances)
intra_std = np.nanstd(intra_distances)

inter_dist = stats.norm(loc=inter_mean, scale=inter_std)
intra_dist = stats.norm(loc=intra_mean, scale=intra_std)

sns.distplot(inter_distances, kde=False, norm_hist=True)
sns.distplot(intra_distances, kde=False, norm_hist=True)

xs = np.linspace(0.0, 1.0, 200)
inter_y = []
intra_y = []
for x in xs:
    inter_y.append(inter_dist.pdf(x))
    intra_y.append(intra_dist.pdf(x))

plt.plot(xs, inter_y)
plt.plot(xs, intra_y)
st.pyplot()

thresholds = np.linspace(start, stop, 30)
far = []
frr = []
far_est = []
frr_est = []

for x in thresholds:
    far.append((inter_distances < x).sum() / len(inter_distances))
    frr.append((intra_distances > x).sum() / len(intra_distances))
    far_est.append(inter_dist.cdf(x))
    frr_est.append(1 - intra_dist.cdf(x))

"""## Acceptable FAR consequences
The following shows the resulting FRR given a specified acceptable FAR."""
max_far = st.number_input('Acceptable FAR', 0.0, 1.0, 0.1, format='%f', step=10e-8)
count = len(intra_distances)
n = 0

while far[n] < max_far:
    n += 1

f'FRR: {frr[n]}'

"""## ROC curve"""
log_scale = st.sidebar.checkbox('Log scale')
display_estimate = st.sidebar.checkbox('Show estimate')

plt.plot(far, frr, label='Data')
if display_estimate:
    plt.plot(far_est, frr_est, label='Estimated distribution')
# plt.vlines([max_far], 0, 1)
plt.xlabel('FAR')
plt.ylabel('FRR')
plt.grid()

if log_scale:
    plt.xscale('log')
    plt.yscale('log')
    # plt.axis(xmin=10 ** -4, xmax=1, ymin=10 ** -4, ymax=1)
# st.write(far)
# st.write(frr)
plt.legend()
st.pyplot()

