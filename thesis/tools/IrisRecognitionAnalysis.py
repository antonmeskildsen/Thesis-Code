import streamlit as st

import os
from collections import defaultdict
import json
import numpy as np
import cv2 as cv
from glob2 import glob
from itertools import product

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

intra_distances = np.array(data['results']['intra_distances'])
inter_distances = np.array(data['results']['inter_distances'])

threshold = st.slider('Threshold', 0.3, 0.5, 0.35)

false_accepts = (inter_distances < threshold).sum() / len(inter_distances)
false_rejects = (intra_distances > threshold).sum() / len(intra_distances)

f'{false_accepts}, {false_rejects}'


thresholds = np.linspace(0.33, 0.35, 100)
far = []
frr = []

for x in thresholds:
    far.append((inter_distances < x).sum() / len(inter_distances))
    frr.append((intra_distances > x).sum() / len(intra_distances))

plt.scatter(far, frr)
plt.xlabel('FAR')
plt.ylabel('FRR')
plt.grid()
# plt.xscale('log')
# plt.yscale('log')
# plt.axis(xmin=10**-4, xmax=1, ymin=10**-2)
# st.write(far)
# st.write(frr)
st.pyplot()
