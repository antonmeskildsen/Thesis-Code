import streamlit as st
from glob2 import glob
import os

import numpy as np

from thesis.optim import sampling
from pupilfit import fit_else
from thesis.deepeye import deepeye


def fit_else_ref(img, debug=False):
    center, axes, angle = fit_else(img)
    thresh = np.zeros(img.shape)
    if debug:
        return [center[1], center[0], axes[1], axes[0], angle], thresh
    else:
        return [center[1], center[0], axes[1], axes[0], angle]


def create_deepeye_func():
    path = os.path.join(os.getcwd(), 'thesis/deepeye/models/default.ckpt')
    deepeye_model = deepeye.DeepEye(model=path)

    def deepeye_ref(img, debug=False):
        coords = deepeye_model.run(img)
        thresh = np.zeros(img.shape)
        if debug:
            return [coords[1], coords[0], 0, 0, 0], thresh
        else:
            return [coords[1], coords[0], 0, 0, 0]

    return deepeye_ref


def file_select(label, pattern):
    file_list = glob(pattern)
    return st.selectbox(label, file_list)


def file_select_sidebar(label, pattern):
    file_list = glob(pattern)
    return st.sidebar.selectbox(label, file_list)


def type_name(x):
    return x.__name__


def json_to_strategy(data):
    params, generators = [], []
    for k, v in data.items():
        params.append(k)
        generators.append(getattr(sampling, v['type'])(**v['params']))
    return params, generators


def progress(iterator, total):
    bar = st.progress(0)
    for i, v in enumerate(iterator):
        bar.progress(i/total)
        yield v
    bar.progress(100)

