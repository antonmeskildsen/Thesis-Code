import streamlit as st
from glob2 import glob

from thesis.optim import sampling


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
