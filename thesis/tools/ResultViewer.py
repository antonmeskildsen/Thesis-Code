import streamlit as st

import os
import json

import pandas as pd
import altair as alt

from tools.st_utils import file_select

st.title("Obfuscation result analysis")
"""
This tool provides simple visualisations for the analysis of the obfuscation experimental results.
"""

file_path = file_select('Result file', os.path.join('results', '*.json'))
with open(file_path) as file:
    results = json.load(file)

frame = pd.DataFrame(results['results'])
st.write(frame)

st.sidebar.write("# Display settings")
x = st.sidebar.selectbox('X-axis', frame.columns, index=0)
y = st.sidebar.selectbox('Y-axis', frame.columns, index=1)

if results['optimizer']['method'] == 'PopulationMultiObjectiveOptimizer':
    k = st.sidebar.number_input('Choose iteration', 0, max(frame['k']), 0)
else:
    k = 0

# alt.X(x, scale=alt.Scale(domain=(0, dom)))
c = alt.Chart(frame[frame['k'] == k]).mark_point().encode(
    x=x,
    y=y,
    color='filter'
).interactive()

c = c + alt.Chart(frame[frame['k'] == k]).mark_line().encode(
    x=x,
    y=y,
    color='filter'
).transform_filter(alt.datum.pareto).interactive()
st.altair_chart(c, use_container_width=True)

'## Performance analysis'
threshold = st.slider('Maximum gaze error', 0.0, 2.0, 1.0)
filtered = frame[frame['AbsoluteGazeAccuracy'] < threshold]
# st.write(filtered)

c = alt.Chart(filtered[frame['k'] == k]).mark_point().encode(
    x=x,
    y=y,
    color='filter'
).interactive()

c = c + alt.Chart(filtered[frame['k'] == k]).mark_line().encode(
    x=x,
    y=y,
    color='filter'
).transform_filter(alt.datum.pareto).interactive()
st.altair_chart(c, use_container_width=True)
