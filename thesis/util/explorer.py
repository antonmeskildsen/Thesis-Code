import streamlit as st

import os
from collections import defaultdict
import json
import numpy as np
import cv2 as cv
from glob2 import glob

from thesis.segmentation import IrisSegmentation, IrisImage, IrisCode

"# Explorer"

base = '/home/anton/data/eyedata/iris'

files = glob(os.path.join(base, '*.json'))
names = [os.path.basename(p).split('.')[0] for p in files]

dataset = st.selectbox('Dataset', names)

def get_code(img, info):
    seg = IrisSegmentation.from_json(info)
    iris_img = IrisImage(seg, img)
    polar, _ = iris_img.polar_image(50, 30)
    code = IrisCode(iris_img, 5).code
    saved = code
    code = np.array(code).reshape((20, -1))
    st.image([img, polar, code], ['regular', 'polar', 'code'])
    return saved


with open(os.path.join(base, f'{dataset}.json')) as f:
    data = json.load(f)

    id_map = defaultdict(list)
    for i, x in enumerate(data['data']):
        id_map[x['info']['user_id']].append((i, x['info']))

    num_images = len(data['data'])
    # index = st.number_input(f'Image index (0-{num_images-1})', min_value=0, max_value=num_images-1, value=0)
    user = st.selectbox('User ID', list(id_map.keys()))
    index, val = st.selectbox('Index A', id_map[user])
    index2, val2 = st.selectbox('Index B', id_map[user])

    info = data['data'][index]
    info2 = data['data'][index2]
    img = cv.imread(info['image'], cv.IMREAD_GRAYSCALE)
    img2 = cv.imread(info2['image'], cv.IMREAD_GRAYSCALE)

    c1 = get_code(img, info['points'])
    c2 = get_code(img2, info2['points'])
    dist = np.linalg.norm(np.array(c1, np.float64)-np.array(c2, np.float64))
    f'Distance: {dist}'
