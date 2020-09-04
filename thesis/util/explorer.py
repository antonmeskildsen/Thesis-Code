import streamlit as st

import os
from collections import defaultdict
import json
import numpy as np
import cv2 as cv
from glob2 import glob

from eyeinfo import IrisImage, IrisEncoder, IrisSegmentationDescriptor

"# Explorer"

# base = '/home/anton/data/eyedata/iris'
base = '/Users/Anton/Desktop/data/iris'

files = glob(os.path.join(base, '*.json'))
names = [os.path.basename(p).split('.')[0] for p in files]

dataset = st.selectbox('Dataset', names)

encoder = IrisEncoder(3, 5, 2)


def get_code(img, info):
    inner = info['inner']
    outer = info['outer']
    lower = info['lower']
    upper = info['upper']
    seg = IrisSegmentationDescriptor.from_points(inner, outer, lower, upper)
    # m = seg.get_mask((250, 250))
    iris_img = IrisImage(seg, img)
    polar, _ = iris_img.to_polar(50, 30)
    ic = encoder(iris_img)
    code = np.array(ic.code)
    saved = np.array(code)
    code = code / 2 + 0.5
    code = np.array(code).reshape((30, -1))
    st.image([iris_img.image, iris_img.mask*255, code], ['regular', 'polar', 'code'])
    return saved


def create_code(item):
    inner = info['inner']
    outer = info['outer']
    lower = info['lower']
    upper = info['upper']
    seg = IrisSegmentationDescriptor.from_points(inner, outer, lower, upper)
    img = cv.imread(item['image'], cv.IMREAD_GRAYSCALE)
    iris_img = IrisImage(seg, img)
    ic = encoder(iris_img)
    return ic.code


# @st.cache(suppress_st_warning=True)
def create_codes(data):
    bar = st.progress(0)
    res = []
    for i, item in enumerate(data):
        res.append(create_code(item))
        bar.progress(i / len(data))
    return res


def hamming_distance(c1: np.ndarray, c2: np.ndarray):
    n = ((c1 * c2) == 0).sum()
    c1[c2 == 0] = 0
    c2[c1 == 0] = 0
    div = c1.size - n
    if div == 0:
        return 0
    else:
        return (c1 != c2).sum() / (c1.size - n)


with open(os.path.join(base, f'{dataset}.json')) as f:
    data = json.load(f)

    id_map = defaultdict(list)
    for i, x in enumerate(data['data']):
        id_map[x['info']['user_id']].append((i, x['info']))

    num_images = len(data['data'])
    # index = st.number_input(f'Image index (0-{num_images-1})', min_value=0, max_value=num_images-1, value=0)

    if st.checkbox('Compare'):
        user1 = st.selectbox('User ID', list(id_map.keys()))
        index, val = st.selectbox('Index A', id_map[user1])
        user2 = st.selectbox('User ID 2', list(id_map.keys()))
        index2, val2 = st.selectbox('Index B', id_map[user2])

        info = data['data'][index]
        info2 = data['data'][index2]
        img = cv.imread(info['image'], cv.IMREAD_GRAYSCALE)
        # img2 = cv.imread(info2['image'], cv.IMREAD_GRAYSCALE)

        # if img is None or img2 is None:
        #     raise IOError("Could not open image")

        c1 = get_code(img, info['points'])
        # c2 = get_code(img2, info2['points'])
        # n = ((c1 * c2) == 0).sum()
        # c1[c2 == 0] = 0
        # c2[c1 == 0] = 0
        # dist = (c1 != c2).sum() / (c1.size - n)
        # dist = np.linalg.norm(np.array(c1, np.float64)-np.array(c2, np.float64))
        # f'Distance: {dist}'

    "## Code generation"
    if st.checkbox('Stats'):
        codes = create_codes(data['data'])

        bar = st.progress(0)
        n = len(codes)
        distance_matrix = np.zeros((n, n))
        same_mask = np.zeros((n, n), np.bool)
        for i in range(n):
            bar.progress(i / n)
            for j in range(n):
                distance_matrix[i, j] = hamming_distance(codes[i], codes[j])
                in_data = data['data']
                info_i = in_data[i]['info']
                info_j = in_data[j]['info']
                if info_i['user_id'] == info_j['user_id'] and info_i['eye'] == info_j['eye']:
                    same_mask[i, j] = True
