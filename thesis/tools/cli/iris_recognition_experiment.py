import streamlit as st

import click

import os
from collections import defaultdict
import json
import numpy as np
import random
from scipy import stats
from sklearn.neighbors.kde import KernelDensity
import cv2 as cv
from glob2 import glob
from itertools import product

import seaborn as sns
import matplotlib.pyplot as plt

import npeet.entropy_estimators as ee

from thesis.optim.filters import gaussian_filter
from thesis.segmentation import IrisImage, IrisSegmentation, IrisCodeEncoder, IrisCode, SKImageIrisCodeEncoder

# base = '/home/anton/data/eyedata/iris'
base = '/Users/Anton/Desktop/data/iris'


@click.command()
@click.argument('config')
def main(config):
    with open(config) as file:
        config = json.load(file)

        scales = config['scales']
        angles = config['angles']
        wavelength_base = config['wavelength_base']
        angular = config['angular']
        radial = config['radial']


        num_rotations = config['rotation']['num']
        angle_step = config['rotation']['angle_step']



files = glob(os.path.join(base, '*.json'))
names = [os.path.basename(p).split('.')[0] for p in files]

dataset = st.selectbox('Dataset', names)

st.sidebar.markdown('# Filter setup')
the_filter = gaussian_filter

scales = st.sidebar.slider('Scales', 1, 10, 6)
angles = st.sidebar.slider('Angles', 1, 20, 6)
wavelength = st.sidebar.number_input('Wavelength Base', 0.0, 10.0, 0.5)
mult = st.sidebar.number_input('Wavelength multiplier', 1.0, 5.0, 1.81)
angular = st.sidebar.slider('Angular Resolution', 5, 100, 30, 1)
radial = st.sidebar.slider('Radial Resolution', 2, 50, 18, 1)

angle_tests = st.sidebar.number_input('Test angles', 1, 20, 7)
spacing = st.sidebar.number_input('Angular spacing', 0, 20, 5)

eps = st.sidebar.number_input('Epsilon', 0.0001, 20.0, 0.001, 0.0001)

# encoder = IrisCodeEncoder(scales, angles, angular, radial, wavelength, mult, eps)
encoder = SKImageIrisCodeEncoder(angles, angular, radial, eps)


def get_code(img, info):
    seg = IrisSegmentation.from_dict(info)
    # m = seg.get_mask((250, 250))
    iris_img = IrisImage(seg, img)
    polar, polar_mask = iris_img.to_polar(50, 30)
    scale = 3
    polar = cv.resize(polar, (0, 0), fx=scale, fy=scale)
    polar_mask = cv.resize(polar_mask, (0, 0), fx=scale, fy=scale)
    ic = encoder.encode(iris_img)
    code = ic.masked_image()
    saved = np.array(code)

    height = 30
    while height < len(code) and len(code) % height != 0:
        height += 1
    code = np.array(code).reshape((height, -1))
    st.image([iris_img.image, iris_img.mask * 255, polar, polar_mask * 255, code],
             ['regular', 'mask', 'polar', 'polar_mask', 'code'])
    return ic


def create_code(item, angles=1, angular_spacing=5):
    seg = IrisSegmentation.from_dict(item['points'])
    img = cv.imread(item['image'], cv.IMREAD_GRAYSCALE)
    # img = np.uint8(np.random.uniform(0, 255, img.shape))
    iris_img = IrisImage(seg, img)
    if angles == 1:
        ic = encoder.encode(iris_img)
        return [ic]
    else:
        angular_spacing_radians = angular_spacing / 360 * 2 * np.pi
        codes = []
        # for a in np.arange(-angles//2*angular_spacing, angles//2*angular_spacing, angular_spacing):
        #     codes.append(ic.shift(a))
        for a in np.linspace(-angular_spacing_radians / 2 * angles, angular_spacing_radians / 2 * angles, angles):
            codes.append(encoder.encode(iris_img, start_angle=a))
        return codes


# @st.cache(suppress_st_warning=True)
def create_codes(data):
    bar = st.progress(0)
    res = []
    for i, item in enumerate(data):
        res.append(create_code(item, angle_tests, spacing))
        bar.progress(i / len(data))
    bar.progress(1.0)
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

codes = create_codes(data['data'])
st.write("Codes created!")

bar = st.progress(0)
n = len(codes)
distance_matrix = np.zeros((n, n))
same_mask = np.zeros((n, n), np.bool)
num_samples = 0
for i in range(n):
    bar.progress(i / n)
    for j in range(n):
        # distance_matrix[i, j] = min([ca.dist(cb) for ca, cb in product(codes[i], codes[j])])
        # distance_matrix[i, j] = codes[i].dist(codes[j])
        in_data = data['data']
        info_i = in_data[i]['info']
        info_j = in_data[j]['info']
        same = False
        if info_i['user_id'] == info_j['user_id'] and info_i['eye'] == info_j['eye']:
            same = True
            same_mask[i, j] = True

        if same or random.random() < 2:  # Rate
            num_samples += 1
            distance_matrix[i, j] = min([codes[i][angle_tests // 2].dist(cb) for cb in codes[j]])
bar.progress(1.0)
# st.write(same_mask)
# st.write(distance_matrix)

intra_distances = []
inter_distances = []
for i in range(n):
    for j in range(n):
        if i == j:
            continue
        if same_mask[i, j]:
            intra_distances.append(distance_matrix[i, j])
        else:
            inter_distances.append(distance_matrix[i, j])

intra_distances = np.array(intra_distances)
inter_distances = np.array(inter_distances)
f'**Intra-distance mean:** {np.mean(intra_distances)}'
f'**Inter-distance mean:** {np.mean(inter_distances)}'

sns.distplot(intra_distances)
sns.distplot(inter_distances)
st.pyplot()

with open(os.path.join('results', 'recognition', f'{name}.json'), 'w') as file:
    json.dump({
        'parameters': {
            'scales': scales,
            'angles': angles,
            'wavelength': wavelength,
            'wavelength_multiplier': mult,
            'resolution': {
                'angular': angular,
                'radial': radial,
            }
        },
        'results': {
            'intra_distances': list(intra_distances),
            'inter_distances': list(inter_distances),
        }
    }, file)
