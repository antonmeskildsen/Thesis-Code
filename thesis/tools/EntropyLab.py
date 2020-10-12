import streamlit as st

import os
import json
import math
from glob2 import glob
from collections import defaultdict

import numpy as np
import cv2 as cv

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import time

import altair as alt

from thesis.entropy import gradient_histogram, histogram, entropy
from thesis.segmentation import IrisSegmentation, IrisImage, IrisCodeEncoder
from thesis.entropy import *

"""
# Entropy Test Lab
"""

base = '/home/anton/data/eyedata/iris'
# base = '/Users/Anton/Desktop/data/iris'

files = glob(os.path.join(base, '*.json'))
names = [os.path.basename(p).split('.')[0] for p in files]

dataset = st.selectbox('Dataset', names)

with open(os.path.join(base, f'{dataset}.json')) as f:
    data = json.load(f)

id_map = defaultdict(list)
for i, x in enumerate(data['data']):
    id_map[x['info']['user_id']].append((i, x['info']))

num_images = len(data['data'])

user = st.selectbox('User ID', list(id_map.keys()))
index, val = st.selectbox('Index A', id_map[user])

sample = data['data'][index]

seg = IrisSegmentation.from_dict(sample['points'])
img = cv.imread(sample['image'], cv.IMREAD_GRAYSCALE)
ints = st.slider('Intensity', 1, 255, 50)
filtered = img.copy()
# filtered = np.uint8(np.clip(img + np.random.uniform(-ints // 2, ints // 2, img.shape), 0, 255))
# img = np.uint8(np.random.uniform(0, 255, img.shape))
# filtered = np.uint8(np.random.uniform(0, 255, img.shape))

tmp_img = IrisImage(seg, img)

f'Number of pixels considered: {tmp_img.mask.sum()} -> suggested bin count: {np.sqrt(tmp_img.mask.sum()*4)}'
# num = 5000
# coords = np.random.randint(0, filtered.size, num)
# height, width = filtered.shape
# filtered[coords // width, coords % width] = 255
# ints = 100
# s = 30
# filtered[tmp_img.mask == 1] += np.uint8(np.random.uniform(-ints // 2, ints//2, filtered[tmp_img.mask == 1].shape))
# filtered = np.int32(filtered)
# for x in range(0, filtered.shape[1] // s, 2):
#     filtered[:, x * s:(x + 1) * s] += 35
# filtered = np.uint8(np.clip(filtered, 0, 255))
filtered = cv.GaussianBlur(filtered, (0, 0), 3)
# filtered = cv.medianBlur(img, 3)
# filtered = cv.bilateralFilter(img, 0, 15, 20)
# filtered = np.uint8(np.random.uniform(0, 255, img.shape))

st.image([img, filtered])

scales = st.sidebar.slider('Scales', 1, 10, 6)
angles = st.sidebar.slider('Angles', 1, 20, 6)
wavelength = st.sidebar.number_input('Wavelength Base', 0.0, 10.0, 0.5)
mult = st.sidebar.number_input('Wavelength multiplier', 1.0, 5.0, 1.81)
angular = st.sidebar.slider('Angular Resolution', 5, 100, 30, 1)
radial = st.sidebar.slider('Radial Resolution', 2, 50, 18, 1)

angle_tests = st.sidebar.number_input('Test angles', 1, 20, 7)
spacing = st.sidebar.number_input('Angular spacing', 0, 20, 5)

eps = st.sidebar.number_input('Epsilon', 0.0, 20.0, 0.01)

encoder = IrisCodeEncoder(scales, angles, angular, radial, wavelength, mult, eps)

iris_img = IrisImage(seg, img)
pimg, _ = iris_img.to_polar(angular, radial)
pimg = cv.equalizeHist(pimg)
base_code = encoder.encode(iris_img)

filter_img = IrisImage(seg, filtered)
pfiltered, _ = filter_img.to_polar(angular, radial)
pfiltered = cv.equalizeHist(pfiltered)
filtered_code = encoder.encode(filter_img)

d = base_code.dist(filtered_code)
f'Hamming Distance: {d}'

st.image([base_code.masked_image().reshape((30, -1)), filtered_code.masked_image().reshape((30, -1))])

bins = 50
div = 64
# hist_base = np.histogram(img, div, normed=True)[0].reshape((1, -1))
# hist_filt = np.histogram(filtered, div, normed=True)[0].reshape((1, -1))
# hist_base = np.zeros((256 // div))
# hist_filt = np.zeros((256 // div))
#
# joint = np.zeros((256 // div, 256 // div))
# height, width = pimg.shape
# for y in range(height):
#     for x in range(width):
#         joint[pimg[y, x] // div, pfiltered[y, x] // div] += 1
#         hist_base[pimg[y, x] // div] += 1
#         hist_filt[pfiltered[y, x] // div] += 1
#
# joint /= joint.sum()
# hist_base /= hist_base.sum()
# hist_filt /= hist_filt.sum()

divi = 512
hist_base, hist_filt, joint = joint__gabor_1d_histogram(img, filtered, iris_img.mask, divi)

fig, ax = plt.subplots()
ax.bar(np.arange(0, divi, 1), hist_base)
ax.bar(np.arange(0, divi, 1), hist_filt)
st.pyplot(fig)

fig, ax = plt.subplots()
j2 = np.zeros((divi, divi))
for k, v in joint.items():
    j2[k] = v
sns.heatmap(j2, ax=ax)
st.pyplot(fig)

# fig, ax = plt.subplots()
# sns.heatmap(joint, ax=ax)
# st.pyplot(fig)

mi = mutual_information(hist_base, hist_filt, joint)

div = 128

st.image(iris_img.mask * 255)

d = st.slider('Number of Divisions', 4, 512, 16)
t = time.perf_counter()
img_grad, fil_grad, joint_grad = joint_gabor_histogram(img, filtered, mask=iris_img.mask, scale=1, theta=0, divisions=d)
elapsed = time.perf_counter() - t
f'Elapsed time for gradient histogram: {elapsed:.4f} seconds'

# fig, ax = plt.subplots()
# ax.imshow(img_grad-fil_grad)
# st.pyplot(fig)

eps = 10e-5

joint_stuff = [(k, v) for k, v in joint_grad.items()]
coord = joint_stuff[0][0]
st.write(joint_stuff[0])
st.write(img_grad[coord[:2]])
st.write(fil_grad[coord[2:]])

log_norm_img = LogNorm(vmin=img_grad.min() + eps, vmax=img_grad.max())
log_norm_fil = LogNorm(vmin=fil_grad.min() + eps, vmax=fil_grad.max())
cbar_ticks_img = [math.pow(10, i) for i in
                  range(math.floor(math.log10(img_grad.min() + eps)), 1 + math.ceil(math.log10(img_grad.max())))]
cbar_ticks_fil = [math.pow(10, i) for i in
                  range(math.floor(math.log10(fil_grad.min() + eps)), 1 + math.ceil(math.log10(fil_grad.max())))]

fig, ax = plt.subplots(2, 1)
sns.heatmap(img_grad, ax=ax[0], norm=log_norm_img, cbar_kws={'ticks': cbar_ticks_img})
sns.heatmap(fil_grad, ax=ax[1], norm=log_norm_fil, cbar_kws={'ticks': cbar_ticks_fil})
st.pyplot(fig)

# for pos, v in joint_grad.items():
#     val_a = img_grad[pos[:2]]
#     val_b = img_grad[pos[2:]]
#
#     if val_a == 0 or val_b == 0:
#         continue
#
#     e_joint = v * (np.log2(v) - np.log2(val_a) - np.log2(val_b))
#     e_single = -val_a * np.log2(val_a)
#     if e_joint != e_single:
#         f'Mismatch at ({pos}): {e_joint}, {e_single}'
#         f'Values: joint: {v}, img_grad: {val_a}, fil_grad: {val_b}'

e = entropy(img_grad)
f'Gradient entropy of original: {e}'

e2 = entropy(fil_grad)
f'Gradient entropy of filtered: {e2}'

t = time.perf_counter()
m2 = mutual_information_grad(img_grad, fil_grad, joint_grad)
elapsed = time.perf_counter() - t
f'Elapsed time for entropy calculation: {elapsed:.4f} seconds'

# for a in range(512 // div):
#     for b in range(512 // div):
#         for c in range(512 // div):
#             for d in range(512 // div):
#                 v = joint_grad[a, b, c, d]
#                 base_v = img_grad[a, b]
#                 filt_v = fil_grad[c, d]
#                 if v > 0 and base_v > 0 and filt_v > 0:
#                     d = base_v * filt_v
#                     t = np.log2(v / d)
#                     r = v * t
#                     m2 += r


f'Gradient mutual: {m2}'

f'Ratio: {(e - m2) / e * 100:.2f} %'
f'Direct: {m2/e}'

joint = hist_base.T.dot(hist_filt)
joint /= joint.max()

ei = entropy(hist_base)
f'Intensity original: {ei}'
f'Intensity mutual: {mi}'
f'Ratio: {(ei - mi) / ei * 100:.2f}%'

pimg = cv.resize(pimg, (0, 0), fx=4, fy=4)
pfiltered = cv.resize(pfiltered, (0, 0), fx=4, fy=4)
st.image([pimg, pfiltered])

# fig, ax = plt.subplots(1, 2)
# ax[0].bar(range(bins), hist_base[0])
# ax[1].bar(range(bins), hist_filt[0])
# st.pyplot(fig)

# st.write(joint.max())
#
# st.image(joint)

# img = cv.GaussianBlur(img, (0, 0), 100)
# img = cv.medianBlur(img, 201)
#
# scales = st.sidebar.slider('Scales', 1, 10, 6)
# angles = st.sidebar.slider('Angles', 1, 20, 6)
# wavelength = st.sidebar.number_input('Wavelength Base', 0.0, 10.0, 0.5)
# mult = st.sidebar.number_input('Wavelength multiplier', 1.0, 5.0, 1.81)
# angular = st.sidebar.slider('Angular Resolution', 5, 100, 30, 1)
# radial = st.sidebar.slider('Radial Resolution', 2, 50, 18, 1)
#
# angle_tests = st.sidebar.number_input('Test angles', 1, 20, 7)
# spacing = st.sidebar.number_input('Angular spacing', 0, 20, 5)
#
# eps = st.sidebar.number_input('Epsilon', 0.0, 20.0, 0.01)
#
# iris_img = IrisImage(seg, img)
# encoder = IrisCodeEncoder(scales, angles, angular, radial, wavelength, mult, eps)
# ic = encoder.encode(iris_img)
#
# angular = st.number_input('Angular resolution', 1, 100, 30)
# radial = st.number_input('Radial resolution', 1, 100, 18)
#
# polar, pmask = iris_img.to_polar(angular, radial)
#
# st.image([img, polar])
#
# st.image(ic.masked_image().reshape((30, -1)))
# code_img = np.uint8(ic.masked_image())
#
# hist = [0, 0]
# code = np.uint8((ic.code + 1) // 2)
# for i, val in enumerate(code):
#     if ic.mask[i] == 1:
#         hist[val] += 1
#
# total = sum(hist)
# hist[0] /= total
# hist[1] /= total
#
# st.write(hist)
# # hist = np.histogram(code_img, 2, normed=True)[0]
#
# grad_entropy = - (hist[0] * np.log2(hist[0]) + hist[1] * np.log2(hist[1]))
# f'Iris code entropy: {grad_entropy}'

# maximal = np.uint8(np.random.uniform(0, 255, (radial, angular)))
