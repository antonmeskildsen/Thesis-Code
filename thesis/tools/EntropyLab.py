import streamlit as st

import os
import json
from glob2 import glob
from collections import defaultdict

import numpy as np
import cv2 as cv

import seaborn as sns
import matplotlib.pyplot as plt

from thesis.entropy import gradient_histogram, histogram, entropy
from thesis.segmentation import IrisSegmentation, IrisImage, IrisCodeEncoder
from thesis.entropy import dx, dy

"""
# Entropy Test Lab
"""

# base = '/home/anton/data/eyedata/iris'
base = '/Users/Anton/Desktop/data/iris'

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
ints = 100
filtered = img.copy()
filtered = np.uint8(np.clip(img + np.random.uniform(-ints // 2, ints // 2, img.shape), 0, 255))
# filtered = np.uint8(np.clip(img + np.random.uniform(-ints // 2, ints // 2, img.shape), 0, 255))

tmp_img = IrisImage(seg, img)
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
filtered = cv.GaussianBlur(filtered, (0, 0), 0.5)
# filtered = cv.medianBlur(img, 55)
# filtered = cv.bilateralFilter(img, 0, 50, 80)
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
div = 16
# hist_base = np.histogram(img, div, normed=True)[0].reshape((1, -1))
# hist_filt = np.histogram(filtered, div, normed=True)[0].reshape((1, -1))
hist_base = np.zeros((256 // div))
hist_filt = np.zeros((256 // div))

joint = np.zeros((256 // div, 256 // div))
height, width = pimg.shape
for y in range(height):
    for x in range(width):
        joint[pimg[y, x] // div, pfiltered[y, x] // div] += 1
        hist_base[pimg[y, x] // div] += 1
        hist_filt[pfiltered[y, x] // div] += 1

joint /= joint.sum()
hist_base /= hist_base.sum()
hist_filt /= hist_filt.sum()

fig, ax = plt.subplots()
sns.heatmap(joint, ax=ax)
st.pyplot(fig)

mutual_information = 0
for y in range(256 // div):
    for x in range(256 // div):
        v = joint[y, x]
        base_v = hist_base[x]
        filt_v = hist_filt[x]
        d = base_v * filt_v
        if v > 0 and base_v > 0 and filt_v > 0:
            t = np.log2(v / d)
            r = v * t
            mutual_information += r

joint_grad = np.zeros((512 // div, 512 // div, 512 // div, 512 // div))
img_grad = np.zeros((512 // div, 512 // div))
fil_grad = np.zeros((512 // div, 512 // div))
img_dx = dx(pimg)
img_dx = np.int32(img_dx / img_dx.max() * (256 // div))
img_dy = dy(img)
img_dy = np.int32(img_dy / img_dy.max() * (256 // div))
fil_dx = dx(pfiltered)
fil_dx = np.int32(fil_dx / fil_dx.max() * (256 // div))
fil_dy = dy(filtered)
fil_dy = np.int32(fil_dy / fil_dy.max() * (256 // div))

offset = 256 // div - 1

height, width = pimg.shape
for y in range(height):
    for x in range(width):
        joint_grad[
            img_dy[y, x] + offset,
            img_dx[y, x] + offset,
            fil_dy[y, x] + offset,
            fil_dx[y, x] + offset] += 1
        img_grad[img_dy[y, x] + offset, img_dx[y, x] + offset] += 1
        fil_grad[fil_dy[y, x] + offset, fil_dx[y, x] + offset] += 1

joint_grad /= joint_grad.sum()
img_grad /= img_grad.sum()
fil_grad /= fil_grad.sum()

m2 = 0
for a in range(512 // div):
    for b in range(512 // div):
        for c in range(512 // div):
            for d in range(512 // div):
                v = joint_grad[a, b, c, d]
                base_v = img_grad[a, b]
                filt_v = fil_grad[c, d]
                if v > 0 and base_v > 0 and filt_v > 0:
                    d = base_v * filt_v
                    t = np.log2(v / d)
                    r = v * t
                    m2 += r

f'Gradient mutual: {m2}'

joint = hist_base.T.dot(hist_filt)
joint /= joint.max()

f'Intensity: {mutual_information}'

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
