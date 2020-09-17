import os

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

from glob import glob

from thesis.tracking.features import *
from thesis.tracking.gaze import BasicGaze
from tools.cli.utilities import load_json, load_images
from thesis.optim.filters import uniform_noise

import streamlit as st

st.title('Gaze experiments')

# path = '/home/anton/data/cap04'
sets = glob(os.path.join('/Users/Anton/Desktop/data/caps/', '*/'))
path = st.selectbox('Dataset', sets)

gaze_positions = load_json(path, 'positions')

WIDTH = 3840
HEIGHT = 2160

images = load_images(path)
images = [cv.cvtColor(img, cv.COLOR_BGR2GRAY) for img in images]

camera_matrix = np.load('thesis/util/cam/cameraMatrix.npy')
dist_coeffs = np.load('thesis/util/cam/distCoeffs.npy')
rms = np.load('thesis/util/cam/rms.npy')
rvecs = np.load('thesis/util/cam/rvecs.npy')
tvecs = np.load('thesis/util/cam/tvecs.npy')

if st.sidebar.checkbox('Remove distortion'):
    new_mat, roi = cv.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (640, 480), 0)
    mapx, mapy = cv.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, new_mat, (640, 480), 5)

    images = [cv.remap(img, mapx, mapy, cv.INTER_LINEAR) for img in images]

if st.sidebar.checkbox('Bilateral filter'):
    """
    # Bilateral filter
    """
    d = st.selectbox('Kernel size', [i for i in range(3, 45, 2)])
    s_color = st.number_input('Sigma color', 0, 100, 5)
    s_space = st.number_input('Sigma space', 0, 100, 5)
    images = [cv.bilateralFilter(img, d, s_color, s_space) for img in images]
    idx = st.number_input('Image', 0, len(images), 0)
    st.image(images[int(idx)], 'Sample')


st.sidebar.write("# Filter")
filter_choice = st.sidebar.selectbox('Obfuscation filter', ('uniform noise',))
if filter_choice == 'uniform noise':
    intensity = st.sidebar.slider('Noise intensity', 0, 128, 10)
    images = [uniform_noise(img, intensity) for img in images]


def show(img, **kwargs):
    pup, th = pupil_detector(img, debug=True)
    glints, timg = find_glints(img, pup[:2], debug=True, **kwargs)
    rgb_img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    if len(glints) > 0:
        glints = glints[[0]]
        cv.drawMarker(rgb_img, (int(pup[1]), int(pup[0])), (0, 50, 255), cv.MARKER_STAR, 40, 2)
        glints = sorted(glints, key=lambda g: g[0] + g[1])
        for i, g in enumerate(glints):
            cv.drawMarker(rgb_img, (int(g[1]), int(g[0])), (255 - i * 70, i * 70, 0), cv.MARKER_CROSS, 20, 2)

        pup = np.array(pup[:2])
        if len(glints) > 0:
            p = (pup - glints[0])  # /np.linalg.norm(pup-glints[0])
            st.write(p)

    st.image([th, timg, rgb_img], ['Pupil threshold', 'Threshold', 'Image'], width=200)


"""
# Glint detection options
"""

img = st.number_input('img', 0, 50, 5)
thresh = st.slider('threshold', 0, 255, 184)
radius = st.slider('radius', 0, 500, 80)
area = st.slider('min_area', 0, 50, 20)
ratio = st.slider('min_ratio', 0., 1., 0.3)

with st.spinner('Wait...'):
    show(images[img], threshold=thresh, radius=radius, max_area=area, min_ratio=ratio)

"""
# Gaze estimation
"""


def err(X, y, fov=87):
    err = model.predict(X) - normalize_coordinates(y, HEIGHT, WIDTH)
    return err * fov


def norm_err(X, y, fov=87):
    diff = model.predict(X) - normalize_coordinates(y, HEIGHT, WIDTH)
    dist = np.linalg.norm(diff, axis=1)
    return dist * fov


if st.sidebar.checkbox('Calc gaze'):
    order = st.number_input('Model order', 1, 5, 1)

    model = Pipeline([
        ('design matrix', PolynomialFeatures(order)),
        ('model', LinearRegression())
    ])

    n_cal = st.sidebar.selectbox('Calibration samples', [9, 25])

    train_X = images[:n_cal]
    train_y = gaze_positions[:n_cal]

    test_X = images[n_cal:]
    test_y = gaze_positions[n_cal:]

    model = BasicGaze(dict(threshold=thresh, radius=radius, max_area=area, min_ratio=ratio), model)
    model.calibrate(train_X, train_y)

    e = np.abs(err(test_X, test_y)).mean(axis=0)
    st.write(e)

    d = norm_err(test_X, test_y).mean(axis=0)
    st.write(d)

    if st.sidebar.checkbox('Display pupil positions'):
        plt.figure()
        pupils = np.array([pupil_detector(img) for img in images])
        centers = [p[:2] for p in pupils]
        glints = [
            find_glints(img, center)
            for img, center in zip(images, centers)
        ]

        # nan_removed = np.array([g for g in glints if ~np.isnan(g).any()])
        # avg = nan_removed.mean(axis=0)
        normed = []
        for i, (c, g) in enumerate(zip(centers, glints)):
            # print(i)
            normed.append([c[0] - g[0, 0], c[1] - g[0, 1]])
        normed = np.array(normed)
        plt.scatter(normed[:, 1], normed[:, 0])
        st.pyplot()

    plt.figure()
    ny = normalize_coordinates(test_y, HEIGHT, WIDTH)
    plt.scatter(ny[:, 0], ny[:, 1])

    pred_y = model.predict(test_X)
    plt.scatter(pred_y[:, 0], pred_y[:, 1])
    st.pyplot()

    plt.figure()
    ny = normalize_coordinates(train_y, HEIGHT, WIDTH)
    plt.scatter(ny[:, 0], ny[:, 1])

    pred_y = model.predict(train_X)
    plt.scatter(pred_y[:, 0], pred_y[:, 1])
    st.pyplot()

# thresh 184, radius 106, min_area 20, min_ratio 0.64

# e = ElseDetector()
# a = e.detect(train_X[0])
# print(a)

#
# g = BasicGaze(model)
# g.calibrate(train_X, train_y)
# #
