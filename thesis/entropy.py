import eyeinfo

import numpy as np
import cv2 as cv

from collections import defaultdict

GRAD_RESOLUTION = 255 * 2 + 1


def dx(img):
    return cv.Sobel(img, cv.CV_32F, 1, 0, ksize=3)


def dy(img):
    return cv.Sobel(img, cv.CV_32F, 0, 1, ksize=3)


def gradient_histogram(img, mask=None):
    img = cv.equalizeHist(img)
    xm = np.int16(dx(img) / 4)  # important for histogram calculation!
    ym = np.int16(dy(img) / 4)

    if mask is None:
        mask = np.ones(img.shape, np.uint8)

    hist = eyeinfo.gradient_histogram(xm, ym, mask)
    return hist


def joint_gradient_histogram(img_a, img_b, divisions=4):
    img_a = cv.equalizeHist(img_a)
    img_b = cv.equalizeHist(img_b)

    xm_a = dx(img_a)
    ym_a = dy(img_a)
    xm_b = dx(img_b)
    ym_b = dy(img_b)
    xm_a = np.int32(xm_a / xm_a.max() * (divisions // 2))
    ym_a = np.int32(ym_a / ym_a.max() * (divisions // 2))
    xm_b = np.int32(xm_b / xm_b.max() * (divisions // 2))
    ym_b = np.int32(ym_b / ym_b.max() * (divisions // 2))

    hist_a = np.zeros((divisions, divisions))
    hist_b = np.zeros((divisions, divisions))

    hist_joint = defaultdict(lambda: np.double(0))

    offset = (divisions // 2) - 1
    height, width = img_a.shape
    for y in range(height):
        for x in range(width):
            hist_joint[(ym_a[y, x] + offset,
                        xm_a[y, x] + offset,
                        ym_b[y, x] + offset,
                        xm_b[y, x] + offset)] += 1
            hist_a[ym_a[y, x] + offset, xm_a[y, x] + offset] += 1
            hist_b[ym_b[y, x] + offset, xm_b[y, x] + offset] += 1

    joint_sum = height * width
    assert height * width == sum(hist_joint.values())
    hist_joint = defaultdict(float, {k: v / joint_sum for k, v in hist_joint.items()})
    hist_a /= hist_a.sum()
    hist_b /= hist_b.sum()

    return hist_a, hist_b, hist_joint


def mutual_information(hist_a, hist_b, hist_joint):
    e = 0
    n = len(next(iter(hist_joint))) // 2
    for pos, v in hist_joint.items():
        pos_left = pos[:n]
        pos_right = pos[n:]
        base_v = hist_a[pos_left]
        filt_v = hist_b[pos_right]
        if base_v > 0 and filt_v > 0:
            d = base_v * filt_v
            t = np.log2(v / d)
            r = v * t
            e += r
    return e


def joint_histogram(img_a, img_b, divisions=32):
    img_a = cv.equalizeHist(img_a)
    img_b = cv.equalizeHist(img_b)
    img_a = np.int32(img_a / img_a.max() * (divisions // 2))
    img_b = np.int32(img_b / img_b.max() * (divisions // 2))

    hist_a = np.zeros((divisions))
    hist_b = np.zeros((divisions))

    hist_joint = defaultdict(float)

    offset = (divisions // 2) - 1
    height, width = img_a.shape
    for y in range(height):
        for x in range(width):
            hist_joint[(img_a[y, x] + offset,
                        img_b[y, x] + offset)] += 1
            hist_a[img_a[y, x]] += 1
            hist_b[img_b[y, x]] += 1

    joint_sum = height * width
    assert height * width == sum(hist_joint.values())
    hist_joint = defaultdict(float, {k: v / joint_sum for k, v in hist_joint.items()})
    hist_a /= hist_a.sum()
    hist_b /= hist_b.sum()

    return hist_a, hist_b, hist_joint


def entropy(hist):
    r = -eyeinfo.entropy(hist)
    return r


def histogram(img, mask=None):
    return eyeinfo.histogram(img, mask)


def kl_divergence(h1, h2):
    e = 0
    for j in range(-255, 256):
        for i in range(-255, 256):
            v1 = h1[i, j]
            v2 = h2[i, j]

            if v2 != 0:
                vt = v1 / v2
                if vt != 0:
                    e += v1 * np.log2(vt)

    return e
