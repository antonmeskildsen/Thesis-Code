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


def joint_gradient_histogram(img_a, img_b, divisions=32):
    div = 512//divisions

    img_a = cv.equalizeHist(img_a)
    img_b = cv.equalizeHist(img_b)
    xm_a = dx(img_a)
    ym_a = dy(img_a)
    xm_b = dx(img_b)
    ym_b = dy(img_b)
    xm_a = np.int32(xm_a / xm_a.max() * (256 // div))
    ym_a = np.int32(ym_a / ym_a.max() * (256 // div))
    xm_b = np.int32(xm_b / xm_b.max() * (256 // div))
    ym_b = np.int32(ym_b / ym_b.max() * (256 // div))

    hist_a = np.zeros((512 // div, 512 // div))
    hist_b = np.zeros((512 // div, 512 // div))

    hist_joint = defaultdict(lambda: 0)

    offset = 256 // div - 1
    height, width = img_a.shape
    for y in range(height):
        for x in range(width):
            hist_joint[(ym_a[y, x] + offset,
                        xm_a[y, x] + offset,
                        ym_b[y, x] + offset,
                        xm_b[y, x] + offset)] += 1
            hist_a[ym_a[y, x] + offset, xm_a[y, x] + offset] += 1
            hist_b[ym_b[y, x] + offset, xm_b[y, x] + offset] += 1

    joint_sum = sum(hist_joint.values())
    hist_joint = {k: v / joint_sum for k, v in hist_joint.items()}
    hist_a /= hist_a.sum()
    hist_b /= hist_b.sum()

    return hist_a, hist_b, hist_joint


def joint_histogram(img_a, img_b, divisions=32):
    div = 512//divisions

    img_a = cv.equalizeHist(img_a)
    img_b = cv.equalizeHist(img_b)

    hist_a = np.zeros((512 // div, 512 // div))
    hist_b = np.zeros((512 // div, 512 // div))


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
