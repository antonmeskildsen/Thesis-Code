import eyeinfo

import numpy as np
import cv2 as cv

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