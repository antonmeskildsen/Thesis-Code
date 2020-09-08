import cv2 as cv


def bfilter(img, **kwargs):
    return cv.bilateralFilter(img, kwargs['ksize'], kwargs['sigma_c'], kwargs['sigma_s'])


def gfilter(img, **kwargs):
    k = kwargs['ksize']
    return cv.GaussianBlur(img, (k, k), kwargs['sigma'])