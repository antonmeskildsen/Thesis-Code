import cv2 as cv
import numpy as np


def bfilter(img, sigma_c, sigma_s):
    k = (int(sigma_s * 3) // 2) * 2 + 1
    return cv.bilateralFilter(img, k, sigma_c, sigma_s)


def gfilter(img, sigma):
    k = (int(sigma * 3) // 2) * 2 + 1
    return cv.GaussianBlur(img, (k, k), sigma)


def uniform_noise(img, intensity):
    return np.uint8(np.clip(img + np.random.uniform(-intensity // 2, intensity // 2, img.shape), 0, 255))


def gaussian_noise(img, loc, scale):
    return np.uint8(np.clip(img + np.random.normal(loc, scale, img.shape), 0, 255))
