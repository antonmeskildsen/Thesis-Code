import cv2 as cv
import numpy as np
from scipy import stats

rng = np.random.default_rng()


def bilateral_filter(img, sigma_c, sigma_s):
    k = (int(sigma_s * 3) // 2) * 2 + 1
    return cv.bilateralFilter(img, k, sigma_c, sigma_s)


def gaussian_filter(img, sigma):
    k = (int(sigma * 3) // 2) * 2 + 1
    return cv.GaussianBlur(img, (k, k), sigma)


def uniform_noise(img, intensity):
    return np.uint8(np.clip(img + np.random.uniform(-intensity // 2, intensity // 2, img.shape), 0, 255))


def gaussian_noise(img, loc, scale):
    return np.uint8(np.clip(img + np.random.normal(loc, scale, img.shape), 0, 255))


def cauchy_noise(img, scale):
    return np.uint8(np.clip(img + rng.standard_cauchy(img.shape) * scale, 0, 255))


def salt_and_pepper(img, intensity, density):
    mask = 1 - np.random.rand(*img.shape)
    mask[mask > density] = 1
    mask[mask <= density] = 0
    return np.uint8(np.clip(img + mask * np.random.uniform(intensity), 0, 255))
