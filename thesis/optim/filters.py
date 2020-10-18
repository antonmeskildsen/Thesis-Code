import cv2 as cv
import numpy as np
from scipy import stats
from medpy.filter import smoothing

rng = np.random.default_rng()


def mean_filter(img, size):
    return cv.blur(img, size)


def anisotropic_diffusion(img, kappa: float, gamma: float, iterations: int=1):
    img = smoothing.anisotropic_diffusion(img, iterations, kappa, gamma)
    return img


def bilateral_filter(img, sigma_c: float, sigma_s: float, steps: int=0):
    # k = (int(sigma_s * 3) // 2) * 2 + 1
    for _ in range(steps):
        img = cv.bilateralFilter(img, (0, 0), sigma_c, sigma_s)
    return img


def gaussian_filter(img, sigma: float):
    k = (int(sigma * 3) // 2) * 2 + 1
    return cv.GaussianBlur(img, (k, k), sigma)


def uniform_noise(img, intensity: float):
    return np.uint8(np.clip(img + np.random.uniform(-intensity // 2, intensity // 2, img.shape), 0, 255))


def gaussian_noise(img, loc, scale: float):
    return np.uint8(np.clip(img + np.random.normal(loc, scale, img.shape), 0, 255))


def cauchy_noise(img, scale: float):
    return np.uint8(np.clip(img + rng.standard_cauchy(img.shape) * scale, 0, 255))


def salt_and_pepper(img, intensity: float, density: float):
    mask = 1 - np.random.rand(*img.shape)
    mask[mask > density] = 1
    mask[mask <= density] = 0
    return np.uint8(np.clip(img + mask * np.random.uniform(intensity), 0, 255))
