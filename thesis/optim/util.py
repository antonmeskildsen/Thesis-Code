import cv2 as cv
import numpy as np
from scipy import stats

def iris_params_to_img(params, size):
    img = np.zeros(size, dtype=np.uint8)
    p = params
    cv.ellipse(img, (int(p['cx']), int(p['cy'])), (int(p['width']//2), int(p['height']//2)), p['angle'], 0, 360, 255, -1)
    return img


def iris_mask(img, params):
    mask = iris_params_to_img(params, img.shape)
    return cv.bitwise_and(img, img, mask=mask)


def bfilter(img, **kwargs):
    return cv.bilateralFilter(img, kwargs['ksize'], kwargs['sigma_c'], kwargs['sigma_s'])


def gfilter(img, **kwargs):
    k = kwargs['ksize']
    return cv.GaussianBlur(img, (k, k), kwargs['sigma'])


def result_info(res):
    res = np.array(res)
    print(f'Mean: {res.mean()}, Std: {res.std()}')
    inte = stats.t.interval(0.95, df=len(res), loc=res.mean(), scale=stats.sem(res))
    print(f'95% confidence: {inte}')