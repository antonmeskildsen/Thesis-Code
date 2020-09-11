import cv2 as cv


def bfilter(img, sigma_c, sigma_s):
    k = (int(sigma_s * 3) // 2) * 2 + 1
    return cv.bilateralFilter(img, k, sigma_c, sigma_s)


def gfilter(img, sigma):
    k = (int(sigma*3) // 2) * 2 + 1
    return cv.GaussianBlur(img, (k, k), sigma)