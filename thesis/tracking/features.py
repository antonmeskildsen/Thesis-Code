import numpy as np
from typing import Tuple

import cv2 as cv
import matplotlib.pyplot as plt


def pupil_detector(input, debug=False):
    """Detects and returns a single pupil candidate for a given image.

    Returns: A pupil candidate in OpenCV ellipse format.
    """
    _, thresh = cv.threshold(input, 55, 255, cv.THRESH_BINARY_INV)

    contours, _ = cv.findContours(thresh, cv.RETR_LIST,
                                  cv.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return [-1, -1, -1, -1, -1]

    contours = filter(lambda x: ratio(x) > 0.5, contours)

    best = sorted(contours,
                  key=lambda c: cv.contourArea(c),
                  reverse=True)[0]

    (cx, cy), (ax, ay), angle = cv.fitEllipse(best)
    if debug:
        return [cy, cx, ay, ax, angle], thresh
    return [cy, cx, ay, ax, angle]


def dist(a, b):
    return np.linalg.norm(a - b)


def contour_center(contour):
    moments = cv.moments(contour)
    if moments['m00'] == 0:
        cx, cy = 0, 0
    else:
        cx = moments['m10'] / moments['m00']
        cy = moments['m01'] / moments['m00']
    return np.array([cy, cx])


def circularity(contour):
    perimeter = cv.arcLength(contour, True)
    return np.sqrt(4 * np.pi * cv.contourArea(contour)) / (perimeter ** 2)


def ratio(contour):
    rect = cv.minAreaRect(contour)
    w, h = rect[1][0], rect[1][1]
    if w == 0 or h == 0:
        return 0
    return min(w, h) / max(w, h)


def find_glints(gray, center, *,
                radius=80,
                threshold=168,
                max_area=20,
                min_ratio=0.5,
                debug=False):
    """Detects and returns up to four glint candidates for a given image.

    Returns: Detected glint positions.
    """

    c = center

    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv.circle(mask, (int(c[1]), int(c[0])), radius, 255, -1)

    gray = cv.bitwise_and(gray, gray, mask=mask)

    _, thresh = cv.threshold(gray, threshold, 255, cv.THRESH_BINARY)

    kernel = cv.getStructuringElement(cv.MORPH_CROSS, (5, 5))
    thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)

    contours, _ = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    contours = list(filter(lambda cn: cv.contourArea(cn) < max_area, contours))
    contours = list(filter(lambda cn: ratio(cn) > min_ratio, contours))
    chosen = np.array([contour_center(c) for c in contours])
    chosen = sorted(chosen, key=lambda cn: dist(cn, c))[:4]
    # centers = np.array([contour_center(c) for c in contours])



    # centers = np.array(list(filter(lambda cn: dist(cn, c) < 100, centers)))[:4]
    if debug:
        return np.array(chosen), thresh
    else:
        return np.array(chosen)


def normalize_coordinates(coordinates, height, width):
    coordinates = np.copy(coordinates)
    coordinates[:, 0] /= height
    coordinates[:, 1] /= width
    return coordinates


def unnormalize_coordinates(coordinates, height, width):
    return normalize_coordinates(coordinates, 1 / height, 1 / width)
