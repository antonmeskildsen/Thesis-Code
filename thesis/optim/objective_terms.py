from abc import ABC, abstractmethod
from typing import Generic, TypeVar
from collections import Callable
from dataclasses import dataclass

import cv2 as cv
import numpy as np

from thesis.data import SegmentationSample, GazeImage
from thesis.information.entropy import gradient_histogram, histogram, entropy
from thesis.tracking.gaze import GazeModel
from thesis.tracking.features import normalize_coordinates


def bilateral_filter(img, kernel_size, sigma_c, sigma_s):
    return cv.bilateralFilter(img, kernel_size, sigma_c, sigma_s)


def gradient_entropy(image, mask):
    hist = gradient_histogram(image, mask)
    return entropy(hist)


def intensity_entropy(image, mask):
    hist = histogram(image, mask)
    return entropy(hist)


T = TypeVar('T')


class SegmentationTerm(ABC):
    def __init__(self):
        ...

    @abstractmethod
    def __call__(self, sample: T, filtered: np.ndarray) -> float:
        ...


class GazeTerm(ABC):
    def __init__(self):
        ...

    @abstractmethod
    def __call__(self, model: GazeModel, sample: T, filtered: np.ndarray) -> float:
        ...


class AbsoluteGradientEntropy(SegmentationTerm):
    def __call__(self, sample: SegmentationSample, filtered: np.ndarray) -> float:
        hist = gradient_histogram(filtered, sample.image.mask)
        return entropy(hist)


class RelativeGradientEntropy(SegmentationTerm):
    def __call__(self, sample: SegmentationSample, filtered: np.ndarray) -> float:
        hist_a = gradient_histogram(sample.image.image, sample.image.mask)
        hist_b = gradient_histogram(filtered, sample.image.mask)
        entropy_a = entropy(hist_a)
        entropy_b = entropy(hist_b)
        return entropy_b / entropy_a


class GazeAbsoluteAccuracy(GazeTerm):
    def __call__(self, model: GazeModel, sample: GazeImage, filtered: np.ndarray) -> float:
        gaze = model.predict(filtered)
        true = normalize_coordinates(np.array([sample.screen_position]), 2160, 3840)
        return np.linalg.norm(np.array(gaze) - np.array(true))


class GazeRelativeAccuracy(GazeTerm):
    def __call__(self, model: GazeModel, sample: GazeImage, filtered: np.ndarray) -> float:
        gaze_a = model.predict(sample.image)
        gaze_b = model.predict(filtered)
        screen = normalize_coordinates(np.array([sample.screen_position]), 2160, 3840)
        dist_a = np.linalg.norm(np.array(gaze_a) - np.array(screen))
        dist_b = np.linalg.norm(np.array(gaze_b) - np.array(screen))

        return dist_b / dist_a
