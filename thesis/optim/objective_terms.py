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


class Term(ABC, Generic[T]):
    @abstractmethod
    def __call__(self, sample: T, filtered: np.ndarray) -> float:
        ...


@dataclass
class AbsoluteEntropy(Term[SegmentationSample]):
    histogram_func: Callable

    def __call__(self, sample: SegmentationSample, filtered: np.ndarray) -> float:
        hist = self.histogram_func(filtered, sample.image.mask)
        return entropy(hist)


@dataclass
class RelativeEntropy(Term[SegmentationSample]):
    histogram_func: Callable

    def __call__(self, sample: SegmentationSample, filtered: np.ndarray) -> float:
        hist_a = self.histogram_func(sample.image.image, sample.image.mask)
        hist_b = self.histogram_func(filtered, sample.image.mask)
        entropy_a = entropy(hist_a)
        entropy_b = entropy(hist_b)
        return entropy_b / entropy_a


@dataclass
class GazeAbsoluteAccuracy(Term[GazeImage]):
    model: GazeModel

    def __call__(self, sample: GazeImage, filtered: np.ndarray) -> float:
        gaze = self.model.predict(filtered)
        true = normalize_coordinates(np.array([sample.screen_position]), 2160, 3840)
        return np.linalg.norm(np.array(gaze) - np.array(true))


@dataclass
class GazeRelativeAccuracy(Term[GazeImage]):
    model: GazeModel

    def __call__(self, sample: GazeImage, filtered: np.ndarray) -> float:
        gaze_a = self.model.predict(sample.image)
        gaze_b = self.model.predict(filtered)
        dist_a = np.linalg.norm(np.array(gaze_a) - np.array(sample.screen_position))
        dist_b = np.linalg.norm(np.array(gaze_b) - np.array(sample.screen_position))

        return dist_b / dist_a
