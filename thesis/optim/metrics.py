from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from itertools import chain
from typing import Callable, List

import cv2 as cv
import numpy as np
from pupilfit import fit_else, fit_excuse

from thesis.data import GazeImage, PupilSample
from thesis.entropy import joint_gradient_histogram, entropy, mutual_information_grad, joint_gabor_histogram
from thesis.tracking.gaze import GazeModel
from thesis.tracking.features import normalize_coordinates, pupil_detector
from thesis.segmentation import SKImageIrisCodeEncoder

import sys

this = sys.modules[__name__]


class Logger:

    def __init__(self):
        self.data = defaultdict(list)

    def add(self, point: str, value: float):
        self.data[point].append(value)

    def columns(self):
        return list(self.data.keys())

    def means(self):
        return list(map(np.mean, self.data.values()))


class IrisMetric(ABC):
    @property
    def columns(self):
        ...

    @abstractmethod
    def log(self, results: Logger, polar_image, polar_filtered, mask):
        ...


class GazeMetric(ABC):
    @property
    def columns(self):
        ...

    @abstractmethod
    def log(self, results: Logger, model: GazeModel, sample: GazeImage, filtered: np.ndarray):
        ...


class PupilMetric(ABC):
    @property
    def columns(self):
        ...

    @abstractmethod
    def log(self, results: Logger, pupil_sample: PupilSample, filtered: np.ndarray):
        ...


class GradientEntropy(IrisMetric):
    columns = ['gradient_entropy_source', 'gradient_entropy_filtered', 'gradient_mutual_information']

    def __init__(self, histogram_divisions):
        self.histogram_divisions = histogram_divisions

    def log(self, results: Logger, polar_image, polar_filtered, mask):
        hist_source, hist_filtered, hist_joint = joint_gradient_histogram(polar_image, polar_filtered, mask,
                                                                          self.histogram_divisions)
        entropy_source = entropy(hist_source)
        entropy_filtered = entropy(hist_filtered)
        mutual_information = mutual_information_grad(hist_source, hist_filtered, hist_joint)

        results.add('gradient_entropy_source', entropy_source)
        results.add('gradient_entropy_filtered', entropy_filtered)
        results.add('gradient_mutual_information', mutual_information)


class GaborEntropy(IrisMetric):
    def __init__(self, scales, angles_per_scale, histogram_divisions):
        self.scales = scales
        self.angles_per_scale = angles_per_scale
        self.histogram_divisions = histogram_divisions

        self._columns = list(chain.from_iterable([
            [f'gabor_entropy_source_{1 / 2**scale}x', f'gabor_entropy_filtered_{1 / 2**scale}x',
             f'gabor_mutual_information_{1 / 2**scale}x']
            for scale in range(self.scales)
        ]))

    @property
    def columns(self):
        return self._columns

    def log(self, results: Logger, polar_image, polar_filtered, mask):
        angles = np.linspace(0, np.pi - np.pi/self.angles_per_scale, self.angles_per_scale)  # TODO: Consider subtracting small amount
        for scale in range(self.scales):
            for theta in angles:
                hist_source, hist_filtered, hist_joint = joint_gabor_histogram(polar_image, polar_filtered, mask,
                                                                               theta, self.histogram_divisions)

                entropy_source = entropy(hist_source)
                entropy_filtered = entropy(hist_filtered)
                mutual_information = mutual_information_grad(hist_source, hist_filtered, hist_joint)

                results.add(f'gabor_entropy_source_{1 / 2**scale}x', entropy_source)
                results.add(f'gabor_entropy_filtered_{1 / 2**scale}x', entropy_filtered)
                results.add(f'gabor_mutual_information_{1 / 2**scale}x', mutual_information)

            polar_image = cv.pyrDown(polar_image)
            polar_filtered = cv.pyrDown(polar_filtered)
            mask = cv.resize(mask, (polar_image.shape[1], polar_image.shape[0]), interpolation=cv.INTER_NEAREST)


class GazeAccuracy(GazeMetric):
    columns = ['gaze_angle_error_source', 'gaze_angle_error_filtered']

    def log(self, results: Logger, model: GazeModel, sample: GazeImage, filtered: np.ndarray):
        gaze_source = model.predict(sample.image)
        gaze_filtered = model.predict(filtered)
        screen = normalize_coordinates(np.array([sample.screen_position]), model.screen_height,
                                       model.screen_width)

        dist_source = np.linalg.norm(np.array(gaze_source) - np.array(screen))
        dist_filtered = np.linalg.norm(np.array(gaze_filtered) - np.array(screen))

        angle_error_source = model.fov * dist_source
        angle_error_filtered = model.fov * dist_filtered

        results.add('gaze_angle_error_source', angle_error_source)
        results.add('gaze_angle_error_filtered', angle_error_filtered)


class PupilDetector(ABC):
    name = ...

    @abstractmethod
    def __call__(self, image: np.ndarray) -> (float, float):
        ...


class BaseDetector(PupilDetector):
    name = 'base'

    def __call__(self, image: np.ndarray) -> (float, float):
        y, x = pupil_detector(image)[:2]
        return x, y


class ElseDetector(PupilDetector):
    name = 'base'

    def __call__(self, image: np.ndarray) -> (float, float):
        return fit_else(image)[0]


class ExcuseDetector(PupilDetector):
    name = 'base'

    def __call__(self, image: np.ndarray) -> (float, float):
        return fit_excuse(image)[0]


class PupilDetectionError(PupilMetric):
    def __init__(self, detectors: List[type(PupilDetector)]):
        self.detectors = [getattr(this, x)() for x in detectors]
        self._columns = list(chain.from_iterable([
            [f'pupil_distance_{d.name}_pixel_error_source', f'pupil_distance_{d.name}_pixel_error_filtered']
            for d in self.detectors
        ]))

    @property
    def columns(self):
        return self._columns

    def log(self, results: Logger, pupil_sample: PupilSample, filtered: np.ndarray):
        for d in self.detectors:
            predicted_unmodified = d(pupil_sample.image)
            predicted_filtered = d(filtered)
            dist_unmodified = np.linalg.norm(np.array(predicted_unmodified) - np.array(pupil_sample.center))
            dist_filtered = np.linalg.norm(np.array(predicted_filtered) - np.array(pupil_sample.center))
            results.add(f'pupil_distance_{d.name}_pixel_error_source', dist_unmodified)
            results.add(f'pupil_distance_{d.name}_pixel_error_filtered', dist_filtered)


class ImageSimilarity(IrisMetric):
    columns = ['iris_code_distance', 'image_norm_distance']

    def __init__(self, angles, scales, eps):
        self.encoder = SKImageIrisCodeEncoder(angles, -1, -1, scales, eps)

    def log(self, results: Logger, polar_image, polar_filtered, mask):
        code_source = self.encoder.encode_raw(polar_image, mask)
        code_filtered = self.encoder.encode_raw(polar_filtered, mask)
        iris_code_similarity = 1 - code_source.dist(code_filtered)
        results.add('iris_code_similarity', iris_code_similarity)

        source_masked = polar_image * mask
        source_masked = source_masked / np.linalg.norm(source_masked)
        filtered_masked = polar_filtered * mask
        norm = np.linalg.norm(filtered_masked)
        if norm == 0:
            return np.nan
        else:
            filtered_masked = filtered_masked / norm
            similarity = 1 - np.linalg.norm(source_masked - filtered_masked)
            results.add('image_normalized_similarity', similarity)