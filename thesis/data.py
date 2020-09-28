from __future__ import annotations
from dataclasses import dataclass
from typing import List
import os
import json

import numpy as np
import pandas as pd
import cv2 as cv

from thesis.geometry import Ellipse
from thesis.segmentation import IrisImage
from thesis.tracking.gaze import GazeModel, BasicGaze
from thesis.tracking.features import normalize_coordinates

from thesis.tools.st_utils import fit_else_ref, create_deepeye_func


@dataclass
class GazeImage:
    image: np.ndarray
    # pupil: Ellipse
    # glints: List[(float, float)]
    screen_position: (int, int)

    @staticmethod
    def from_json(path: str, data: dict):
        image = cv.imread(os.path.join(path, data['image']), cv.IMREAD_GRAYSCALE)
        # pupil = Ellipse.from_dict(data['pupil'])
        # glints = data['glints']
        screen_position = data['position']
        return GazeImage(image, screen_position)


@dataclass
class GazeDataset:
    name: str
    calibration_samples: List[GazeImage]
    test_samples: List[GazeImage]
    model: GazeModel

    @staticmethod
    def from_path(path: str):
        with open(os.path.join(path, 'data.json')) as file:
            data = json.load(file)
            calibration_samples = list(map(lambda d: GazeImage.from_json(path, d), data['calibration']))
            test_samples = list(map(lambda d: GazeImage.from_json(path, d), data['test']))

            model = BasicGaze(pupil_detector=fit_else_ref)
            images = [s.image for s in calibration_samples]
            gaze_positions = [s.screen_position for s in calibration_samples]
            model.calibrate(images, gaze_positions)

            if 'name' in data:
                name = data['name']
            else:
                name = 'unnamed'

            # print(normalize_coordinates(gaze_positions, 2160, 3840))
            # print(model.predict(images))

            return GazeDataset(name, calibration_samples, test_samples, model)

    def __repr__(self):
        return f'calibration samples: {len(self.calibration_samples)}, test samples: {len(self.test_samples)}'


@dataclass
class SegmentationSample:
    image: IrisImage
    user_id: str
    eye: str
    image_id: str
    session_id: str

    @staticmethod
    def from_dict(data: dict):
        image = IrisImage.from_dict(data)
        info = data['info']
        return SegmentationSample(image, **info)


@dataclass
class SegmentationDataset:
    name: str
    samples: List[SegmentationSample]

    @staticmethod
    def from_path(path: str) -> SegmentationDataset:
        with open(path) as file:
            data = json.load(file)
            images = map(SegmentationSample.from_dict, data['data'])

            if 'name' in data:
                name = data['name']
            else:
                name = 'unnamed'

            return SegmentationDataset(name, list(images))


@dataclass
class PupilSample:
    image: np.ndarray
    center: (int, int)

    @staticmethod
    def from_json(path: str, data: dict):
        image = cv.imread(os.path.join(path, data['image']), cv.IMREAD_GRAYSCALE)
        screen_position = data['position']
        return GazeImage(image, screen_position)


@dataclass
class PupilDataset:
    name: str
    samples: List[PupilSample]

    @staticmethod
    def from_path(path: str) -> SegmentationDataset:
        with open(path) as file:
            data = json.load(file)
            images = map(SegmentationSample.from_dict, data['data'])

            if 'name' in data:
                name = data['name']
            else:
                name = 'unnamed'

            return SegmentationDataset(name, list(images))
