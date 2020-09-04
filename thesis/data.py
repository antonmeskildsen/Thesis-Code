from __future__ import annotations
from dataclasses import dataclass
from typing import List
import os
import json

import numpy as np
import cv2 as cv

from thesis.geometry import Ellipse
from thesis.segmentation import IrisImage


@dataclass
class GazeImage:
    image: np.ndarray
    pupil: Ellipse
    glints: List[(float, float)]
    screen_position: (int, int)

    @staticmethod
    def from_json(path: str, data: dict):
        image = cv.imread(os.path.join(path, data['image']))
        pupil = Ellipse.from_dict(data['pupil'])
        glints = data['glints']
        screen_position = data['position']
        return GazeImage(image, pupil, glints, screen_position)


@dataclass
class GazeDataset:
    calibration_samples: List[GazeImage]
    test_samples: List[GazeImage]

    @staticmethod
    def from_path(path: str):
        with open(os.path.join(path, 'data.json')) as file:
            data = json.load(file)
            calibration_samples = map(lambda d: GazeImage.from_json(path, d), data['calibration'])
            test_samples = map(lambda d: GazeImage.from_json(path, d), data['test'])
            GazeDataset(list(calibration_samples), list(test_samples))


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
    samples: List[SegmentationSample]

    @staticmethod
    def from_path(path: str) -> SegmentationDataset:
        with open(path) as file:
            data = json.load(file)
            images = map(SegmentationSample.from_dict, data['data'])
            return SegmentationDataset(list(images))