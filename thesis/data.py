from __future__ import annotations
from dataclasses import dataclass
from typing import List
import os
import json

import numpy as np
import cv2 as cv

from thesis.geometry import Ellipse
from thesis.segmentation import IrisImage
from thesis.tracking.gaze import GazeModel, BasicGaze


@dataclass
class GazeImage:
    image: np.ndarray
    pupil: Ellipse
    glints: List[(float, float)]
    screen_position: (int, int)

    @staticmethod
    def from_json(path: str, data: dict):
        image = cv.imread(os.path.join(path, data['image']), cv.IMREAD_GRAYSCALE)
        pupil = Ellipse.from_dict(data['pupil'])
        glints = data['glints']
        screen_position = data['position']
        return GazeImage(image, pupil, glints, screen_position)


@dataclass
class GazeDataset:
    calibration_samples: List[GazeImage]
    test_samples: List[GazeImage]
    model: GazeModel

    @staticmethod
    def from_path(path: str):
        with open(os.path.join(path, 'data.json')) as file:
            data = json.load(file)
            calibration_samples = list(map(lambda d: GazeImage.from_json(path, d), data['calibration']))
            test_samples = list(map(lambda d: GazeImage.from_json(path, d), data['test']))

            model = BasicGaze()
            images = [s.image for s in calibration_samples]
            gaze_positions = [s.screen_position for s in calibration_samples]
            model.calibrate(images, gaze_positions)

            return GazeDataset(calibration_samples, test_samples, model)


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