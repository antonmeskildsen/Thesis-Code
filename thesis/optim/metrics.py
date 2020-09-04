from thesis.data import GazeImage
from thesis.tracking.gaze import GazeModel

import numpy as np


def gaze(model: GazeModel, sample: GazeImage, filtered: np.ndarray) -> float:
    prediction = model.predict([filtered])
    # Comparison
    return 0.


def pupil(detector, image: np.ndarray) -> float:
    return 0.


def entropy(image: np.ndarray) -> float:
    return 0.
