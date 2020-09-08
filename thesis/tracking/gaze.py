from abc import ABC, abstractmethod
from typing import Tuple, List
from typing_extensions import Protocol
import numpy as np
import cv2 as cv

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

from thesis.tracking import features


class FeatureTransformer(Protocol):
    @abstractmethod
    def fit_transform(self, X, y=None):
        ...


class Model(Protocol):
    @abstractmethod
    def fit(self, X, y=None):
        ...


class GazeModel:
    @abstractmethod
    def calibrate(self, images, gaze_positions):
        """Calibrate the model.

        Args:
            samples:
        """
        ...

    @abstractmethod
    def predict(self, images):
        """Predict gaze from an input image.

        Args:
            images (array-like): Single image or list of images to predict.

        Returns:
            Tuple[float, float]: ...
        """
        ...


def solve_similarity(source, target):
    mats = []
    for p in source:
        mats.append(np.array([
            [p[0], -p[1], 1, 0],
            [p[1], p[0], 0, 1]
        ]))

    mat = np.vstack(mats)

    b = target.reshape(-1)

    s: np.ndarray = np.linalg.inv(mat).dot(b)
    solution = np.array([
        [s[0], -s[1], s[2]],
        [s[1], s[0], s[3]],
        [0., 0., 1.]
    ])
    return solution


def solve_affine(source, target):
    mats = []
    for p in source:
        mats.append(np.array([
            [p[0], p[1], 1, 0, 0, 0],
            [0, 0, 0, p[0], p[1], 1]
        ]))

    mat = np.vstack(mats)

    b = target.reshape(-1)

    solution = np.linalg.inv(mat).dot(b)
    solution = solution.reshape(2, 3)
    solution = np.vstack((solution, [0, 0, 1]))
    return solution


def solve_homography(source, target):
    hom, mask = cv.findHomography(source, target, 0)
    return hom


def hom(points):
    points = np.array(points, dtype=np.float)
    if points.ndim == 1:
        points = points.reshape((points.shape[0], 1))
    z = np.ones((points.shape[1], 1))
    r = np.vstack((points, z))
    return r


class BasicGaze(GazeModel):
    def __init__(self, glint_args=None, model: Pipeline = None):
        if glint_args is None:
            self.glint_args = {}
        else:
            self.glint_args = glint_args
        if model is None:
            model = Pipeline([
                ('design matrix', PolynomialFeatures(1)),
                ('model', LinearRegression())
            ])

        super().__init__()
        self.model = model

    def _preprocess(self, images):
        images = np.array(images)
        if len(images.shape) == 2:
            images = [images]

        pupils = np.array([features.pupil_detector(img) for img in images])

        # print(pupils[0])
        centers = [[p[0], p[1]] for p in pupils]
        glints = [
            features.find_glints(img, center, **self.glint_args)
            for img, center in zip(images, centers)
        ]

        # nan_removed = np.array([g for g in glints if ~np.isnan(g).any()])
        # avg = nan_removed.mean(axis=0)
        normed = []
        for i, (c, g) in enumerate(zip(centers, glints)):
            # print(i)
            normed.append([c[0] - g[0, 0], c[1] - g[0, 1]])

        # print(normed)

        scale_x = 1
        scale_y = 1

        # for c, g in zip(centers, glints):
        #     if len(g) > 0:
        #         p = (c - g[0]) / np.linalg.norm(c - g[0])
        #
        #     if len(glints) > 1:
        #         a = g[0]
        #         b = g[1]
        #
        #         vec = (a - b) / np.linalg.norm(a - b)
        #         if vec.dot([0, 1]) < vec.dot([1, 0]):
        #             arr = np.array([[0, 0], [0, 1]])
        #             scale_x
        #         else:
        #             arr = np.array([[0, 0], [1, 0]])
        #
        #         sim = solve_similarity(g[:2], arr)
        #         p = sim.dot(hom(c))
        #
        #     if len(glints) > 2:
        #         scale_x = np.linalg.norm(g[0] - g[1])
        #         scale_y = np.linalg.norm(g[1] - g[2])
        #         aff = solve_affine(glints[:3], np.array([[0, 0], [1, 0], [1, 1]]))
        #         # st.write(aff)
        #         p = aff.dot(hom(p))

        # if np.isnan(g).any():
        #     normed.append(c - avg)
        # else:
        #     normed.append(c - g)
        # print(centers)

        return np.array(normed)

    def calibrate(self, images, gaze_positions):
        pupil = self._preprocess(images)
        norm_pos = features.normalize_coordinates(gaze_positions, 2160, 3840)
        self.model.fit(pupil, norm_pos)

    def predict(self, images):
        pupil = self._preprocess(images)
        norm_predict = self.model.predict(pupil)
        return norm_predict  # features.unnormalize_coordinates(norm_predict, 2160, 3840)

    def score(self, images, gaze_positions):
        pupil = self._preprocess(images)
        norm_pos = features.normalize_coordinates(gaze_positions, 2160, 3840)
        return self.model.score(pupil, norm_pos)
