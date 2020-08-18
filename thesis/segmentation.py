from dataclasses import dataclass

import cv2 as cv
import numpy as np
from skimage.filters import gabor

from thesis.geometry import Quadratic, Ellipse, Mask


@dataclass
class IrisSegmentation(Mask):
    inner: Ellipse
    outer: Ellipse
    upper_eyelid: Quadratic
    lower_eyelid: Quadratic

    @staticmethod
    def from_json(obj):
        print(len(obj['upper']))
        return IrisSegmentation(
            inner=Ellipse.from_points(obj['inner']),
            outer=Ellipse.from_points(obj['outer']),
            upper_eyelid=Quadratic.from_points_least_sq(obj['upper']),
            lower_eyelid=Quadratic.from_points_least_sq(obj['lower'])
        )

    def get_mask(self, size):
        mask_inner = self.inner.get_mask(size)
        mask_outer = self.outer.get_mask(size)
        mask_upper = self.upper_eyelid.get_mask(size)
        mask_lower = 1 - self.lower_eyelid.get_mask(size)

        base = mask_outer - mask_inner
        with_eyelids = base * mask_upper * mask_lower
        return with_eyelids

    def intersect_angle(self, theta):
        p1 = self.inner.intersect_angle(theta)
        p2 = self.outer.intersect_angle(theta)
        return p1, p2


@dataclass
class IrisImage:
    segmentation: IrisSegmentation
    mask: np.ndarray
    image: np.ndarray

    def __init__(self, segmentation: IrisSegmentation, image: np.ndarray):
        self.segmentation = segmentation
        if len(image.shape) > 2:
            self.image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        else:
            self.image = image
        self.mask = segmentation.get_mask((image.shape[1], image.shape[0]))

    def polar_image(self, angular_resolution, linear_resolution) -> np.ndarray:
        """Create polar image.

        Args:
            angular_resolution: Number of angular stops.
            linear_resolution: Number of stops from pupil to iris.

        Returns:

        """
        output = np.zeros((linear_resolution, angular_resolution), np.uint8)
        output_mask = np.zeros((linear_resolution, angular_resolution), np.uint8)

        angle_steps = np.linspace(0, 2 * np.pi, angular_resolution)
        for i, theta in enumerate(angle_steps):
            start, stop = self.segmentation.intersect_angle(theta)
            x_coord, y_coord = start.linear_interpolation(stop, linear_resolution)
            for j, (x, y) in enumerate(zip(x_coord, y_coord)):
                if x < 0 or y < 0 or x >= self.image.shape[1] or y >= self.image.shape[0]:
                    continue
                output[j, i] = self.image[int(y), int(x)]
                output_mask[j, i] = self.mask[int(y), int(x)]

        return output, output_mask


class IrisCode:
    def __init__(self, iris_image: IrisImage,
                 scales: int = 3,
                 angles: int = 3,
                 frequency_scale_base=1):
        polar, polar_mask = iris_image.polar_image(10, 10)

        res = []
        for s in range(1, scales + 1):
            frequency_scale = frequency_scale_base / 2 ** s
            for t in np.linspace(0, np.pi, angles):
                real, imag = gabor(polar, frequency_scale, theta=t, bandwidth=1)
                res.extend(real.reshape(-1))

        self.code = res