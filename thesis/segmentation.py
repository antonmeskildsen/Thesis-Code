from dataclasses import dataclass

import cv2 as cv
import numpy as np

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
        self.image = image
        self.mask = segmentation.get_mask((image.shape[1], image.shape[0]))

    def polar_image(self, angular_resolution, linear_resolution) -> np.ndarray:
        """Create polar image.

        Args:
            angular_resolution: Number of angular stops.
            linear_resolution: Number of stops from pupil to iris.

        Returns:

        """
        if len(self.image.shape) == 2:
            output = np.zeros((linear_resolution, angular_resolution), np.uint8)
        else:
            output = np.zeros((linear_resolution, angular_resolution, self.image.shape[2]), np.uint8)

        output_mask = np.zeros((linear_resolution, angular_resolution), np.uint8)

        angle_steps = np.linspace(0, 2 * np.pi, angular_resolution)
        for i, theta in enumerate(angle_steps):
            start, stop = self.segmentation.intersect_angle(theta)
            x_coord, y_coord = start.linear_interpolation(stop, linear_resolution)
            for j, (x, y) in enumerate(zip(x_coord, y_coord)):
                output[j, i] = self.image[int(y), int(x)]
                output_mask[j, i] = self.mask[int(y), int(x)]

        return output, output_mask
