from __future__ import annotations
from dataclasses import dataclass
from typing import List
import json

import cv2 as cv
import numpy as np
from skimage.filters import gabor

from thesis.geometry import Quadratic, Ellipse, Mask, Vec2


@dataclass
class IrisSegmentation(Mask):
    inner: Ellipse
    outer: Ellipse
    upper_eyelid: Quadratic
    lower_eyelid: Quadratic

    @staticmethod
    def from_dict(obj: dict) -> IrisSegmentation:
        return IrisSegmentation(
            inner=Ellipse.from_points(obj['inner']),
            outer=Ellipse.from_points(obj['outer']),
            upper_eyelid=Quadratic.from_points_least_sq(obj['upper']),
            lower_eyelid=Quadratic.from_points_least_sq(obj['lower'])
        )

    def get_mask(self, size: (int, int)) -> np.ndarray:
        mask_inner = self.inner.get_mask(size)
        mask_outer = self.outer.get_mask(size)
        mask_upper = self.upper_eyelid.get_mask(size)
        mask_lower = 1 - self.lower_eyelid.get_mask(size)

        base = mask_outer - mask_inner
        with_eyelids = base * mask_upper * mask_lower
        return with_eyelids

    def intersect_angle(self, theta: float) -> (Vec2, Vec2):
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

    @staticmethod
    def from_dict(data: dict) -> IrisImage:
        segmentation = IrisSegmentation.from_dict(data['points'])
        image = cv.imread(data['image'])
        return IrisImage(segmentation, image)

    def to_polar(self, angular_resolution, linear_resolution, start_angle=0) -> np.ndarray:
        """Create polar image.

        Args:
            angular_resolution: Number of angular stops.
            linear_resolution: Number of stops from pupil to iris.
            start_angle:

        Returns:

        """
        output = np.zeros((linear_resolution, angular_resolution), np.uint8)
        output_mask = np.zeros((linear_resolution, angular_resolution), np.uint8)

        angle_steps = np.linspace(start_angle, start_angle + 2 * np.pi, angular_resolution)
        for i, theta in enumerate(angle_steps):
            start, stop = self.segmentation.intersect_angle(theta)
            x_coord, y_coord = start.linear_interpolation(stop, linear_resolution)
            for j, (x, y) in enumerate(zip(x_coord, y_coord)):
                if x < 0 or y < 0 or x >= self.image.shape[1] or y >= self.image.shape[0]:
                    continue
                output[j, i] = self.image[int(y), int(x)]
                output_mask[j, i] = self.mask[int(y), int(x)]

        return output, output_mask


@dataclass
class IrisCode:
    code: np.ndarray
    mask: np.ndarray

    # def __init__(self, iris_image: IrisImage,
    #              scales: int = 3,
    #              angles: int = 3,
    #              frequency_scale_base=2):
    #     polar, polar_mask = iris_image.to_polar(20, 5)
    #     polar = np.float64(polar)
    #     res = []
    #     for s in range(1, scales + 1):
    #         frequency_scale = frequency_scale_base / 2 ** s
    #         for t in np.linspace(0, np.pi - (np.pi / angles), angles):
    #             real, imag = gabor(polar, frequency_scale, theta=t, bandwidth=0.1, mode='wrap')
    #             real = np.sign(real)
    #             imag = np.sign(imag)
    #             real[polar_mask == 0] = 0
    #             imag[polar_mask == 0] = 0
    #             res.extend(real.reshape(-1))
    #             res.extend(imag.reshape(-1))
    #
    #     self.code = np.array(res)

    def dist(self, other):
        mask = self.mask | other.mask
        n = mask.sum()
        return (self.code != other.code)[mask == 0].sum() / (len(self) - n)

    def __len__(self):
        return self.code.size

    def shift(self, n_bits):
        return IrisCode(np.concatenate((self.code[n_bits:], self.code[:n_bits])),
                        np.concatenate((self.mask[n_bits:], self.mask[:n_bits])))

    def masked_image(self):
        c = np.array(self.code)
        c[self.mask == 1] = 0
        c = c / 2 + 0.5
        return c


class IrisCodeEncoder:
    def __init__(self, scales: int = 3,
                 angles: int = 3,
                 angular_resolution=20,
                 radial_resolution=10,
                 wavelength_base=0.5,
                 mult=1.41,
                 eps=0.01):
        self.kernels = []
        self.angular_resolution = angular_resolution
        self.radial_resolution = radial_resolution
        self.eps = eps
        wavelength = wavelength_base
        for s in range(scales):
            sigma = wavelength / 0.5
            k = max(3, int(sigma // 2 * 2 + 1))
            # print(sigma, k)
            for t in np.pi / np.arange(1, angles + 1):
                kernel = cv.getGaborKernel((k, k), sigma, theta=t, lambd=wavelength, gamma=1, psi=np.pi * 0.5,
                                           ktype=cv.CV_64F)
                self.kernels.append(kernel)
                # kernel = cv.getGaborKernel((5, 5), sigma=frequency_scale, theta=t + np.pi / 2, lambd=frequency_scale, gamma=1,
                #                            psi=np.pi * 0.5, ktype=cv.CV_64F)
                # self.kernels.append(kernel)
            wavelength *= mult

    def encode(self, image, start_angle=0):
        polar, polar_mask = image.to_polar(self.angular_resolution, self.radial_resolution, start_angle)
        polar = cv.equalizeHist(polar)
        polar = np.float64(polar)
        res = []
        mask = []
        for k in self.kernels:
            f = cv.filter2D(polar, cv.CV_64F, k)
            m = np.zeros(f.shape, np.uint8)
            m[np.abs(f) < self.eps] = 1
            f = np.sign(f)
            m[polar_mask == 0] = 1
            res.extend(f.reshape(-1))
            mask.extend(m.reshape(-1))

        return IrisCode(np.array(res), np.array(mask))
