from __future__ import annotations
from dataclasses import dataclass
from typing import List
import json

import cv2 as cv
import numpy as np
from skimage.filters import gabor, gabor_kernel
from scipy import ndimage as ndi

from numba import jit

from thesis.geometry import Quadratic, Ellipse, Mask, Vec2


@jit(nopython=True)
def radius_at_angle(ellipse, theta):
    _, (a, b), angle = ellipse
    return (a * b) / np.sqrt(a ** 2 * np.sin(theta - angle) ** 2 + b ** 2 * np.cos(theta - angle) ** 2)


@jit(nopython=True)
def intersect_angle(ellipse, theta):
    (cx, cy), _, _ = ellipse
    r = radius_at_angle(ellipse, theta)
    return cx + r * np.sin(theta), cy + r * np.cos(theta)


@jit(nopython=True)
def linear_interpolation(start: (float, float), stop: (float, float), num: int) -> (np.ndarray, np.ndarray):
    return np.linspace(start[0], stop[0], num), np.linspace(start[1], stop[1], num)


@jit(nopython=True, parallel=True)
def polar_from_ellipses(img: np.ndarray, mask: np.ndarray, size: (int, int), inner, outer, start_angle=0) -> (
        np.ndarray, np.ndarray):
    """Create polar image.

    Args:
        img: input image
        mask: image mask
        size: (angular resolution, radial resolution)
        inner: inner boundary ellipse
        outer: outer boundary ellipse
        start_angle:

    Returns:

    """
    radial_resolution, angular_resolution = size
    output = np.zeros((radial_resolution, angular_resolution), np.uint8)
    output_mask = np.zeros((radial_resolution, angular_resolution), np.uint8)

    angle_steps = np.linspace(start_angle, start_angle + 2 * np.pi, angular_resolution)
    for i, theta in enumerate(angle_steps):
        start = intersect_angle(inner, theta)
        stop = intersect_angle(outer, theta)
        margin = radial_resolution // 4
        x_coord, y_coord = linear_interpolation(start, stop, radial_resolution + margin * 2)
        for j, (x, y) in enumerate(zip(x_coord[margin:][:-margin], y_coord[margin:][:-margin])):
            if x < 0 or y < 0 or x >= img.shape[1] or y >= img.shape[0]:
                continue
            output[j, i] = img[int(y), int(x)]
            output_mask[j, i] = mask[int(y), int(x)]

    return output, output_mask


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

        self.polar = None
        self.saved_angular_resolution = 0
        self.saved_radial_resolution = 0

    @staticmethod
    def from_dict(data: dict) -> IrisImage:
        segmentation = IrisSegmentation.from_dict(data['points'])
        image = cv.imread(data['image'])
        return IrisImage(segmentation, image)

    def to_polar(self, angular_resolution, radial_resolution, start_angle=0) -> (np.ndarray, np.ndarray):
        """Create polar image.

        Args:
            angular_resolution: Number of angular stops.
            radial_resolution: Number of stops from pupil to iris.
            start_angle:

        Returns:

        """
        if self.saved_angular_resolution != angular_resolution \
                or self.saved_radial_resolution != radial_resolution \
                or self.polar is None:
            self.polar = polar_from_ellipses(self.image, self.mask, (radial_resolution, angular_resolution),
                                             self.segmentation.inner.as_tuple(), self.segmentation.outer.as_tuple(),
                                             start_angle)

        return self.polar


@dataclass
class IrisCode:
    code: np.ndarray
    mask: np.ndarray

    def dist(self, other):
        mask = self.mask | other.mask
        n = mask.sum()
        if n == len(self):
            return 1
        else:
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
                # kernel = cv.getGaborKernel((k, k), sigma, theta=t + np.pi/4, lambd=wavelength, gamma=1, psi=np.pi * 0.5,
                #                            ktype=cv.CV_64F)
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


class SKImageIrisCodeEncoder:
    def __init__(self, angles: int = 3,
                 angular_resolution=20,
                 radial_resolution=10,
                 scales=6,
                 eps=0.01):
        self.angular_resolution = angular_resolution
        self.radial_resolution = radial_resolution
        self.eps = eps
        self.scales = scales
        self.kernels = []
        for theta in range(0, angles):
            a = theta / angles * np.pi / 2
            kernel = gabor_kernel(1 / 3, a, bandwidth=1)
            self.kernels.append(kernel)

    def encode_raw(self, polar, polar_mask):
        polar = np.float64(polar) / 255

        pyramid = [(polar, polar_mask)]
        for _ in range(self.scales):
            p_next = cv.pyrDown(polar)
            m_next = cv.resize(polar_mask, (p_next.shape[1], p_next.shape[0]), cv.INTER_NEAREST)
            pyramid.append((p_next, m_next))

        res = []
        mask = []
        for k in self.kernels:
            for pl, pmask in pyramid:
                # f_real = ndi.convolve(pl, k.real, mode='wrap')
                # f_imag = ndi.convolve(pl, k.imag, mode='wrap')
                f_real = cv.filter2D(pl, cv.CV_64F, k.real)
                f_imag = cv.filter2D(pl, cv.CV_64F, k.imag)
                m_real = np.zeros(f_real.shape, np.uint8)

                f_complex = np.dstack((f_real, f_imag)).view(dtype=np.complex128)[:, :, 0]
                # print(f_complex.shape)

                m_real[np.abs(f_complex) < self.eps] = 1
                f_real = np.sign(f_real)
                m_real[pmask == 0] = 1
                res.extend(f_real.reshape(-1))
                mask.extend(m_real.reshape(-1))

                m_imag = np.zeros(f_imag.shape, np.uint8)
                m_imag[np.abs(f_complex) < self.eps] = 1
                f_imag = np.sign(f_imag)
                m_imag[pmask == 0] = 1
                res.extend(f_imag.reshape(-1))
                mask.extend(m_imag.reshape(-1))

        return IrisCode(np.array(res), np.array(mask))

    def encode(self, image, start_angle=0):
        polar, polar_mask = image.to_polar(self.angular_resolution, self.radial_resolution, start_angle)
        return self.encode_raw(polar, polar_mask)
