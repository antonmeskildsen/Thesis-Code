from functools import reduce
from typing import Any, Callable, Dict, List, NewType
from dataclasses import dataclass

import cv2 as cv
import numpy as np
from typing_extensions import Protocol

from thesis.optim_old.types import *
from thesis.information.entropy import grad_entropy, grad_hist
from thesis.fnc.features import extract_feature
from eyelab.fnc.matching import calc_hamming_dist


def _detect(algo, p, t):
    c1, s1, a1 = algo(p)
    c2, s2, a2 = algo(t)
    b1 = np.zeros(p.shape, dtype=np.uint8)
    b2 = np.zeros(p.shape, dtype=np.unit8)
    cv.ellipse(b1, (int(c1[0]), int(c1[1])),
               (int(s1[0] // 2), int(s1[1] // 2)), int(a1), 0, 360, 1, -1)
    cv.ellipse(b2, (int(c2[0]), int(c2[1])),
               (int(s2[0] // 2), int(s2[1] // 2)), int(a2), 0, 360, 1, -1)
    return b1, b2, c2


def distance(c: float) -> LossTerm:
    def inner(p: Array, t: Array, **kwargs) -> Array:
        return c * np.linalg.norm(p - t, ord=2)

    return inner


def utility(algo: FeaturePredictor, gamma: float) -> LossTerm:
    def inner(p, t, w, **kwargs) -> Array:
        b1, b2, c2 = _detect(algo, p, t)

        union = b1 + b2
        union[union == 2] = 1

        intersect = b1 * b2

        # Precision of first guess (how usable is the result)
        dist = np.sqrt((w['cx'] - c2[0])**2 + (w['cy'] - c2[1])**2)
        #print('dist', dist)
        frac = dist / max(p.shape)
        modifier = (1 - frac)**gamma

        res = modifier * (1 - intersect.sum() / union.sum())
        #print(res)
        return res

    return inner


def unsupervised_utility(algo: FeaturePredictor) -> LossTerm:
    def inner(p, t, **kwargs):
        b1, b2, _ = _detect(algo, p, t)

        union = b1 | b2
        intersect = b1 * b2

        return 1 - intersect.sum()/union.sum()

    return inner


def entropy2d() -> LossTerm:
    def inner(p, **kwargs) -> Array:
        hist = grad_hist(p)
        return grad_entropy(hist)
    return inner


def parameter_loss(gamma: float) -> LossTerm:
    def inner(ksize, **kwargs) -> Array:
        return ksize**gamma
    return inner


@dataclass
class LossFunction:

    terms: List[LossTerm]
    weights: List[float]

    def __call__(self, **kwargs):
        """Calculate loss

        :param p: prediction
        :param t: target

        :returns: the sum of the specified loss terms.
        """
        def reducer(total, item):
            f, weight = item
            res = weight * f(**kwargs)
            return total + res

        return reduce(reducer, zip(self.terms, self.weights), np.array([0]))
