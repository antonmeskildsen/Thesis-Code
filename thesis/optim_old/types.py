from typing import Any, Callable, Dict, NewType, Union, Tuple, TypeVar
from eyelab._c_ext import Mat

import numpy as np
import torch
from typing_extensions import Protocol

Array = np.ndarray

LossTerm = Callable[[Dict[str, Any]], Array]
LossTermInitializer = Callable[..., LossTerm]

T = TypeVar('T')
Pair = Tuple[T, T]
FeaturePredictor = Callable[[Mat], Tuple[Pair[float], Pair[float], float]]