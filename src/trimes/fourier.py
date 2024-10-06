from typing import Union

import pandas as pd
import numpy as np
from numpy.typing import ArrayLike
from icecream import ic


def get_fourier_coef_real(x: ArrayLike, k: int = 1, angle: float = 0.0):
    cos = np.cos(
        np.linspace(angle, k * 2 * np.pi + angle - 2 * np.pi / len(x), k * len(x))
    )
    return np.mean(cos * x)


def get_fourier_coef_imag(x: ArrayLike, k: int = 1, angle: float = 0.0):
    sin = np.sin(
        np.linspace(angle, k * 2 * np.pi + angle - 2 * np.pi / len(x), k * len(x))
    )
    return np.mean(sin * x)
