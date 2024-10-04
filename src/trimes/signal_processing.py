from typing import Union

import pandas as pd
import numpy as np
from numpy.typing import ArrayLike
from scipy.ndimage import convolve1d
from scipy.optimize import minimize_scalar
from icecream import ic

from trimes.fourier import get_fourier_coef_real


def extend(x: ArrayLike, n: int, mode: str) -> Union[pd.DataFrame, pd.Series]:
    if mode == "wrap":
        if x.ndim > 1:
            return np.concatenate((x[:n, :], x), axis=0)
        else:
            return np.concatenate((x[:n], x), axis=0)


def get_angle(x: ArrayLike, **kwargs) -> float:
    f = lambda y, args: -get_fourier_coef_real(args, angle=y)
    opt_res = minimize_scalar(f, args=(x), bounds=(-np.pi, np.pi), **kwargs)
    return opt_res.x
