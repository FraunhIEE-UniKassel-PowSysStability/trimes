from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
from icecream import ic


def transform_time_series(ts: pd.DataFrame, transform: Callable) -> pd.DataFrame:
    """Apply transformation to time series.

    Args:
        ts (pd.DataFrame): original
        transform (Callable): transformation

    Returns:
        pd.DataFrame: Transformed time series
    """
    ts_transformed = np.reshape(
        np.concatenate(ts.apply(transform, axis=1).to_numpy()), ts.shape
    )
    return pd.DataFrame(ts_transformed, index=ts.index)


def abc_2_dq0(abc: np.array, phi: float) -> np.array:
    """Park transform (3-phase, amplitude invariant).

    According to mathworks.com/help/sps/ref/parktransform

    Args:
        abc (np.array): Three phase signal in natural frame
        phi (float): angle between a and d axis

    Returns:
        np.array: Signals in rotating frame
    """
    abc_2_dq0_matrix: np.array = (
        2
        / 3
        * np.array(
            [
                [np.cos(phi), np.cos(phi - 2 / 3 * np.pi), np.cos(phi + 2 / 3 * np.pi)],
                [
                    -np.sin(phi),
                    -np.sin(phi - 2 / 3 * np.pi),
                    -np.sin(phi + 2 / 3 * np.pi),
                ],
                [1 / 2, 1 / 2, 1 / 2],
            ]
        )
    )
    return np.dot(abc_2_dq0_matrix, abc)


def dq0_2_abc(dq0: np.array, phi: float) -> np.array:
    """Inverse park transform (3-phase, amplitude invariant).

    According to mathworks.com/help/sps/ref/inverseparktransform

    Args:
        dq0 (np.array): Signals in rotating frame
        phi (float): angle between a and d axis

    Returns:
        np.array: Signals in natural frame
    """
    dq0_2_abc_matrix = np.array(
        [
            [np.cos(phi), -np.sin(phi), 1],
            [
                np.cos(phi - 2 / 3 * np.pi),
                -np.sin(phi - 2 / 3 * np.pi),
                1,
            ],
            [
                np.cos(phi + 2 / 3 * np.pi),
                -np.sin(phi + 2 / 3 * np.pi),
                1,
            ],
        ]
    )
    return np.dot(dq0_2_abc_matrix, dq0)


def frobenius_matrix() -> ArrayLike:
    a = np.exp(2 / 3 * np.pi * 1j)
    return 1 / 3 * np.array([[1, a, a * a], [1, a * a, a], [1, 1, 1]])


def inverse_frobenius_matrix() -> ArrayLike:
    a = np.exp(2 / 3 * np.pi * 1j)
    return 1 / 3 * np.array([[1, 1, 1], [1, a * a, a], [1, a, a * a]])


def abc_2_symmetrical_components(abc: np.array) -> ArrayLike:
    return np.dot(frobenius_matrix(), abc)


def symmetrical_components_2_abc(sym_cmp: np.array) -> ArrayLike:
    return np.dot(inverse_frobenius_matrix(), sym_cmp)


def abc_2_ab0(abc: np.array) -> np.array:
    """Clarke transformation from abc to alpha, beta, zero.

    Args:
        abc (np.array): abc values

    Returns:
        np.array: alpha, beta, zero values
    """
    abc2ab0_matrix: np.matrix = (
        2
        / 3
        * np.array(
            [
                [1, -1 / 2, -1 / 2],
                [0, np.sqrt(3) / 2, -np.sqrt(3) / 2],
                [1 / 2, 1 / 2, 1 / 2],
            ]
        )
    )
    return np.dot(abc2ab0_matrix, abc)
