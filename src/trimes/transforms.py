"""Reference-frame transforms for three-phase signals.

Park (``dq0``), Clarke (``alpha-beta-0``) and Fortescue (symmetrical
components), all in the amplitude-invariant convention, plus a generic
row-wise transform helper.
"""

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike
import pandas as pd

__all__ = [
    "transform_time_series",
    "abc_2_dq0",
    "dq0_2_abc",
    "frobenius_matrix_symcomp",
    "inverse_frobenius_matrix_symcomp",
    "abc_2_symmetrical_components",
    "symmetrical_components_2_abc",
    "abc_2_ab0",
]


def transform_time_series(ts: pd.DataFrame, transform: Callable) -> pd.DataFrame:
    """Apply a per-row transform to a time series.

    Parameters
    ----------
    ts : pandas.DataFrame
        Time series; each row is passed to ``transform``.
    transform : callable
        Maps one row (a length-``n_columns`` array) to a new length-``n_columns``
        array.

    Returns
    -------
    pandas.DataFrame
        The transformed series, same shape and index as ``ts``.
    """
    ts_transformed = np.reshape(
        np.concatenate(ts.apply(transform, axis=1).to_numpy()), ts.shape
    )
    return pd.DataFrame(ts_transformed, index=ts.index)


def abc_2_dq0(abc: ArrayLike, phi: float) -> np.ndarray:
    """Park transform, three-phase, amplitude invariant.

    Follows mathworks.com/help/sps/ref/parktransform.

    Parameters
    ----------
    abc : array_like
        Three-phase signal in the natural frame (3 rows).
    phi : float
        Angle between the ``a`` and ``d`` axes, in radians.

    Returns
    -------
    numpy.ndarray
        The ``d``, ``q``, ``0`` components.
    """
    abc_2_dq0_matrix = (
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


def dq0_2_abc(dq0: ArrayLike, phi: float) -> np.ndarray:
    """Inverse Park transform, three-phase, amplitude invariant.

    Follows mathworks.com/help/sps/ref/inverseparktransform.

    Parameters
    ----------
    dq0 : array_like
        Signal in the rotating frame (3 rows: ``d``, ``q``, ``0``).
    phi : float
        Angle between the ``a`` and ``d`` axes, in radians.

    Returns
    -------
    numpy.ndarray
        The ``a``, ``b``, ``c`` components.
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


def frobenius_matrix_symcomp() -> np.ndarray:
    """Fortescue matrix transforming ``abc`` to symmetrical components ``012``.

    Returns
    -------
    numpy.ndarray
        The 3x3 complex matrix, including the ``1/3`` factor. Rows are the
        zero-, positive- and negative-sequence projections.
    """
    a = np.exp(2 / 3 * np.pi * 1j)
    return 1 / 3 * np.array([[1, 1, 1], [1, a, a * a], [1, a * a, a]])


def inverse_frobenius_matrix_symcomp() -> np.ndarray:
    """Inverse Fortescue matrix transforming ``012`` back to ``abc``.

    The exact inverse of :func:`frobenius_matrix_symcomp` (no ``1/3`` factor --
    that sits on the forward transform only), so
    ``symmetrical_components_2_abc(abc_2_symmetrical_components(x)) == x``.

    Returns
    -------
    numpy.ndarray
        The 3x3 complex matrix.
    """
    a = np.exp(2 / 3 * np.pi * 1j)
    return np.array([[1, 1, 1], [1, a * a, a], [1, a, a * a]])


def abc_2_symmetrical_components(abc: ArrayLike) -> np.ndarray:
    """Transform three-phase values to symmetrical components.

    Parameters
    ----------
    abc : array_like
        Columns are the phase values over time (3 columns).

    Returns
    -------
    numpy.ndarray
        Columns are the zero-, positive- and negative-sequence values
        (order ``012``).
    """
    return np.dot(frobenius_matrix_symcomp(), np.asarray(abc).T).T


def symmetrical_components_2_abc(sym_cmp: ArrayLike) -> np.ndarray:
    """Transform symmetrical components (``012``) back to three-phase ``abc``.

    Inverse of :func:`abc_2_symmetrical_components`.
    """
    return np.dot(inverse_frobenius_matrix_symcomp(), np.asarray(sym_cmp).T).T


def abc_2_ab0(abc: ArrayLike) -> np.ndarray:
    """Clarke transform from ``abc`` to ``alpha``, ``beta``, ``0``.

    Amplitude-invariant convention.

    Parameters
    ----------
    abc : array_like
        Three-phase signal (3 rows).

    Returns
    -------
    numpy.ndarray
        The ``alpha``, ``beta``, ``0`` components.
    """
    abc2ab0_matrix = (
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
