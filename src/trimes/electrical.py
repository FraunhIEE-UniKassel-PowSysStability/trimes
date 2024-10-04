import numpy as np
from numpy.typing import ArrayLike


def get_power_power_symmetrical_components(
    u: ArrayLike, i: ArrayLike, factor: float = 3
):
    return get_active_power_symmetrical_components(
        u, i, factor=factor
    ) + 1j * get_reactive_power_symmetrical_components(u, i, factor=factor)


def get_active_power_symmetrical_components(
    u: ArrayLike, i: ArrayLike, factor: float = 3
):
    return factor * (np.dot(np.real(u), np.real(i)) + np.dot(np.imag(u), np.imag(i)))


def get_reactive_power_symmetrical_components(
    u: ArrayLike, i: ArrayLike, factor: float = 3
):
    return factor * (-np.dot(np.real(u), np.imag(i)) + np.dot(np.imag(u), np.real(i)))
