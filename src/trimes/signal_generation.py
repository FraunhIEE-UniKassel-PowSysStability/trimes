from collections.abc import Callable
from numbers import Number

import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from icecream import ic

from trimes.base import superpose_series, get_index, to_numpy_array


class PeriodicSignal:
    """Create periodic time series signals."""

    def __init__(
        self,
        t: ArrayLike,
        func: Callable = np.cos,
        f: ArrayLike | tuple = 1.0,
        mag: ArrayLike | tuple = 1.0,
        offset: float = 0.0,
        phi: float = 0.0,
    ) -> None:
        self.t: ArrayLike = t
        "time"
        self.func: Callable = func
        "function to create periodic signals (e.g. np.sin)"
        if isinstance(f, tuple):
            f = np.linspace(f[0], f[1], len(self.t))
        self.f: ArrayLike = f
        "frequency"
        if isinstance(mag, tuple):
            mag = np.linspace(mag[0], mag[1], len(self.t))
        self.mag: ArrayLike = mag
        "magnitude"
        if isinstance(offset, tuple):
            offset = np.linspace(offset[0], offset[1], len(self.t))
        self.offset: ArrayLike = offset
        "y offset"
        self.phi: float = phi
        "initial angle"

    def get_signal(self) -> ArrayLike:
        return self.mag * self.func(self.get_angle_over_time()) + self.offset

    def get_signal_from_angle(self, angle: ArrayLike) -> ArrayLike:
        return self.mag * self.func(angle) + self.offset

    def get_signal_series(self) -> pd.Series:
        series = pd.Series(self.get_signal(), index=self.t)
        series.index.name = "time"
        return series

    def get_signal_series_from_angle(self, angle: ArrayLike) -> pd.Series:
        series = pd.Series(self.get_signal_from_angle(angle), index=self.t)
        series.index.name = "time"
        return series

    def get_signal_n_phases(self, n: int) -> pd.DataFrame:
        return superimpose_and_concat_periodic_signals([[self]], num_phases=n)

    def get_angle_over_time(self) -> ArrayLike:
        """Get angle over time.

        Calculation is dependent on the frequency format (constant scalar number or gradient).
        """
        if isinstance(self.f, Number):
            return 2 * np.pi * self.f * (self.t - self.t[0]) + self.phi
        else:
            frequency_integrated = (
                np.cumsum(2 * np.pi * self.f[1:] * np.diff(self.t)) + self.phi
            )
            return np.array([self.phi, *frequency_integrated])

    def get_signal_attributes_over_time(
        self, attributes: list[str] | None = None
    ) -> pd.DataFrame:
        if attributes is None:
            attributes = ["f", "mag", "offset"]
        attrs_over_time = pd.DataFrame(columns=attributes, index=self.t, dtype=float)
        for attr in attributes:
            attrs_over_time[attr] = self.__getattribute__(attr)
        return attrs_over_time

    def plot(self, **kwargs):
        series = self.get_signal_series()
        series.plot(**kwargs)

    def plot_with_attributes(self, attributes: list[str] | None = None) -> list[Axes]:
        """Plot signal with attributes

        Args:
            attributes (list[str] | None, optional): Atrributes ('f', 'mag', 'offset'). Defaults to None (all).

        Returns:
            list[Axes]: Axes of plot.
        """
        attr = self.get_signal_attributes_over_time(attributes)
        fig, axes = plt.subplots(nrows=attr.shape[1] + 1, ncols=1)
        fig.tight_layout()
        plt.sca(axes[0])
        self.plot()
        plt.title("signal")
        plt.xlabel("")
        for col, ax in enumerate(axes[1:]):
            ax.plot(attr.index, attr.iloc[:, col])
            ax.grid()
            ax.set_title(attr.columns[col])
        return axes


def superimpose_and_concat_periodic_signals(
    signals: list[list[PeriodicSignal]],
    num_phases: int = 1,
    angle_continuation: bool = True,
) -> pd.Series | pd.DataFrame:
    """Concatenate array of periodic signals.

    Args:
        signals (list[list[PeriodicSignal]]): List of lists of PeriodicSignal objects. The PeriodicSignals in each row are superimposed and the resulting column is concatenated.

        num_phases (int, optional): To create multiples phases with angle separation of pi/num_phases. Defaults to 1.

        angle_continuation (bool, optional): If true, jumps in phase angle between signals that are concatenated are avoided (the phase angle PeriodicSignal.phi is adapted).  Defaults to True.

    Returns:
         pd.Series | pd.DataFrame: Series or DataFrame depending on the number of phases.

    Examples:
        Define array of PeriodicSignals. Rows will be superimposed and the resulting column will be concatenated:

        ```
        step_size = 1e-3
        signals = [
            [
                PeriodicSignal(np.arange(0, 0.1, step_size), f=(50, 52)),
                PeriodicSignal(np.arange(0, 0.1, step_size), f=5*50, mag=(0.03, 0.06)),
            ],
            [
                PeriodicSignal(np.arange(0.1, 0.3, step_size), f=50),
                PeriodicSignal(np.arange(0.1, 0.3, step_size), f=7*50, mag=0.06),
            ],
            [
                PeriodicSignal(np.arange(0.3, 0.5, step_size), f=49),
                PeriodicSignal(np.arange(0.3, 0.5, step_size), f=9*50, mag=0.06),
            ],
        ]
        res = concat_periodic_signals(signals, num_phases = 3)
        ```
    """
    phase_signals = [None] * num_phases
    for phase in range(num_phases):
        phase_angle = phase * (-2 * np.pi / num_phases)
        series_to_concat = [None] * np.shape(signals)[0]
        if angle_continuation:
            phi = [signal.phi for signal in signals[0]]
        for n, signals_to_superpose in enumerate(signals):
            signals_superposed = [None] * np.shape(signals)[1]
            for m, signal in enumerate(signals_to_superpose):
                phi_initial = signal.phi
                if angle_continuation:
                    signal.phi = phi[m]
                signal.phi += phase_angle
                angle = signal.get_angle_over_time()
                signals_superposed[m] = signal.get_signal_series_from_angle(angle)
                if angle_continuation:
                    phi[m] = angle[-1] + (angle[-1] - angle[-2]) - phase_angle
                signal.phi = phi_initial
            series_to_concat[n] = superpose_series(signals_superposed)

        phase_signals[phase] = pd.concat(series_to_concat, axis=0)
    if num_phases == 1:
        return phase_signals[0]
    else:
        return pd.concat(phase_signals, axis=1)


def get_attributes_of_superimposed_and_concatenated_signals_over_time(
    signals: list[list[PeriodicSignal]],
    columns: list[int] | None = None,
    attributes: list[str] | None = None,
) -> list[pd.DataFrame]:
    """_summary_

    Args:
        signals (list[list[PeriodicSignal]]): List of lists of PeriodicSignal objects. The PeriodicSignals in each row are superimposed and the resulting column is concatenated.

        columns (list[int] | None, optional): Columns for which attribute are returned. Defaults to None (all columns).

        attributes (list[str] | None, optional): Attributes ('f', 'mag', 'offset'). Defaults to None (all).

    Returns:
        list[pd.DataFrame]: Attributes for columns.
    """
    if columns is None:
        columns = range(0, len(signals[0]))
    attrs_of_columns = [None] * len(columns)
    for n, col in enumerate(columns):
        frames = [pd.DataFrame()] * len(signals)
        for row in range(len(signals)):
            frames[row] = signals[row][col].get_signal_attributes_over_time(attributes)
        attrs_of_columns[n] = pd.concat(frames)
    return attrs_of_columns


def linear_time_series(
    t: ArrayLike, y: ArrayLike, sample_time: float
) -> pd.DataFrame | pd.Series:
    """Get linear time series.

    Args:
        t (ArrayLike): points in time
        y (ArrayLike): y values corresponding to 't'
        sample_time (float): sample time

    Returns:
        pd.DataFrame | pd.Series: time series
    """
    t = make_monotonously_increasing(t, sample_time)
    ts = get_interpolated(t, y, sample_time)
    return ts


def make_monotonously_increasing(t: ArrayLike, sample_time: float) -> ArrayLike:
    """Make time 't' monotonously increasing.

    Two consecutive values n, k in 't', where k <= n, are replaced with n, k+sample_time.

    Args:
        t (ArrayLike): time
        sample_time (float): sample time

    Returns:
        ArrayLike: monotonously increasing time
    """
    t = np.array(t, dtype=float)
    diff_t_is_zero = np.concat([[False], np.diff(t) <= 0])
    t[diff_t_is_zero] = t[diff_t_is_zero] + sample_time
    return t


def get_interpolated(
    t: ArrayLike, y: ArrayLike | list[ArrayLike], sample_time: float
) -> pd.DataFrame | pd.Series:
    """Linear interpolated y (specified at 'sample_time') over 't'.

    Args:
        t (ArrayLike): time

        y (ArrayLike | list[ArrayLike]): y values (length of first dimension matches 'sample_time') or list of y values to get several results.

        sample_time (float): sample time (length of first dimension matches 'y')

    Returns:
        pd.DataFrame | pd.Series: interpolated values
    """
    index = np.arange(t[0], t[-1] + sample_time, sample_time)
    if not isinstance(y, list):
        return_series = True
        y = [y]
    else:
        return_series = False
    ts = pd.DataFrame(np.nan, index=index, columns=range(len(y)))
    ts.iloc[get_index(ts, t), :] = np.transpose(y)
    ts.interpolate(inplace=True)
    ts.index.name = "time"
    if return_series:
        ts = ts.squeeze()
    return ts


def mirror_y(ts: pd.Series, y: float, inplace=False):
    """Mirror 'ts' at 'y'.

    Args:
        ts (pd.Series): time series

        y (float): y (horizontal line for mirroring)

        inplace (bool, optional): If True the mirored signal is added to 'ts' and a DataFrame is returned. Defaults to False (only mirrored signal series is returned).

    Returns:
        _type_: _description_
    """
    mirrored_y = y - (ts - y)
    if inplace:
        ts = ts.to_frame()
        ts[1] = mirrored_y
        return ts
    else:
        return mirrored_y
