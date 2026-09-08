"""The ``.trimes`` pandas accessor.

Importing this module (which :mod:`trimes` does automatically) registers a
``trimes`` accessor on :class:`pandas.Series` and :class:`pandas.DataFrame`, so
the most common :mod:`trimes.base` / rolling helpers can be called as methods::

    df.trimes.resample(1e-3)
    df.trimes.between(0.1, 0.3)
    s.trimes.rms(samples_per_window=200)

It is a thin, numpy + pandas only convenience layer -- the free functions stay
the reference API, and heavier features (FFT, phasor estimation, the
synchrophasor benchmark) keep their own entry points.
"""

from __future__ import annotations

import pandas as pd

from trimes import base
from trimes.root_mean_square import rms_rolling
from trimes.signal_processing import average_rolling

__all__ = ["TrimesAccessor"]


class TrimesAccessor:
    """Backs ``Series.trimes`` and ``DataFrame.trimes`` (see :mod:`trimes.accessor`)."""

    def __init__(self, pandas_obj: pd.Series | pd.DataFrame) -> None:
        self._obj = pandas_obj

    # -- sample access ---------------------------------------------------- #
    def sample_at(self, time):
        """First sample at or after ``time`` -- :func:`trimes.base.get_sample`."""
        return base.get_sample(self._obj, time)

    def index_at(self, time):
        """Integer position at or after ``time`` -- :func:`trimes.base.get_index`."""
        return base.get_index(self._obj, time)

    def between(self, tstart, tend, *, interp: bool = False):
        """Samples in ``[tstart, tend)``.

        With ``interp=True`` the endpoints are interpolated exactly
        (:func:`trimes.base.get_between_interp`) instead of snapped to the grid
        (:func:`trimes.base.get_between`).
        """
        if interp:
            return base.get_between_interp(self._obj, tstart, tend)
        return base.get_between(self._obj, tstart, tend)

    def delta(self, tstart, tend, *, interp: bool = False):
        """Change in value between ``tstart`` and ``tend``.

        ``interp=False`` (default) uses the nearest samples
        (:func:`trimes.base.get_delta`); ``interp=True`` interpolates the
        endpoints.
        """
        if not interp:
            return base.get_delta(self._obj, tstart, tend)
        if isinstance(self._obj, pd.DataFrame):
            return base.get_delta_interp_df(self._obj, tstart, tend)
        return base.get_delta_interp_series(self._obj, tstart, tend)

    # -- resampling ----------------------------------------------------- #
    def resample(self, new_time):
        """Resample onto ``new_time`` by linear interpolation -- :func:`trimes.base.resample`."""
        return base.resample(self._obj, new_time)

    def resample_uniform(self, sampling_rate: float | None = None, method: str = "interp"):
        """Resample onto an equidistant grid -- :func:`trimes.base.resample_uniform`."""
        return base.resample_uniform(self._obj, sampling_rate, method)

    # -- rolling windows ---------------------------------------------- #
    def rms(self, samples_per_window: int, **kwargs):
        """Rolling RMS -- :func:`trimes.root_mean_square.rms_rolling`."""
        return rms_rolling(self._obj, samples_per_window, **kwargs)

    def rolling_mean(self, samples_per_window: int, **kwargs):
        """Rolling mean -- :func:`trimes.signal_processing.average_rolling`."""
        return average_rolling(self._obj, samples_per_window, **kwargs)

    # -- properties --------------------------------------------------- #
    @property
    def duration(self) -> float:
        """Time span in seconds -- :func:`trimes.base.get_duration`."""
        return base.get_duration(self._obj)

    @property
    def sampling_rate(self) -> float:
        """Sampling rate in Hz -- :func:`trimes.base.get_sampling_rate`."""
        return base.get_sampling_rate(self._obj)

    @property
    def is_uniform(self) -> bool:
        """Whether sampling is equidistant -- :func:`trimes.base.is_uniformly_sampled`."""
        return base.is_uniformly_sampled(self._obj)


pd.api.extensions.register_series_accessor("trimes")(TrimesAccessor)
pd.api.extensions.register_dataframe_accessor("trimes")(TrimesAccessor)
