"""Tests for the ``.trimes`` pandas accessor."""

import numpy as np
import pandas as pd
import pytest

import trimes  # noqa: F401  (registers the accessor)
from trimes import base
from trimes.root_mean_square import rms_rolling
from trimes.signal_processing import average_rolling


@pytest.fixture
def series():
    t = np.arange(0.0, 1.0, 1e-3)
    return pd.Series(np.sin(2 * np.pi * 5 * t), index=pd.Index(t, name="time"))


@pytest.fixture
def frame(series):
    return series.to_frame("a").assign(b=lambda d: 2 * d["a"])


def test_accessor_registered(series, frame):
    assert hasattr(series, "trimes")
    assert hasattr(frame, "trimes")


def test_properties(series):
    assert series.trimes.duration == pytest.approx(series.index[-1] - series.index[0])
    assert series.trimes.sampling_rate == pytest.approx(1000.0)
    assert series.trimes.is_uniform is True


def test_access_delegates(series):
    pd.testing.assert_series_equal(
        series.trimes.sample_at([0.1, 0.2]), base.get_sample(series, [0.1, 0.2])
    )
    pd.testing.assert_series_equal(
        series.trimes.between(0.1, 0.3), base.get_between(series, 0.1, 0.3)
    )
    pd.testing.assert_series_equal(
        series.trimes.between(0.1, 0.3, interp=True),
        base.get_between_interp(series, 0.1, 0.3),
    )


def test_delta_interp_dispatch(series, frame):
    assert series.trimes.delta(0.1, 0.3) == base.get_delta(series, 0.1, 0.3)
    assert series.trimes.delta(0.1, 0.3, interp=True) == pytest.approx(
        base.get_delta_interp_series(series, 0.1, 0.3)
    )
    pd.testing.assert_series_equal(
        frame.trimes.delta(0.1, 0.3, interp=True),
        base.get_delta_interp_df(frame, 0.1, 0.3),
    )


def test_resample(series):
    pd.testing.assert_series_equal(
        series.trimes.resample(2e-3), base.resample(series, 2e-3)
    )
    pd.testing.assert_series_equal(
        series.trimes.resample_uniform(500.0),
        base.resample_uniform(series, 500.0),
    )


def test_rolling(series, frame):
    pd.testing.assert_series_equal(series.trimes.rms(100), rms_rolling(series, 100))
    pd.testing.assert_frame_equal(
        frame.trimes.rolling_mean(100), average_rolling(frame, 100)
    )
