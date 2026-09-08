import numpy as np
import pandas as pd
import pytest

from trimes import derivative, filter as tfilter, regression


class TestDiffMovingAvg:
    def test_derivative_of_ramp_is_slope(self):
        dt = 0.1
        t = np.arange(0.0, 5.0, dt)
        ramp = pd.Series(3.0 * t, index=t)
        got = derivative.diff_moving_avg(ramp, 5, dt)
        np.testing.assert_allclose(got.to_numpy()[10:-1], 3.0, atol=1e-6)

    def test_dataframe_shape(self):
        dt = 0.1
        t = np.arange(0.0, 5.0, dt)
        df = pd.DataFrame({"a": 2.0 * t, "b": -t}, index=t)
        got = derivative.diff_moving_avg(df, 5, dt)
        assert list(got.columns) == ["a", "b"]


class TestSavgolDerivative:
    def test_derivative_of_quadratic(self):
        dt = 0.01
        t = np.arange(0.0, 2.0, dt)
        df = pd.DataFrame({"x": t**2}, index=t)  # d/dt = 2 t
        got = derivative.savgol_derivative(df, 11, dt, polyorder=2)
        np.testing.assert_allclose(got["x"].to_numpy()[20:-20], (2 * t)[20:-20], atol=1e-3)

    def test_even_window_is_incremented_to_odd(self):
        dt = 0.01
        t = np.arange(0.0, 1.0, dt)
        df = pd.DataFrame({"x": t}, index=t)
        # window 10 -> 11 internally; call must not raise
        got = derivative.savgol_derivative(df, 10, dt)
        np.testing.assert_allclose(got["x"].to_numpy()[5:-5], 1.0, atol=1e-6)


class TestPt1Filter:
    def test_step_response_is_monotonic_and_approaches_one(self):
        step = pd.Series(np.ones(500), index=np.arange(500.0))
        got = tfilter.pt1_filter(step, T=10.0, dt=1.0)
        arr = got.to_numpy()
        assert np.all(np.diff(arr) >= -1e-12)
        assert arr[-1] == pytest.approx(1.0, abs=1e-3)
        assert arr[0] == pytest.approx(1.0 / 11.0, rel=1e-9)

    def test_constant_input_is_passed_through_after_settling(self):
        const = pd.Series(np.full(1000, 5.0), index=np.arange(1000.0))
        got = tfilter.pt1_filter(const, T=5.0, dt=1.0)
        assert got.to_numpy()[-1] == pytest.approx(5.0, abs=1e-6)


class TestPolyfit:
    def test_recovers_line_coefficients(self):
        t = np.arange(10.0)
        line = pd.Series(3.0 * t + 2.0, index=t)
        coeffs = regression.polyfit(line, 1)
        np.testing.assert_allclose(coeffs, [3.0, 2.0], atol=1e-9)

    def test_dataframe_returns_column_wise_coefficients(self):
        t = np.arange(10.0)
        df = pd.DataFrame({"a": 2.0 * t, "b": -t + 1.0}, index=t)
        coeffs = regression.polyfit(df, 1)
        assert coeffs.shape == (2, 2)
        np.testing.assert_allclose(coeffs[0], [2.0, -1.0], atol=1e-9)

    def test_time_offset_shifts_intercept(self):
        t = np.arange(10.0)
        line = pd.Series(2.0 * t, index=t)
        slope, intercept = regression.polyfit(line, 1, t_offset=1.0)
        assert slope == pytest.approx(2.0)
        assert intercept == pytest.approx(-2.0, abs=1e-9)
