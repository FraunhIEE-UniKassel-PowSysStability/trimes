"""Step-response and recovery characteristics (numpy + pandas only)."""

import numpy as np
import pandas as pd
import pytest

from trimes import step_response as sr


def _series(t, y) -> pd.Series:
    return pd.Series(np.asarray(y, dtype=float), index=pd.Index(np.asarray(t, dtype=float), name="time"))


@pytest.fixture
def first_order() -> pd.Series:
    """0 -> 1 at t = 0 with a 100 ms time constant, 1 ms steps, one sample before."""
    t = np.round(np.arange(-0.001, 1.0, 0.001), 6)
    return _series(t, np.where(t < 0, 0.0, 1.0 - np.exp(-np.clip(t, 0, None) / 0.1)))


class TestCrossingTime:
    def test_first_sample_reaching_level(self, first_order):
        t = sr.crossing_time(first_order, 0.5)
        assert t == pytest.approx(0.07, abs=0.001)  # 0.1 * ln 2 = 69.3 ms

    def test_down_and_start(self):
        s = _series([0, 1, 2, 3, 4], [1.0, 0.2, 1.0, 0.1, 1.0])
        assert sr.crossing_time(s, 0.5, direction="down") == 1.0
        assert sr.crossing_time(s, 0.5, direction="down", start=1.5) == 3.0

    def test_never_reached_is_inf(self, first_order):
        assert sr.crossing_time(first_order, 2.0) == np.inf


class TestRiseTime:
    def test_10_to_90_percent(self, first_order):
        # analytic: 0.1 * ln 9 = 219.7 ms
        assert sr.rise_time(first_order, initial=0.0, final=1.0) == pytest.approx(0.2197, abs=0.002)

    def test_from_step_instant(self, first_order):
        # limits (0, 0.9): from the step instant to 90 %, 0.1 * ln 10 = 230.3 ms
        rt = sr.rise_time(first_order, initial=0.0, final=1.0, limits=(0.0, 0.9), start=0.0)
        assert rt == pytest.approx(0.2303, abs=0.002)

    def test_falling_step(self, first_order):
        falling = 1.0 - first_order
        assert sr.rise_time(falling, initial=1.0, final=0.0) == pytest.approx(0.2197, abs=0.002)

    def test_not_reached_and_zero_step(self, first_order):
        assert sr.rise_time(first_order, initial=0.0, final=2.0) == np.inf
        assert np.isnan(sr.rise_time(first_order, initial=1.0, final=1.0))

    def test_rejects_bad_limits(self, first_order):
        with pytest.raises(ValueError):
            sr.rise_time(first_order, initial=0.0, final=1.0, limits=(0.9, 0.1))


class TestSettlingTime:
    def test_absolute_band(self, first_order):
        # |1 - y| <= 0.05 from t = 0.1 * ln 20 = 299.6 ms
        assert sr.settling_time(first_order, final=1.0, band=0.05, start=0.0) == pytest.approx(0.2996, abs=0.002)

    def test_inside_throughout_and_never_settled(self, first_order):
        assert sr.settling_time(first_order, final=0.5, band=1.0) == 0.0
        assert sr.settling_time(first_order, final=2.0, band=0.1) == np.inf

    def test_non_uniform_sampling(self):
        # extra samples at the event, as simulators write them
        s = _series([0.0, 0.0005, 0.001, 0.01, 0.1, 1.0], [0.0, 0.5, 0.8, 0.95, 1.0, 1.0])
        assert sr.settling_time(s, final=1.0, band=0.1, start=0.0) == pytest.approx(0.01)


class TestRecoveryTime:
    def test_last_crossing_counts(self):
        # rises above 0.9 at t = 2, dips again at t = 3, stays above from t = 5
        s = _series([0, 1, 2, 3, 4, 5, 6], [0.2, 0.5, 0.95, 0.85, 0.89, 0.92, 0.95])
        assert sr.recovery_time(s, 0.9, start=0.0) == 5.0

    def test_never_below_and_not_recovered(self):
        s = _series([0, 1, 2], [0.95, 0.96, 0.97])
        assert sr.recovery_time(s, 0.9) == 0.0
        assert sr.recovery_time(_series([0, 1, 2], [0.5, 0.95, 0.5]), 0.9) == np.inf

    def test_down(self):
        s = _series([0, 1, 2, 3], [1.2, 1.15, 1.05, 1.0])
        assert sr.recovery_time(s, 1.1, direction="down") == 2.0


class TestOvershoot:
    def test_fraction_of_step(self):
        s = _series([0, 1, 2, 3], [0.0, 1.2, 0.95, 1.0])
        assert sr.overshoot(s, initial=0.0, final=1.0) == pytest.approx(0.2)
        assert sr.overshoot(-s, initial=0.0, final=-1.0) == pytest.approx(0.2)

    def test_none_and_zero_step(self, first_order):
        assert sr.overshoot(first_order, initial=0.0, final=1.0) == 0.0
        assert np.isnan(sr.overshoot(first_order, initial=1.0, final=1.0))
