import numpy as np
import pandas as pd
import pytest

from trimes import control as tc


@pytest.fixture
def first_order_step() -> pd.Series:
    """Unit step response of a 1st-order system with tau = 0.1 s."""
    t = np.arange(0.0, 2.0, 1e-3)
    return pd.Series(1.0 - np.exp(-t / 0.1), index=t)


class TestStepInfoSeries:
    def test_returns_control_step_info_dict(self, first_order_step):
        info = tc.step_info_series(first_order_step)
        assert {"RiseTime", "SettlingTime", "SteadyStateValue"} <= set(info)

    def test_steady_state_value_recovered(self, first_order_step):
        info = tc.step_info_series(first_order_step)
        assert info["SteadyStateValue"] == pytest.approx(1.0, abs=1e-3)

    def test_first_order_has_no_overshoot(self, first_order_step):
        info = tc.step_info_series(first_order_step)
        assert info["Overshoot"] == pytest.approx(0.0, abs=1e-6)

    def test_settling_time_near_four_tau(self, first_order_step):
        info = tc.step_info_series(first_order_step)
        # 2 % settling time of a 1st-order system is ~3.9 tau
        assert info["SettlingTime"] == pytest.approx(0.39, abs=0.05)

    def test_explicit_initial_value_offsets_response(self):
        t = np.arange(0.0, 1.0, 1e-3)
        ts = 10.0 + (1.0 - np.exp(-t / 0.05))
        info = tc.step_info_series(pd.Series(ts, index=t), yinitial=10.0)
        assert info["SteadyStateValue"] == pytest.approx(1.0, abs=1e-3)
