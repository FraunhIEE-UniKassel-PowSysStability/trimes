import numpy as np
import pandas as pd
import pytest

from trimes import root_mean_square as rms


@pytest.fixture
def sine():
    n, per = 500, 100
    t = np.arange(float(n))
    return pd.Series(np.sin(2 * np.pi * t / per), index=t), per


class TestRmsRolling:
    def test_converges_to_one_over_sqrt_two(self, sine):
        s, per = sine
        got = rms.rms_rolling(s, per)
        np.testing.assert_allclose(got.to_numpy()[2 * per :], 1 / np.sqrt(2), atol=1e-6)

    def test_shape_and_type_preserved(self, sine):
        s, per = sine
        got = rms.rms_rolling(s, per)
        assert isinstance(got, pd.Series)
        assert got.shape == s.shape

    def test_normalize_magnitude_scales_by_sqrt_two(self, sine):
        s, per = sine
        plain = rms.rms_rolling(s, per).to_numpy()[2 * per :]
        norm = rms.rms_rolling(s, per, normalize_magnitude=True).to_numpy()[2 * per :]
        np.testing.assert_allclose(norm, plain * np.sqrt(2), atol=1e-6)

    def test_dataframe_columns(self):
        n, per = 400, 100
        t = np.arange(float(n))
        df = pd.DataFrame(
            {"a": np.sin(2 * np.pi * t / per), "b": 2 * np.sin(2 * np.pi * t / per)},
            index=t,
        )
        got = rms.rms_rolling(df, per)
        assert list(got.columns) == ["a", "b"]
        np.testing.assert_allclose(
            got["b"].to_numpy()[2 * per :], 2 / np.sqrt(2), atol=1e-6
        )


class TestRmsMinMax:
    def test_balanced_three_phase_settles_to_amplitude(self):
        n, per = 2000, 200
        t = np.arange(n) * (1.0 / (50 * per))
        w = 2 * np.pi * 50 * t
        df = pd.DataFrame(
            {
                "a": np.cos(w),
                "b": np.cos(w - 2 / 3 * np.pi),
                "c": np.cos(w + 2 / 3 * np.pi),
            },
            index=t,
        )
        got = rms.rms_min_max(df, filter_time_constant=0.01)
        assert isinstance(got, pd.Series)
        # peak amplitude 1 -> the min/max estimator tracks the peak value
        assert got.to_numpy()[-1] == pytest.approx(1.0, abs=0.05)
