import numpy as np
import pandas as pd
import pytest

from trimes import signal_processing as sp


class TestExtendNp:
    def test_wrap_at_start(self):
        got = sp.extend_np(np.array([1, 2, 3, 4]), 2, "wrap")
        np.testing.assert_array_equal(got, [1, 2, 1, 2, 3, 4])

    def test_wrap_at_end(self):
        got = sp.extend_np(np.array([1, 2, 3, 4]), -2, "wrap")
        np.testing.assert_array_equal(got, [1, 2, 3, 4, 3, 4])

    def test_reflect_at_start(self):
        got = sp.extend_np(np.array([1, 2, 3, 4]), 2, "reflect")
        np.testing.assert_array_equal(got, [2, 1, 1, 2, 3, 4])

    def test_2d_array_preserves_columns(self):
        x = np.arange(12).reshape(6, 2)
        got = sp.extend_np(x, 2, "wrap")
        assert got.shape == (8, 2)
        np.testing.assert_array_equal(got[:2], x[:2])

    def test_rejects_3d(self):
        with pytest.raises(ValueError):
            sp.extend_np(np.zeros((2, 2, 2)), 1)


class TestExtendTs:
    def test_extends_index_and_values(self):
        df = pd.DataFrame({"a": [0.0, 1.0, 2.0, 3.0]}, index=[0.0, 1.0, 2.0, 3.0])
        got = sp.extend_ts(df, 2)
        assert len(got) == 6
        assert got.index[0] < 0.0


class TestAverageRolling:
    def test_constant_signal_is_unchanged_past_the_pad(self):
        s = pd.Series(np.full(20, 5.0), index=np.arange(20.0))
        got = sp.average_rolling(s, 4)
        np.testing.assert_allclose(got.to_numpy()[4:], 5.0)

    def test_first_window_is_zero_padded_by_default(self):
        s = pd.Series(np.full(20, 5.0), index=np.arange(20.0))
        got = sp.average_rolling(s, 4)
        np.testing.assert_allclose(got.to_numpy()[:4], 0.0)

    def test_edge_pad_mode_fills_first_window(self):
        s = pd.Series(np.full(20, 5.0), index=np.arange(20.0))
        got = sp.average_rolling(s, 4, pad_mode="edge")
        np.testing.assert_allclose(got.to_numpy(), 5.0)

    def test_smooths_alternating_signal(self):
        s = pd.Series(np.tile([0.0, 2.0], 20), index=np.arange(40.0))
        got = sp.average_rolling(s, 2)
        # after the first (padded) window the running mean is exactly 1.0
        np.testing.assert_allclose(got.to_numpy()[2:], 1.0)

    def test_preserves_shape_and_type_for_dataframe(self):
        df = pd.DataFrame(
            {"a": np.arange(20.0), "b": np.arange(20.0) * 2}, index=np.arange(20.0)
        )
        got = sp.average_rolling(df, 5)
        assert isinstance(got, pd.DataFrame)
        assert got.shape == df.shape


class TestGetAngle:
    def test_recovers_phase_of_cosine(self):
        n = 400
        phi = 0.6
        x = np.cos(np.linspace(0, 2 * np.pi, n, endpoint=False) + phi)
        assert sp.get_angle(x) == pytest.approx(phi, abs=1e-2)
