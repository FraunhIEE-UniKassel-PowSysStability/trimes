import numpy as np
import pandas as pd
import pytest

from trimes import phasor


@pytest.fixture
def fs() -> float:
    return 10_000.0


def _signal(fs, freq, amp, phase, cycles=5.0):
    n = int(round(cycles / freq * fs))
    t = np.arange(n) / fs
    x = amp * np.cos(2 * np.pi * freq * t + phase)
    return pd.DataFrame({"u_a": x}, index=pd.Index(t, name="time"))


class TestEstimateFrequency:
    def test_off_nominal_frequency(self, fs):
        df = _signal(fs, 49.37, 100.0, 0.5)
        got = phasor.estimate_frequency(df)
        assert got.loc["u_a", "frequency"] == pytest.approx(49.37, abs=1e-2)
        assert got.loc["u_a", "amplitude"] == pytest.approx(100.0, rel=1e-3)
        assert got.loc["u_a", "phase"] == pytest.approx(0.5, abs=1e-2)

    def test_nominal_frequency_clean_signal(self, fs):
        df = _signal(fs, 50.0, 10.0, 0.0)
        got = phasor.estimate_frequency(df)
        assert got.loc["u_a", "frequency"] == pytest.approx(50.0, abs=1e-3)

    def test_series_input_supported(self, fs):
        s = _signal(fs, 50.0, 1.0, 0.0)["u_a"]
        got = phasor.estimate_frequency(s)
        assert got.index.tolist() == ["u_a"]

    def test_multichannel_independent(self, fs):
        n = 1000
        t = np.arange(n) / fs
        df = pd.DataFrame(
            {
                "a": 100.0 * np.cos(2 * np.pi * 49.5 * t),
                "b": 50.0 * np.cos(2 * np.pi * 50.5 * t),
            },
            index=pd.Index(t, name="time"),
        )
        got = phasor.estimate_frequency(df)
        assert got.loc["a", "frequency"] == pytest.approx(49.5, abs=5e-2)
        assert got.loc["b", "frequency"] == pytest.approx(50.5, abs=5e-2)
        assert got.loc["a", "amplitude"] == pytest.approx(100.0, rel=1e-2)

    def test_rejects_non_uniform_sampling(self):
        idx = np.concatenate([np.arange(0, 0.05, 1e-4), [0.1]])
        df = pd.DataFrame({"u_a": np.zeros(idx.size)}, index=idx)
        with pytest.raises(ValueError):
            phasor.estimate_frequency(df)

    def test_rejects_nan(self, fs):
        df = _signal(fs, 50.0, 1.0, 0.0)
        df.iloc[3, 0] = np.nan
        with pytest.raises(ValueError):
            phasor.estimate_frequency(df)

    def test_rejects_too_few_samples(self):
        df = pd.DataFrame({"u_a": [1.0, 2.0, 3.0]}, index=[0.0, 1e-4, 2e-4])
        with pytest.raises(ValueError):
            phasor.estimate_frequency(df)


class TestFrequencyTrace:
    def test_tracks_constant_frequency(self, fs):
        df = _signal(fs, 50.0, 100.0, 0.0, cycles=30.0)
        trace = phasor.frequency_trace(df, nominal=50.0, window_cycles=3.0)
        freq = trace.xs("frequency", axis=1, level="quantity")["u_a"]
        np.testing.assert_allclose(freq.to_numpy(), 50.0, atol=1e-2)
        assert trace.index.name == "time"

    def test_rocof_column_present_and_near_zero(self, fs):
        df = _signal(fs, 50.0, 100.0, 0.0, cycles=30.0)
        trace = phasor.frequency_trace(df, nominal=50.0, rocof=True)
        rocof = trace.xs("rocof", axis=1, level="quantity")["u_a"]
        np.testing.assert_allclose(rocof.to_numpy(), 0.0, atol=1.0)

    def test_invalid_cycle_arguments(self, fs):
        df = _signal(fs, 50.0, 1.0, 0.0, cycles=10.0)
        with pytest.raises(ValueError):
            phasor.frequency_trace(df, window_cycles=0.0)

    def test_window_longer_than_record_raises(self, fs):
        df = _signal(fs, 50.0, 1.0, 0.0, cycles=2.0)
        with pytest.raises(ValueError):
            phasor.frequency_trace(df, window_cycles=5.0)


class TestReferencedPhase:
    def test_constant_for_nominal_signal(self, fs):
        df = _signal(fs, 50.0, 100.0, 0.3, cycles=40.0)
        trace = phasor.frequency_trace(df, nominal=50.0)
        ref = phasor.referenced_phase(trace, nominal=50.0)
        # de-rotated phase is roughly constant across the trace
        spread = np.ptp(ref["u_a"].to_numpy())
        assert spread < 0.1
