import numpy as np
import pandas as pd
import pytest

from trimes import fft


@pytest.fixture
def fs() -> float:
    return 10_000.0


@pytest.fixture
def signal(fs) -> pd.DataFrame:
    """u_a: 100 V at 50 Hz + 10 V 5th harmonic + 5 V 7th harmonic, coherent."""
    n = 2000  # 10 fundamental cycles -> 5 Hz bins
    t = np.arange(n) / fs
    x = (
        100.0 * np.cos(2 * np.pi * 50 * t)
        + 10.0 * np.cos(2 * np.pi * 250 * t)
        + 5.0 * np.cos(2 * np.pi * 350 * t)
    )
    return pd.DataFrame({"u_a": x}, index=pd.Index(t, name="time"))


class TestAmplitudeSpectrum:
    def test_bin_centred_tone_recovers_amplitude(self, signal):
        spec = fft.amplitude_spectrum(signal)
        assert fft.at_frequency(spec, 50.0).iloc[0, 0] == pytest.approx(100.0, rel=1e-6)

    def test_rms_scaling(self, signal):
        spec = fft.amplitude_spectrum(signal, scaling="rms")
        assert fft.at_frequency(spec, 50.0).iloc[0, 0] == pytest.approx(
            100.0 / np.sqrt(2), rel=1e-6
        )

    def test_unknown_scaling_raises(self, signal):
        with pytest.raises(ValueError):
            fft.amplitude_spectrum(signal, scaling="bogus")

    def test_index_is_named_frequency_and_starts_at_dc(self, signal):
        spec = fft.amplitude_spectrum(signal)
        assert spec.index.name == "frequency"
        assert spec.index[0] == 0.0

    def test_rejects_nan(self, signal):
        signal.iloc[5, 0] = np.nan
        with pytest.raises(ValueError):
            fft.amplitude_spectrum(signal)

    def test_rejects_non_uniform_sampling(self):
        idx = np.concatenate([np.arange(0, 1, 0.001), [2.0]])
        df = pd.DataFrame({"x": np.zeros(idx.size)}, index=idx)
        with pytest.raises(ValueError):
            fft.amplitude_spectrum(df)


class TestRfftAndPhase:
    def test_rfft_length(self, signal):
        spec = fft.rfft(signal)
        assert len(spec) == len(signal) // 2 + 1

    def test_phase_of_cosine_is_zero_at_its_bin(self, fs):
        n = 2000
        t = np.arange(n) / fs
        df = pd.DataFrame({"x": np.cos(2 * np.pi * 50 * t)}, index=t)
        phase = fft.phase_spectrum(df)
        assert fft.at_frequency(phase, 50.0).iloc[0, 0] == pytest.approx(0.0, abs=1e-6)

    def test_phase_of_sine_is_minus_half_pi(self, fs):
        n = 2000
        t = np.arange(n) / fs
        df = pd.DataFrame({"x": np.sin(2 * np.pi * 50 * t)}, index=t)
        phase = fft.phase_spectrum(df)
        assert fft.at_frequency(phase, 50.0).iloc[0, 0] == pytest.approx(
            -np.pi / 2, abs=1e-4
        )


class TestPSD:
    def test_parseval(self, signal):
        psd = fft.power_spectral_density(signal, window=None, detrend="constant")
        variance = np.trapezoid(psd["u_a"].to_numpy(), psd.index.to_numpy())
        expected = signal["u_a"].to_numpy().var()
        assert variance == pytest.approx(expected, rel=1e-6)


class TestAtFrequency:
    def test_nearest_bin_within_tolerance(self, signal):
        spec = fft.amplitude_spectrum(signal)
        got = fft.at_frequency(spec, [50.0, 250.0, 350.0])
        np.testing.assert_allclose(got["u_a"].to_numpy(), [100.0, 10.0, 5.0], rtol=1e-6)

    def test_out_of_range_returns_nan(self, signal):
        spec = fft.amplitude_spectrum(signal)
        got = fft.at_frequency(spec, 1e9)
        assert np.isnan(got.iloc[0, 0])

    def test_requires_two_bins(self):
        spec = pd.DataFrame({"x": [1.0]}, index=pd.Index([0.0], name="frequency"))
        with pytest.raises(ValueError):
            fft.at_frequency(spec, 0.0)


class TestHarmonicsAndThd:
    def test_harmonic_amplitudes(self, signal):
        got = fft.harmonics(signal, fundamental=50.0, orders=[1, 5, 7])
        np.testing.assert_allclose(
            got["u_a"].to_numpy(), [100.0, 10.0, 5.0], rtol=1e-6
        )

    def test_orders_above_nyquist_are_nan(self, signal):
        got = fft.harmonics(signal, fundamental=50.0, orders=[1, 1000])
        assert np.isnan(got["u_a"].to_numpy()[1])

    def test_thd_matches_definition(self, signal):
        got = fft.thd(signal, fundamental=50.0)
        expected = np.sqrt(10.0**2 + 5.0**2) / 100.0
        assert float(got.iloc[0]) == pytest.approx(expected, rel=1e-6)
        assert got.name == "thd"
