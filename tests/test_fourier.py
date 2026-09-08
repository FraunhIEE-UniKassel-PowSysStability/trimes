import numpy as np
import pandas as pd
import pytest

from trimes import fourier


class TestFourierCoefficients:
    def test_real_part_of_unit_cosine_is_half(self):
        x = np.cos(np.linspace(0.0, 2 * np.pi, 256, endpoint=False))
        assert fourier.get_fourier_coef_real(x) == pytest.approx(0.5, abs=1e-6)
        assert fourier.get_fourier_coef_imag(x) == pytest.approx(0.0, abs=1e-6)

    def test_imag_part_of_unit_sine_is_negative_half(self):
        x = np.sin(np.linspace(0.0, 2 * np.pi, 256, endpoint=False))
        assert fourier.get_fourier_coef_imag(x) == pytest.approx(-0.5, abs=1e-6)
        assert fourier.get_fourier_coef_real(x) == pytest.approx(0.0, abs=1e-6)

    def test_angle_argument_rotates_the_projection(self):
        x = np.cos(np.linspace(0.0, 2 * np.pi, 256, endpoint=False))
        # projecting onto a cosine shifted by pi/2 kills the real part
        assert fourier.get_fourier_coef_real(x, angle=np.pi / 2) == pytest.approx(
            0.0, abs=1e-6
        )

    def test_fundamental_projection_rejects_third_harmonic(self):
        base = np.linspace(0.0, 2 * np.pi, 300, endpoint=False)
        x = np.cos(base) + 0.4 * np.cos(3 * base)
        assert fourier.get_fourier_coef_real(x, k=1) == pytest.approx(0.5, abs=1e-6)

    def test_harmonic_number_selects_that_harmonic(self):
        base = np.linspace(0.0, 2 * np.pi, 300, endpoint=False)
        x = np.cos(3 * base)  # pure third harmonic, unit amplitude
        assert fourier.get_fourier_coef_real(x, k=1) == pytest.approx(0.0, abs=1e-9)
        assert fourier.get_fourier_coef_real(x, k=3) == pytest.approx(0.5, abs=1e-9)

    def test_harmonic_amplitude_and_phase_from_a_mixture(self):
        base = np.linspace(0.0, 2 * np.pi, 512, endpoint=False)
        x = 2.0 * np.cos(base) + 0.5 * np.cos(3 * base - 0.7)
        real = fourier.get_fourier_coef_real(x, k=3)
        imag = fourier.get_fourier_coef_imag(x, k=3)
        assert 2 * np.hypot(real, imag) == pytest.approx(0.5, abs=1e-6)
        assert np.arctan2(imag, real) == pytest.approx(-0.7, abs=1e-6)


class TestRollingFourier:
    @pytest.fixture
    def cosine_frame(self):
        n, per = 400, 100
        t = np.arange(float(n))
        return pd.DataFrame({"x": np.cos(2 * np.pi * t / per)}, index=t), per

    def test_rolling_magnitude_is_rms_phasor(self, cosine_frame):
        df, per = cosine_frame
        fc = fourier.get_fourier_coef_rolling(df, per)
        assert isinstance(fc, pd.DataFrame)
        assert fc.shape == df.shape
        # sqrt(2) scaling turns the 0.5 coefficient into 1/sqrt(2)
        np.testing.assert_allclose(
            np.abs(fc["x"].to_numpy()[2 * per :]), 1 / np.sqrt(2), atol=1e-6
        )

    def test_first_window_is_padded(self, cosine_frame):
        df, per = cosine_frame
        fc = fourier.get_fourier_coef_rolling(df, per)
        np.testing.assert_allclose(fc["x"].to_numpy()[:per], 0.0)

    def test_series_input_still_produces_a_result_per_sample(self, cosine_frame):
        df, per = cosine_frame
        fc = fourier.get_fourier_coef_rolling(df["x"], per)
        assert len(fc) == len(df)
        np.testing.assert_allclose(
            np.abs(np.asarray(fc)[2 * per :]).ravel(), 1 / np.sqrt(2), atol=1e-6
        )

    def test_rolling_can_track_a_harmonic(self):
        n, per = 400, 100
        t = np.arange(float(n))
        df = pd.DataFrame({"x": np.cos(2 * np.pi * 3 * t / per)}, index=t)
        fc = fourier.get_fourier_coef_rolling(df, per, k=3)
        np.testing.assert_allclose(
            np.abs(fc["x"].to_numpy()[2 * per :]), 1 / np.sqrt(2), atol=1e-6
        )
