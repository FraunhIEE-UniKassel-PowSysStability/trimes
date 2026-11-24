"""Phase 1: the analytic-truth signal object.

The closed-form truth callables must reproduce the pure-tone, linear-ramp and
sinusoidal-modulation cases to 1e-9, at arbitrary query instants.
"""

import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest

from trimes.synchrophasor import (
    AnalyticSignal,
    ROCOF_STEP,
    TWO_PI,
    build_analytic_signal,
    rms_phasor,
    wrap_pi,
)
from trimes.synchrophasor import generators as gen
from trimes.signal_generation import (
    PeriodicSignal,
    superimpose_and_concat_analytic_signal,
)

FS = 5000.0
ATOL = 1e-9


@pytest.fixture
def grid() -> np.ndarray:
    return np.arange(0.0, 1.0, 1.0 / FS)


@pytest.fixture
def query() -> np.ndarray:
    # deliberately off the sample grid
    return np.array([0.0, 3.7e-4, 0.12345, 0.5, 0.87321, 0.999])


class TestPureTone:
    def test_frequency_amplitude_phase_are_exact(self, grid, query):
        f0, amp, phi = 50.0, 2.0, 0.3
        asig = PeriodicSignal(grid, f=f0, mag=amp, phi=phi).analytic_signal(nominal=50.0)

        np.testing.assert_allclose(asig.frequency(query), f0, atol=ATOL)
        np.testing.assert_allclose(asig.amplitude(query), amp, atol=ATOL)
        np.testing.assert_allclose(
            asig.total_phase(query), TWO_PI * f0 * query + phi, atol=ATOL
        )
        np.testing.assert_allclose(asig.rocof(query), 0.0, atol=ATOL)

    def test_phase_and_phasor_against_closed_form(self, grid, query):
        f0, amp, phi = 50.0, 2.0, 0.3
        asig = PeriodicSignal(grid, f=f0, mag=amp, phi=phi).analytic_signal(nominal=50.0)

        # f0 == nominal -> nominal-referenced phase is the constant phi
        np.testing.assert_allclose(asig.phase(query, 50.0), wrap_pi(phi), atol=ATOL)
        np.testing.assert_allclose(
            asig.phasor(query, 50.0),
            (amp / np.sqrt(2)) * np.exp(1j * phi),
            atol=ATOL,
        )

    def test_off_nominal_phase_ramps_linearly(self, grid, query):
        f0, phi = 50.5, -0.2
        asig = PeriodicSignal(grid, f=f0, phi=phi).analytic_signal(nominal=50.0)
        np.testing.assert_allclose(
            asig.phase(query, 50.0),
            wrap_pi(TWO_PI * (f0 - 50.0) * query + phi),
            atol=ATOL,
        )

    def test_wrapped_samples_equal_the_generator_output(self, grid):
        ps = PeriodicSignal(grid, f=50.0, mag=2.0, phi=0.3)
        pd.testing.assert_series_equal(ps.analytic_signal().signal, ps.get_signal_series())

    def test_no_transition_times(self, grid):
        asig = PeriodicSignal(grid, f=50.0).analytic_signal()
        assert asig.metadata["transition_times"] == []
        assert asig.transition_times == []


class TestLinearRamp:
    def test_frequency_phase_rocof_against_closed_form(self):
        f_start, f_end = 49.0, 51.0
        t = np.arange(0.0, 2.0, 1.0 / FS)
        asig = PeriodicSignal(t, f=(f_start, f_end)).analytic_signal(nominal=50.0)

        span = t[-1] - t[0]
        slope = (f_end - f_start) / span
        q = np.linspace(0.0, span, 41)

        np.testing.assert_allclose(asig.frequency(q), f_start + slope * q, atol=ATOL)
        np.testing.assert_allclose(
            asig.total_phase(q),
            TWO_PI * (f_start * q + 0.5 * slope * q**2),
            atol=ATOL,
        )
        # central difference of an exactly linear frequency is exactly the slope
        np.testing.assert_allclose(asig.rocof(q), slope, atol=ATOL)

    def test_amplitude_ramp(self):
        t = np.arange(0.0, 1.0, 1.0 / FS)
        asig = PeriodicSignal(t, f=50.0, mag=(1.0, 1.5)).analytic_signal()
        span = t[-1] - t[0]
        q = np.linspace(0.0, span, 25)
        np.testing.assert_allclose(
            asig.amplitude(q), 1.0 + (0.5 / span) * q, atol=ATOL
        )

    def test_explicit_frequency_array_has_no_closed_form(self):
        t = np.arange(0.0, 0.2, 1.0 / FS)
        ps = PeriodicSignal(t, f=50.0 + np.sin(t))
        with pytest.raises(ValueError, match="closed form"):
            ps.analytic_signal()

    def test_nonzero_offset_is_rejected(self):
        t = np.arange(0.0, 0.2, 1.0 / FS)
        with pytest.raises(ValueError, match="offset"):
            PeriodicSignal(t, f=50.0, offset=1.0).analytic_signal()

    def test_non_cosine_carrier_is_rejected(self):
        t = np.arange(0.0, 0.2, 1.0 / FS)
        with pytest.raises(ValueError, match="np.cos"):
            PeriodicSignal(t, func=np.sin, f=50.0).analytic_signal()


class TestModulation:
    """AM+PM modulation: constructed directly, checked against closed forms."""

    @staticmethod
    def _build(am_depth=0.1, pm_depth=0.1, fm=2.0, f0=50.0, amp=1.0):
        # 60255-118-1 Eq 25 sign convention: phase term is -ka cos(wm t)
        wm = TWO_PI * fm
        amplitude_fn = lambda t: amp * (1.0 + am_depth * np.cos(wm * np.asarray(t, float)))
        frequency_fn = lambda t: f0 + pm_depth * fm * np.sin(wm * np.asarray(t, float))
        total_phase_fn = lambda t: (
            TWO_PI * f0 * np.asarray(t, float)
            - pm_depth * np.cos(wm * np.asarray(t, float))
        )
        ts = np.arange(0.0, 2.0, 1.0 / FS)
        wave = pd.Series(
            amplitude_fn(ts) * np.cos(total_phase_fn(ts)),
            index=pd.Index(ts, name="time"),
        )
        return (
            AnalyticSignal(wave, amplitude_fn, frequency_fn, total_phase_fn, nominal=f0),
            dict(wm=wm, am_depth=am_depth, pm_depth=pm_depth, fm=fm, f0=f0, amp=amp),
        )

    def test_phase_and_phasor_against_closed_form(self):
        asig, p = self._build()
        q = np.linspace(0.0, 2.0, 101)
        expected_phase = wrap_pi(-p["pm_depth"] * np.cos(p["wm"] * q))
        np.testing.assert_allclose(asig.phase(q, p["f0"]), expected_phase, atol=ATOL)

        expected_amp = p["amp"] * (1.0 + p["am_depth"] * np.cos(p["wm"] * q))
        np.testing.assert_allclose(
            asig.phasor(q, p["f0"]),
            (expected_amp / np.sqrt(2)) * np.exp(1j * expected_phase),
            atol=ATOL,
        )

    def test_frequency_against_closed_form(self):
        asig, p = self._build()
        q = np.linspace(0.0, 2.0, 101)
        expected = p["f0"] + p["pm_depth"] * p["fm"] * np.sin(p["wm"] * q)
        np.testing.assert_allclose(asig.frequency(q), expected, atol=ATOL)

    def test_rocof_is_the_central_difference_of_frequency(self):
        asig, p = self._build()
        q = np.linspace(0.0, 2.0, 101)
        h = ROCOF_STEP
        definition = (asig.frequency_fn(q + h) - asig.frequency_fn(q - h)) / (2.0 * h)
        np.testing.assert_allclose(asig.rocof(q), definition, atol=ATOL)

    def test_rocof_tracks_the_analytic_derivative_closely(self):
        asig, p = self._build()
        q = np.linspace(0.0, 2.0, 101)
        analytic = p["pm_depth"] * p["fm"] * p["wm"] * np.cos(p["wm"] * q)
        # central difference, O(h^2) truncation error
        np.testing.assert_allclose(asig.rocof(q), analytic, atol=1e-5)


class TestModulationGeneratorMetadata:
    """The generator must record each depth under its own name."""

    def test_am_and_pm_depth_metadata_are_not_swapped(self):
        sig = gen.modulation(
            FS, 50.0, 1.0, am_depth=0.2, pm_depth=0.05, modulation_frequency=2.0
        )
        assert sig.metadata["am_depth"] == pytest.approx(0.2)
        assert sig.metadata["pm_depth"] == pytest.approx(0.05)

    def test_metadata_depths_reproduce_the_truth(self):
        sig = gen.modulation(
            FS, 50.0, 1.0, am_depth=0.2, pm_depth=0.05, modulation_frequency=2.0
        )
        t = np.linspace(0.0, 2.0, 41)
        wm = TWO_PI * sig.metadata["modulation_frequency"]
        np.testing.assert_allclose(
            sig.amplitude(t), 1.0 + sig.metadata["am_depth"] * np.cos(wm * t), atol=ATOL
        )
        np.testing.assert_allclose(
            sig.frequency(t),
            50.0
            + sig.metadata["pm_depth"] * sig.metadata["modulation_frequency"] * np.sin(wm * t),
            atol=ATOL,
        )


class TestGroundTruth:
    def test_dataframe_shape_columns_and_index(self, grid, query):
        asig = PeriodicSignal(grid, f=50.5, mag=1.2, phi=0.4).analytic_signal(50.0)
        gt = asig.ground_truth(query, 50.0)
        assert list(gt.columns) == ["frequency", "amplitude", "phase", "rocof", "phasor"]
        assert gt.index.name == "time"
        np.testing.assert_array_equal(gt.index.to_numpy(), query)

    def test_columns_match_the_individual_methods(self, grid, query):
        asig = PeriodicSignal(grid, f=50.5, mag=1.2, phi=0.4).analytic_signal(50.0)
        gt = asig.ground_truth(query, 50.0)
        np.testing.assert_allclose(gt["frequency"].to_numpy(), asig.frequency(query))
        np.testing.assert_allclose(gt["amplitude"].to_numpy(), asig.amplitude(query))
        np.testing.assert_allclose(gt["phase"].to_numpy(), asig.phase(query, 50.0))
        np.testing.assert_allclose(gt["rocof"].to_numpy(), asig.rocof(query))
        np.testing.assert_allclose(gt["phasor"].to_numpy(), asig.phasor(query, 50.0))

    def test_defaults_to_the_sample_grid_and_stored_nominal(self, grid):
        asig = PeriodicSignal(grid, f=50.0).analytic_signal(nominal=50.0)
        gt = asig.ground_truth()
        np.testing.assert_array_equal(gt.index.to_numpy(), grid)


class TestEstimatorAccessors:
    def test_t_is_the_time_grid(self, grid):
        asig = gen.steady_state(FS, frequency=50.3, amplitude=100.0, duration=1.0)
        np.testing.assert_array_equal(asig.t, asig.time)

    def test_channel_round_trips_the_wrapped_series(self, grid):
        asig = gen.steady_state(FS, frequency=50.3, amplitude=100.0, duration=1.0)
        np.testing.assert_array_equal(asig.channel, asig.signal.to_numpy())
        assert asig.channel.dtype == float

    def test_fs_is_inferred_from_the_index(self):
        asig = gen.steady_state(FS, frequency=50.0, duration=0.5)
        assert asig.fs == pytest.approx(FS)

    def test_fs_needs_two_samples(self):
        one = AnalyticSignal(
            pd.Series([0.0], index=pd.Index([0.0], name="time")),
            lambda t: np.ones_like(t), lambda t: np.full_like(t, 50.0),
            lambda t: TWO_PI * 50.0 * t,
        )
        with pytest.raises(ValueError):
            _ = one.fs

    def test_channel_rejects_a_multicolumn_frame(self):
        frame = pd.DataFrame(
            {"a": [1.0, 2.0], "b": [3.0, 4.0]},
            index=pd.Index([0.0, 1.0], name="time"),
        )
        asig = AnalyticSignal(
            frame, lambda t: np.ones_like(t), lambda t: np.full_like(t, 50.0),
            lambda t: TWO_PI * 50.0 * t,
        )
        with pytest.raises(ValueError):
            _ = asig.channel


class TestConcatenation:
    STEP = 1.0 / FS

    def _seg(self, t0, t1, f):
        return [PeriodicSignal(np.arange(t0, t1, self.STEP), f=f)]

    def test_waveform_matches_the_plain_concatenation(self):
        rows = [self._seg(0.0, 0.1, 50.0), self._seg(0.1, 0.2, 52.0)]
        from trimes.signal_generation import superimpose_and_concat_periodic_signals

        asig = superimpose_and_concat_analytic_signal(rows, nominal=50.0)
        plain = superimpose_and_concat_periodic_signals(
            [list(r) for r in rows], num_phases=1
        )
        pd.testing.assert_series_equal(asig.signal, plain)

    def test_fundamental_phase_is_continuous_without_a_jump(self):
        rows = [self._seg(0.0, 0.1, 50.0), self._seg(0.1, 0.2, 50.0)]
        asig = superimpose_and_concat_analytic_signal(rows, nominal=50.0)

        tb, eps = 0.1, 1e-9
        measured = (
            asig.total_phase(np.array([tb + eps]))[0]
            - asig.total_phase(np.array([tb - eps]))[0]
        )
        # only the ordinary phase advance across 2*eps, no extra jump
        expected = TWO_PI * 50.0 * (2.0 * eps)
        assert abs(measured - expected) < ATOL
        assert asig.metadata["transition_times"] == []

    def test_frequency_change_is_recorded_as_a_transition(self):
        rows = [self._seg(0.0, 0.1, 50.0), self._seg(0.1, 0.2, 52.0)]
        asig = superimpose_and_concat_analytic_signal(rows, nominal=50.0)
        assert asig.metadata["transition_times"] == pytest.approx([0.1])
        # each segment reports its own fundamental frequency
        np.testing.assert_allclose(asig.frequency(np.array([0.05])), 50.0, atol=ATOL)
        np.testing.assert_allclose(asig.frequency(np.array([0.15])), 52.0, atol=ATOL)

    def test_injected_phase_jump_matches(self):
        rows = [self._seg(0.0, 0.1, 50.0), self._seg(0.1, 0.2, 50.0)]
        asig = superimpose_and_concat_analytic_signal(
            rows, angular_jump=[np.pi / 4], nominal=50.0
        )
        tb, eps = 0.1, 1e-9
        measured = (
            asig.total_phase(np.array([tb + eps]))[0]
            - asig.total_phase(np.array([tb - eps]))[0]
        )
        expected = np.pi / 4 + TWO_PI * 50.0 * (2.0 * eps)
        assert abs(measured - expected) < ATOL
        assert asig.metadata["transition_times"] == pytest.approx([0.1])

    def test_superimposed_harmonic_does_not_change_the_truth(self, grid, query):
        rows = [
            [PeriodicSignal(grid, f=50.0, mag=1.0), PeriodicSignal(grid, f=250.0, mag=0.1)]
        ]
        asig = superimpose_and_concat_analytic_signal(rows, nominal=50.0)
        np.testing.assert_allclose(asig.frequency(query), 50.0, atol=ATOL)
        np.testing.assert_allclose(asig.amplitude(query), 1.0, atol=ATOL)
        # but the waveform carries the harmonic
        assert not np.allclose(
            asig.signal.to_numpy(),
            PeriodicSignal(grid, f=50.0, mag=1.0).get_signal_series().to_numpy(),
        )

    def test_wrong_jump_length_raises(self):
        rows = [self._seg(0.0, 0.1, 50.0), self._seg(0.1, 0.2, 50.0)]
        with pytest.raises(ValueError, match="angular_jump"):
            superimpose_and_concat_analytic_signal(rows, angular_jump=[0.1, 0.2])


class TestHelpers:
    def test_wrap_pi_range(self):
        x = np.linspace(-10.0, 10.0, 1001)
        w = wrap_pi(x)
        assert np.all(w >= -np.pi) and np.all(w < np.pi)
        # wrapping is idempotent and preserves the represented angle
        np.testing.assert_allclose(np.cos(w), np.cos(x), atol=1e-12)

    def test_rms_phasor(self):
        got = rms_phasor(np.array([2.0]), np.array([np.pi / 3]))
        np.testing.assert_allclose(got, (2.0 / np.sqrt(2)) * np.exp(1j * np.pi / 3))


class TestDependencyFootprint:
    def test_core_import_needs_only_numpy_and_pandas(self):
        script = textwrap.dedent(
            """
            import builtins
            blocked = {"scipy", "matplotlib", "control", "icecream", "sklearn", "cycler"}
            _real = builtins.__import__

            def _guard(name, *args, **kwargs):
                if name.split(".")[0] in blocked:
                    raise ModuleNotFoundError(name, name=name.split(".")[0])
                return _real(name, *args, **kwargs)

            builtins.__import__ = _guard
            import trimes
            import trimes.base
            import trimes.synchrophasor
            import trimes.synchrophasor.concat
            import trimes.synchrophasor.conditions
            import trimes.synchrophasor.limits
            import trimes.synchrophasor.metrics
            import trimes.synchrophasor.plot
            import trimes.synchrophasor.three_phase
            import trimes.signal_generation
            import trimes.transforms
            import trimes.comparison
            import trimes.metrics
            import trimes.regression
            import trimes.fourier
            import trimes.phasor
            import trimes.electrical
            import trimes.signal_processing
            import trimes.root_mean_square
            from trimes import AnalyticSignal
            from trimes.synchrophasor import (
                AnalyticSignal, build_analytic_signal, steady_state, modulation,
                concat_segments, three_phase, conditions, metrics, compliance,
                P_CLASS_LIMITS, wrap_pi, rms_phasor,
            )
            print("ok")
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True
        )
        assert result.returncode == 0, result.stderr
        assert "ok" in result.stdout

    @pytest.mark.parametrize("module", ["fft", "filter", "derivative"])
    def test_dsp_modules_raise_a_helpful_error_without_scipy(self, module):
        script = textwrap.dedent(
            f"""
            import builtins
            _real = builtins.__import__

            def _guard(name, *a, **k):
                if name.split(".")[0] == "scipy":
                    raise ModuleNotFoundError(name, name="scipy")
                return _real(name, *a, **k)

            builtins.__import__ = _guard
            try:
                import trimes.{module}
            except ModuleNotFoundError as exc:
                assert "trimes[dsp]" in str(exc), str(exc)
                print("ok")
            else:
                raise AssertionError("expected ModuleNotFoundError")
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True
        )
        assert result.returncode == 0, result.stderr
        assert "ok" in result.stdout


class TestGenerators:
    """The closed-form generators (thin wrappers over build_analytic_signal)."""

    FS = 5000.0

    def test_steady_state_truth_and_samples_are_consistent(self):
        asig = gen.steady_state(self.FS, nominal=50.0, frequency=50.5, amplitude=2.0)
        # samples equal amplitude * cos(total_phase) exactly (no integration drift)
        t = asig.time
        np.testing.assert_allclose(
            asig.signal.to_numpy(),
            asig.amplitude(t) * np.cos(asig.total_phase(t)),
            atol=1e-12,
        )
        assert asig.metadata["transition_times"] == []
        q = np.linspace(0.0, 1.0, 51)
        np.testing.assert_allclose(asig.frequency(q), 50.5, atol=ATOL)

    def test_frequency_ramp(self):
        asig = gen.frequency_ramp(
            self.FS, nominal=50.0, rocof=1.0, ramp_time=0.5, duration=1.5
        )
        q = np.array([0.0, 0.25, 0.5, 1.0, 1.4])
        np.testing.assert_allclose(
            asig.frequency(q), 50.0 + 1.0 * np.clip(q, 0.0, 0.5), atol=ATOL
        )
        np.testing.assert_allclose(asig.rocof(np.array([0.25])), 1.0, atol=ATOL)
        np.testing.assert_allclose(asig.rocof(np.array([0.9])), 0.0, atol=ATOL)
        assert asig.metadata["transition_times"] == [0.0, 0.5]

    def test_amplitude_step_transition_and_truth(self):
        asig = gen.amplitude_step(
            self.FS, amplitude=1.0, step=0.1, step_time=0.5, duration=1.0
        )
        np.testing.assert_allclose(asig.amplitude(np.array([0.25])), 1.0, atol=ATOL)
        np.testing.assert_allclose(asig.amplitude(np.array([0.75])), 1.1, atol=ATOL)
        assert asig.metadata["transition_times"] == [0.5]

    def test_phase_step_truth(self):
        asig = gen.phase_step(
            self.FS, step_deg=10.0, step_time=0.5, duration=1.0, nominal=50.0
        )
        before = asig.phase(np.array([0.25]), 50.0)[0]
        after = asig.phase(np.array([0.75]), 50.0)[0]
        assert np.isclose(wrap_pi(after - before), np.radians(10.0), atol=ATOL)
        assert asig.metadata["transition_times"] == [0.5]

    def test_modulation_matches_closed_form(self):
        asig = gen.modulation(
            self.FS, am_depth=0.1, pm_depth=0.1, modulation_frequency=2.0, duration=2.0
        )
        q = np.linspace(0.0, 2.0, 101)
        wm = TWO_PI * 2.0
        np.testing.assert_allclose(
            asig.phase(q, 50.0), wrap_pi(-0.1 * np.cos(wm * q)), atol=ATOL
        )
        np.testing.assert_allclose(
            asig.frequency(q), 50.0 + 0.1 * 2.0 * np.sin(wm * q), atol=ATOL
        )

    def test_noise_is_reproducible_and_harmonics_land_in_the_waveform(self):
        a = gen.steady_state(self.FS, noise=0.01, seed=3, harmonics={5: 0.1})
        b = gen.steady_state(self.FS, noise=0.01, seed=3, harmonics={5: 0.1})
        pd.testing.assert_series_equal(a.signal, b.signal)
        clean = gen.steady_state(self.FS)
        assert not np.allclose(a.signal.to_numpy(), clean.signal.to_numpy())
        # truth is unaffected by harmonics / noise
        np.testing.assert_allclose(a.frequency(a.time), 50.0, atol=ATOL)


class TestBuildAnalyticSignal:
    def test_samples_are_exactly_consistent_with_truth(self):
        amp = lambda t: 1.0 + 0.2 * np.asarray(t, float)
        freq = lambda t: 50.0 + 0.0 * np.asarray(t, float)
        phase = lambda t: TWO_PI * 50.0 * np.asarray(t, float)
        asig = build_analytic_signal(amp, freq, phase, fs=2000.0, duration=1.0)
        t = asig.time
        np.testing.assert_allclose(
            asig.signal.to_numpy(), amp(t) * np.cos(phase(t)), atol=1e-12
        )
        assert asig.signal.index.name == "time"
