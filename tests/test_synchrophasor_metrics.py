"""Phase 5: synchrophasor evaluation metrics."""

import numpy as np
import pandas as pd
import pytest

from trimes.synchrophasor import conditions, metrics, steady_state
from trimes.synchrophasor.util import rms_phasor

FS = 10_000.0


def _trace_from_truth(truth, times, *, phasor_err=0.0, freq_err=0.0, rocof_err=0.0, seed=0):
    """Build an estimator-style trace by perturbing the truth at `times`."""
    gt = truth.ground_truth(times)
    rng = np.random.default_rng(seed)
    n = len(times)
    phasor = gt["phasor"].to_numpy() + phasor_err * (
        rng.standard_normal(n) + 1j * rng.standard_normal(n)
    )
    return pd.DataFrame(
        {
            "phasor": phasor,
            "frequency": gt["frequency"].to_numpy() + freq_err * rng.standard_normal(n),
            "rocof": gt["rocof"].to_numpy() + rocof_err * rng.standard_normal(n),
        },
        index=pd.Index(times, name="time"),
    )


@pytest.fixture
def steady_truth():
    return conditions.frequency_range(FS, "P", fdev=1.0, duration=1.0)


@pytest.fixture
def report_times(steady_truth):
    return np.arange(0.05, steady_truth.duration - 0.02, 0.02)


class TestErrors:
    def test_perfect_trace_has_zero_error(self, steady_truth, report_times):
        trace = _trace_from_truth(steady_truth, report_times)
        e = metrics.errors(trace, steady_truth)
        np.testing.assert_allclose(e["tve"].to_numpy(), 0.0, atol=1e-9)
        np.testing.assert_allclose(e["fe"].to_numpy(), 0.0, atol=1e-9)
        np.testing.assert_allclose(e["rfe"].to_numpy(), 0.0, atol=1e-9)

    def test_tve_of_a_known_magnitude_error(self, steady_truth, report_times):
        gt = steady_truth.ground_truth(report_times)
        trace = pd.DataFrame(
            {
                "phasor": gt["phasor"].to_numpy() * 1.01,  # +1 % magnitude
                "frequency": gt["frequency"].to_numpy(),
                "rocof": gt["rocof"].to_numpy(),
            },
            index=pd.Index(report_times, name="time"),
        )
        np.testing.assert_allclose(
            metrics.tve(trace, steady_truth).to_numpy(), 1.0, atol=1e-6
        )

    def test_fe_is_absolute_frequency_error(self, steady_truth, report_times):
        gt = steady_truth.ground_truth(report_times)
        trace = pd.DataFrame(
            {
                "phasor": gt["phasor"].to_numpy(),
                "frequency": gt["frequency"].to_numpy() + 0.01,
                "rocof": gt["rocof"].to_numpy(),
            },
            index=pd.Index(report_times, name="time"),
        )
        np.testing.assert_allclose(
            metrics.fe(trace, steady_truth).to_numpy(), 0.01, atol=1e-9
        )

    def test_missing_columns_give_nan(self, steady_truth, report_times):
        gt = steady_truth.ground_truth(report_times)
        trace = pd.DataFrame(
            {"phasor": gt["phasor"].to_numpy()},
            index=pd.Index(report_times, name="time"),
        )
        e = metrics.errors(trace, steady_truth)
        assert e["fe"].isna().all()
        assert e["rfe"].isna().all()


class TestSummary:
    def test_max_and_rms(self, steady_truth, report_times):
        trace = _trace_from_truth(
            steady_truth, report_times, phasor_err=0.002, freq_err=0.001, seed=1
        )
        s = metrics.summary(trace, steady_truth)
        assert s["max_tve"] >= s["rms_tve"] > 0.0
        assert s["frames"] == len(report_times)
        assert s["excluded_frames"] == 0

    def test_transition_exclusion_drops_frames(self):
        step = conditions.magnitude_step(FS, "P", step_time=0.5, duration=1.0)
        times = np.arange(0.05, 0.95, 0.02)
        trace = _trace_from_truth(step, times, phasor_err=0.001, seed=2)
        wide = metrics.summary(trace, step, exclude_around_transitions=0.1)
        narrow = metrics.summary(trace, step, exclude_around_transitions=0.0)
        assert wide["excluded_frames"] > 0
        assert wide["frames"] < narrow["frames"]

    def test_ramp_uses_explicit_exclusion_windows(self):
        ramp = conditions.frequency_ramp(FS, "P")
        times = np.arange(0.05, ramp.duration - 0.05, 0.02)
        trace = _trace_from_truth(ramp, times, phasor_err=0.001, seed=3)
        s = metrics.summary(trace, ramp, exclude_around_transitions=0.0)
        # metadata["exclusion"] is honoured even though the half-width is 0
        assert s["excluded_frames"] > 0


class TestStepResponse:
    @pytest.fixture
    def step(self):
        return conditions.magnitude_step(FS, "P", step=0.1, step_time=0.5, duration=1.5)

    @pytest.fixture
    def lagging_trace(self, step):
        """A first-order-ish lag: amplitude and phasor relax to the new level."""
        times = np.arange(0.02, step.duration - 0.02, 0.01)
        gt = step.ground_truth(times)
        tau = 0.03
        s = np.clip(times - 0.5, 0.0, None)
        relax = 1.0 - np.exp(-s / tau)
        a0 = gt["amplitude"].to_numpy()[0]
        a1 = gt["amplitude"].to_numpy()[-1]
        amp = a0 + (a1 - a0) * relax
        phase = step.phase(times)
        return pd.DataFrame(
            {
                "phasor": rms_phasor(amp, phase),
                "amplitude": amp,
                "frequency": gt["frequency"].to_numpy(),
                "rocof": gt["rocof"].to_numpy(),
            },
            index=pd.Index(times, name="time"),
        )

    def test_response_time_is_positive_and_bounded(self, step, lagging_trace):
        rt = metrics.response_time(lagging_trace, step, band=1.0, quantity="tve")
        assert 0.0 < rt < 0.5

    def test_delay_time_near_a_quarter_of_tau(self, step, lagging_trace):
        dt = metrics.delay_time(lagging_trace, step, quantity="amplitude")
        # midpoint of a 1-exp relaxation is reached at tau*ln(2) ~ 0.021 s
        assert dt == pytest.approx(0.03 * np.log(2), abs=0.01)

    def test_overshoot_is_zero_for_a_monotone_relaxation(self, step, lagging_trace):
        assert metrics.overshoot(lagging_trace, step, quantity="amplitude") == pytest.approx(
            0.0, abs=1e-6
        )

    def test_overshoot_detects_an_overshoot(self, step):
        times = np.arange(0.02, step.duration - 0.02, 0.01)
        gt = step.ground_truth(times)
        amp = gt["amplitude"].to_numpy().copy()
        after = times >= 0.5
        final = gt["amplitude"].to_numpy()[-1]
        step_size = final - gt["amplitude"].to_numpy()[0]
        # inject a spike of 20 % of the step size past the final value
        idx = np.argmax(after)
        amp[idx : idx + 3] = final + 0.2 * step_size
        trace = pd.DataFrame(
            {"phasor": rms_phasor(amp, step.phase(times)), "amplitude": amp},
            index=pd.Index(times, name="time"),
        )
        assert metrics.overshoot(trace, step, quantity="amplitude") == pytest.approx(
            0.2, rel=0.05
        )


class TestCompliance:
    def test_pass_table_for_a_clean_trace(self, steady_truth, report_times):
        trace = _trace_from_truth(steady_truth, report_times, phasor_err=1e-5, seed=4)
        table = metrics.compliance(
            metrics.summary(trace, steady_truth), steady_truth.metadata
        )
        assert list(table["metric"]) == ["tve", "fe", "rfe"]
        assert table["pass"].all()

    def test_fail_when_over_the_limit(self, steady_truth, report_times):
        trace = _trace_from_truth(steady_truth, report_times, phasor_err=0.05, seed=5)
        table = metrics.compliance(
            metrics.summary(trace, steady_truth), steady_truth.metadata
        )
        assert not table.loc[table["metric"] == "tve", "pass"].iloc[0]

    def test_exempt_condition_gives_empty_table(self):
        oob = conditions.out_of_band(FS, "P")
        table = metrics.compliance({"max_tve": 0.0}, oob.metadata)
        assert table.empty
