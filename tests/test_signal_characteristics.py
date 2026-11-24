"""Detection of signal characteristics, round-tripped against the generators.

The generators of :mod:`trimes.synchrophasor` build signals whose content is
known exactly, which makes them a labelled corpus for the inverse problem. Every
test here injects something with known parameters and checks that it comes back.

Two properties matter more than the rest and are asserted throughout:

* a characteristic that was injected is always reported (a missed one would let
  an unsuitable measurement method be chosen in confidence), and
* a clean signal reports nothing at all.

The characteristics are coupled by physics, so more than one may fire for a
single injected cause -- see the module docstring. The tests therefore assert the
specific discriminations that a decision depends on rather than an exact set.
"""

import numpy as np
import pandas as pd
import pytest

from trimes.signal_characteristics import (
    CHARACTERISTICS,
    DEFAULT_THRESHOLDS,
    characterise,
    characteristics_table,
)
from trimes.synchrophasor import (
    amplitude_step,
    frequency_ramp,
    modulation,
    phase_step,
    steady_state,
    three_phase,
)

FS = 5000.0
NOMINAL = 50.0
AMPLITUDE = 100.0


def steady(duration=1.0, **content):
    """A steady three-phase set with the given content injected."""
    signal = steady_state(FS, nominal=NOMINAL, amplitude=AMPLITUDE, duration=duration)
    return three_phase(signal, **content)[0]


def present(abc, **kwargs):
    """The set of characteristics reported as present."""
    table = characteristics_table(characterise(abc, FS, nominal=NOMINAL, **kwargs))
    return set(table.index[table["present"]])


def value_of(abc, name, **kwargs):
    return characterise(abc, FS, nominal=NOMINAL, **kwargs)["findings"][name]["value"]


# -- the shape of the result -------------------------------------------------- #


class TestResultShape:
    def test_every_characteristic_is_reported(self):
        result = characterise(steady(), FS)
        assert tuple(result["findings"]) == CHARACTERISTICS
        for finding in result["findings"].values():
            assert set(finding) >= {"value", "unit", "threshold", "present", "assessable"}
            assert isinstance(finding["present"], bool)
            assert np.isfinite(finding["value"])

    def test_the_table_has_one_row_per_characteristic(self):
        table = characteristics_table(characterise(steady(), FS))
        assert list(table.index) == list(CHARACTERISTICS)
        assert {"present", "assessable", "value", "unit", "threshold"} <= set(table.columns)

    def test_the_fundamental_is_reported(self):
        result = characterise(steady(), FS)
        assert result["fundamental_frequency"] == pytest.approx(NOMINAL, abs=1e-6)
        # the positive-sequence RMS of a set whose phase peak is AMPLITUDE
        assert result["fundamental_amplitude"] == pytest.approx(AMPLITUDE / np.sqrt(2.0), rel=1e-6)

    def test_a_three_column_frame_is_accepted_as_well_as_a_mapping(self):
        abc = steady(harmonics={5: 0.04})
        frame = pd.DataFrame({phase: abc[phase].to_numpy() for phase in "abc"})
        assert present(abc) == present(frame)

    def test_thresholds_can_be_overridden(self):
        abc = steady(harmonics={5: 0.02})
        assert "harmonics" in present(abc)
        assert "harmonics" not in present(abc, thresholds={"harmonics": 0.05})


class TestRejectsBadInput:
    def test_a_missing_phase_is_an_error(self):
        abc = steady()
        with pytest.raises(ValueError, match="missing phases"):
            characterise({"a": abc["a"], "b": abc["b"]}, FS)

    def test_nan_is_rejected_rather_than_filled(self):
        abc = steady()
        broken = {phase: abc[phase].copy() for phase in "abc"}
        broken["a"].iloc[10] = np.nan
        with pytest.raises(ValueError, match="nan"):
            characterise(broken, FS)

    def test_an_empty_record_is_an_error(self):
        with pytest.raises(ValueError, match="empty"):
            characterise({p: pd.Series(dtype=float) for p in "abc"}, FS)

    def test_a_non_positive_sampling_rate_is_an_error(self):
        with pytest.raises(ValueError, match="positive"):
            characterise(steady(), 0.0)


# -- a clean signal ----------------------------------------------------------- #


def test_a_clean_signal_reports_nothing():
    """The half of the result that permits a cheap method to be used."""
    assert present(steady(duration=2.0)) == set()


def test_a_clean_signal_is_quiet_by_a_wide_margin():
    """Not merely under the thresholds, but orders of magnitude under them."""
    result = characterise(steady(duration=2.0), FS)
    for name in ("harmonics", "interharmonic", "noise", "negative_sequence",
                 "zero_sequence", "dc_offset", "am_modulation", "pm_modulation"):
        finding = result["findings"][name]
        assert finding["value"] < 0.1 * finding["threshold"], name


# -- steady content ----------------------------------------------------------- #


class TestSteadyContent:
    def test_a_harmonic_is_found_with_its_order_and_level(self):
        result = characterise(steady(harmonics={5: 0.04}), FS)
        finding = result["findings"]["harmonics"]
        assert finding["present"]
        assert finding["value"] == pytest.approx(0.04, rel=0.02)
        assert finding["detail"]["order"] == 5
        assert finding["detail"]["level"] == pytest.approx(0.04, rel=0.02)

    def test_an_interharmonic_is_found_with_its_frequency(self):
        result = characterise(steady(extra_tones={71.0: 0.08}), FS)
        finding = result["findings"]["interharmonic"]
        assert finding["present"]
        assert finding["value"] == pytest.approx(0.08, rel=0.05)
        assert finding["detail"]["frequency"] == pytest.approx(71.0, abs=1.0)

    def test_a_tone_a_few_hertz_off_is_still_found(self):
        """The hard case: a tone beside the main lobe rather than clear of it."""
        result = characterise(steady(duration=2.0, extra_tones={53.0: 0.08}), FS)
        finding = result["findings"]["interharmonic"]
        assert finding["present"]
        assert finding["value"] == pytest.approx(0.08, rel=0.10)
        assert finding["detail"]["frequency"] == pytest.approx(53.0, abs=1.0)

    @pytest.mark.parametrize("fraction", [0.02, 0.05, 0.1])
    def test_the_negative_sequence_ratio_is_recovered(self, fraction):
        assert value_of(steady(negative=fraction), "negative_sequence") == pytest.approx(
            fraction, rel=0.02
        )

    @pytest.mark.parametrize("fraction", [0.02, 0.05, 0.1])
    def test_the_zero_sequence_ratio_is_recovered(self, fraction):
        assert value_of(steady(zero=fraction), "zero_sequence") == pytest.approx(
            fraction, rel=0.02
        )

    def test_a_dc_offset_is_found(self):
        result = characterise(steady(dc_offset=0.03), FS)
        assert result["findings"]["dc_offset"]["present"]
        assert result["findings"]["dc_offset"]["value"] == pytest.approx(0.03, rel=0.05)

    def test_a_dc_offset_is_not_a_fundamental_zero_sequence(self):
        """A one-period coefficient at the fundamental rejects DC, and should."""
        assert "zero_sequence" not in present(steady(dc_offset=0.03))

    def test_noise_is_found_and_a_quiet_signal_is_not_called_noisy(self):
        assert "noise" in present(steady(noise=0.02))
        assert "noise" not in present(steady(noise=0.0))


# -- the discriminations a decision depends on -------------------------------- #


class TestDiscriminations:
    """These are the pairs that must not be confused, because a method is chosen
    on the difference between them."""

    def test_a_harmonic_is_not_reported_as_an_interharmonic(self):
        """A window of whole periods rejects one and not the other, so confusing
        them would invert the choice of method."""
        found = present(steady(duration=2.0, harmonics={5: 0.05}))
        assert "harmonics" in found
        assert "interharmonic" not in found

    def test_an_interharmonic_is_not_reported_as_a_harmonic(self):
        found = present(steady(duration=2.0, extra_tones={71.0: 0.08}))
        assert "interharmonic" in found
        assert "harmonics" not in found

    def test_a_frequency_ramp_is_not_reported_as_a_phase_step(self):
        """A ramp moves the phase continuously, which a first difference cannot
        tell from a step."""
        signal = frequency_ramp(FS, NOMINAL, AMPLITUDE, rocof=1.0, ramp_time=1.0, duration=2.0)
        found = present(three_phase(signal)[0])
        assert "frequency_ramp" in found
        assert "phase_step" not in found

    def test_a_step_is_not_reported_as_a_dip(self):
        signal = amplitude_step(FS, NOMINAL, AMPLITUDE, step=0.2, step_time=0.5, duration=1.0)
        found = present(three_phase(signal)[0])
        assert "magnitude_step" in found
        assert "dip" not in found

    def test_an_off_nominal_frequency_is_not_reported_as_an_unbalance(self):
        """The sequence window is sized from the measured fundamental, so an
        off-nominal signal does not leak into a negative sequence."""
        signal = steady_state(FS, nominal=NOMINAL, frequency=52.0, amplitude=AMPLITUDE, duration=1.0)
        found = present(three_phase(signal)[0])
        assert "off_nominal_frequency" in found
        assert "negative_sequence" not in found
        assert "zero_sequence" not in found


# -- dynamics ----------------------------------------------------------------- #


class TestDynamics:
    def test_an_off_nominal_frequency_is_measured(self):
        signal = steady_state(FS, nominal=NOMINAL, frequency=52.0, amplitude=AMPLITUDE, duration=1.0)
        result = characterise(three_phase(signal)[0], FS, nominal=NOMINAL)
        assert result["findings"]["off_nominal_frequency"]["present"]
        assert result["findings"]["off_nominal_frequency"]["value"] == pytest.approx(2.0, abs=0.02)
        assert result["fundamental_frequency"] == pytest.approx(52.0, abs=0.02)

    @pytest.mark.parametrize("rocof", [1.0, 2.0])
    def test_a_frequency_ramp_is_measured(self, rocof):
        signal = frequency_ramp(FS, NOMINAL, AMPLITUDE, rocof=rocof, ramp_time=1.0, duration=2.0)
        result = characterise(three_phase(signal)[0], FS, nominal=NOMINAL)
        assert result["findings"]["frequency_ramp"]["present"]
        assert result["findings"]["frequency_ramp"]["value"] == pytest.approx(rocof, rel=0.10)

    @pytest.mark.parametrize("degrees", [10.0, 30.0, -45.0])
    def test_a_phase_step_is_measured(self, degrees):
        signal = phase_step(FS, NOMINAL, AMPLITUDE, step_deg=degrees, step_time=0.5, duration=1.0)
        result = characterise(three_phase(signal)[0], FS, nominal=NOMINAL)
        finding = result["findings"]["phase_step"]
        assert finding["present"]
        assert finding["value"] == pytest.approx(abs(degrees), rel=0.02)
        assert finding["interval"][0] == pytest.approx(0.5, abs=0.15)

    @pytest.mark.parametrize("step", [0.05, 0.2, -0.3])
    def test_a_magnitude_step_is_measured(self, step):
        signal = amplitude_step(FS, NOMINAL, AMPLITUDE, step=step, step_time=0.5, duration=1.0)
        result = characterise(three_phase(signal)[0], FS, nominal=NOMINAL)
        finding = result["findings"]["magnitude_step"]
        assert finding["present"]
        assert finding["value"] == pytest.approx(abs(step), rel=0.05)

    def test_modulation_depth_and_frequency_are_measured(self):
        signal = modulation(
            FS, NOMINAL, AMPLITUDE, am_depth=0.1, pm_depth=0.1,
            modulation_frequency=2.0, duration=3.0,
        )
        result = characterise(three_phase(signal)[0], FS, nominal=NOMINAL)
        for name in ("am_modulation", "pm_modulation"):
            finding = result["findings"][name]
            assert finding["present"], name
            assert finding["value"] == pytest.approx(0.1, rel=0.05), name
            assert finding["detail"]["frequency"] == pytest.approx(2.0, abs=0.1), name

    def test_an_amplitude_modulation_alone_is_not_read_as_a_phase_modulation(self):
        signal = modulation(
            FS, NOMINAL, AMPLITUDE, am_depth=0.1, pm_depth=0.0,
            modulation_frequency=2.0, duration=3.0,
        )
        result = characterise(three_phase(signal)[0], FS, nominal=NOMINAL)
        assert result["findings"]["am_modulation"]["present"]
        assert not result["findings"]["pm_modulation"]["present"]


# -- dips and swells ---------------------------------------------------------- #


class TestDipsAndSwells:
    @pytest.mark.parametrize("level", [0.8, 0.5, 0.2])
    def test_a_dip_is_measured_against_the_level_before_it(self, level):
        """Not against the average of the record: a record that spends half its
        length dipped has an average part way into the dip."""
        signal = amplitude_step(
            FS, NOMINAL, AMPLITUDE, step=level - 1.0, step_time=0.5, duration=1.0
        )
        result = characterise(three_phase(signal)[0], FS, nominal=NOMINAL)
        assert result["findings"]["dip"]["present"]
        assert result["findings"]["dip"]["value"] == pytest.approx(level, rel=0.05)

    def test_a_shallow_dip_is_not_called_one(self):
        signal = amplitude_step(FS, NOMINAL, AMPLITUDE, step=-0.05, step_time=0.5, duration=1.0)
        assert "dip" not in present(three_phase(signal)[0])

    def test_a_swell_is_measured(self):
        signal = amplitude_step(FS, NOMINAL, AMPLITUDE, step=0.2, step_time=0.5, duration=1.0)
        result = characterise(three_phase(signal)[0], FS, nominal=NOMINAL)
        assert result["findings"]["swell"]["present"]
        assert result["findings"]["swell"]["value"] == pytest.approx(1.2, rel=0.05)

    def test_an_interruption_is_distinguished_from_a_dip(self):
        """IEC 61000-4-30 puts the interruption threshold at 0.1 pu, so a dip to
        0.2 pu is a dip and not an interruption."""
        shallow = amplitude_step(FS, NOMINAL, AMPLITUDE, step=-0.8, step_time=0.5, duration=1.0)
        deep = amplitude_step(FS, NOMINAL, AMPLITUDE, step=-0.95, step_time=0.5, duration=1.0)
        assert "interruption" not in present(three_phase(shallow)[0])
        assert "interruption" in present(three_phase(deep)[0])


# -- honesty about what cannot be measured ------------------------------------ #


class TestAssessability:
    def test_a_steady_record_can_assess_everything(self):
        result = characterise(steady(duration=2.0, harmonics={5: 0.04}), FS)
        assert result["stationary"]
        assert all(f["assessable"] for f in result["findings"].values())

    def test_a_moving_fundamental_blocks_the_spectral_characteristics(self):
        """Fitting one fixed tone to a ramping fundamental leaves a residual that
        is the ramp, not extra content, so reporting a number would be an
        artefact of the method."""
        signal = frequency_ramp(FS, NOMINAL, AMPLITUDE, rocof=2.0, ramp_time=1.0, duration=2.0)
        result = characterise(three_phase(signal)[0], FS, nominal=NOMINAL)
        assert not result["stationary"]
        for name in ("harmonics", "interharmonic", "noise"):
            assert not result["findings"][name]["assessable"], name
            assert "frequency_ramp" in result["findings"][name]["limited_by"]
        # the dynamics themselves remain assessable
        assert result["findings"]["frequency_ramp"]["assessable"]
        assert result["findings"]["dip"]["assessable"]

    def test_thresholds_cover_every_characteristic(self):
        """A characteristic without a threshold would raise at record time."""
        assert set(DEFAULT_THRESHOLDS) == set(CHARACTERISTICS)


# -- the steady window -------------------------------------------------------- #


def test_the_steady_interval_avoids_an_event():
    """The spectral characteristics are measured away from a step, because a step
    spreads energy across the spectrum and would read as noise and as a tone."""
    signal = phase_step(FS, NOMINAL, AMPLITUDE, step_deg=45.0, step_time=0.5, duration=2.0)
    result = characterise(three_phase(signal)[0], FS, nominal=NOMINAL)
    start, stop = result["steady_interval"]
    assert result["events"]
    assert not (start < 0.5 < stop)


def test_a_record_that_is_never_quiet_still_returns_a_result():
    """Falling back to the whole record is better than raising: the caller still
    needs the dynamics, which do not depend on the steady window."""
    signal = modulation(
        FS, NOMINAL, AMPLITUDE, am_depth=0.3, pm_depth=0.3,
        modulation_frequency=5.0, duration=1.0,
    )
    result = characterise(three_phase(signal)[0], FS, nominal=NOMINAL)
    assert result["findings"]["am_modulation"]["present"]
    assert np.isfinite(result["findings"]["noise"]["value"])
