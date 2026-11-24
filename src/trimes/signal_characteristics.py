"""Detect what a measured three-phase record contains.

This module is the inverse of :mod:`trimes.signal_generation` and of
:mod:`trimes.synchrophasor.three_phase`. Those build a record from a list of
declared characteristics. :func:`characterise` takes a record whose content is
*not* declared and measures which of the same characteristics are present, so
that a later step can be chosen with knowledge of the signal rather than by
assumption.

The intended use is diagnostic. A quantity such as an RMS voltage or a power is
computed from a waveform by one of several methods, and the methods differ most
where the waveform departs from a clean balanced sinusoid. Knowing *how* a
particular record departs from one says which methods are safe to apply to it and
which are not.

Conventions
-----------
The set of characteristics follows the power-quality parameters of
IEC 61000-4-30 -- frequency, magnitude, dips and swells, interruptions,
unbalance, harmonics and interharmonics -- extended by the dynamic conditions of
IEC/IEEE 60255-118-1 that :mod:`trimes.synchrophasor.conditions` generates:
frequency ramps, phase and magnitude steps, and amplitude and phase modulation.

Every characteristic is reported as a magnitude *and* a boolean verdict against a
threshold. The magnitudes are measurements and stand on their own. The verdicts
depend on :data:`DEFAULT_THRESHOLDS` and are a convenience, not a standard: where
a standard fixes a threshold it is used and cited, and where none does the
default is a round number chosen to sit well above the numerical floor of the
measurement. Pass ``thresholds`` to override any of them.

Relative quantities are fractions of the fundamental, not percentages, so 0.05
means five percent. Frequencies are in hertz and times in seconds. The record
must be uniformly sampled, which is the contract of the whole package, and must
not contain ``nan``.

A verdict of *not present* is the useful half of this module. It is what permits
a cheap method to be used, so the thresholds are set to make a false *present*
more likely than a false *absent*, and the tests assert the absence of every
characteristic that was not injected.

Notes
-----
This module needs :mod:`trimes.fft` and therefore scipy. It is not part of the
numpy-only core.

The measurement of every characteristic is a window average or a spectrum over
the steady part of the record, so a disturbance that is present for a short part
of a long record is diluted. Where a characteristic has a natural location -- a
step, a dip -- the interval in which it was found is reported alongside it.

The characteristics are **not mutually exclusive, and several of them are
coupled by physics rather than by any shortcoming of the measurement.** More than
one being reported is normal and is not a contradiction:

* An interharmonic beats against the fundamental, so a phasor taken over one
  period oscillates in both amplitude and phase at the difference frequency.
  ``am_modulation``, ``pm_modulation`` and the two step characteristics therefore
  respond to an interharmonic as well.
* An amplitude modulation of depth 0.1 reaches 1.1 pu, which is the threshold
  IEC 61000-4-30 sets for a swell, so ``swell`` responds to a deep modulation.
* A phase modulation moves the frequency, so ``frequency_ramp`` responds to it.

What the module does guarantee, and what the tests assert, is the other
direction: a characteristic that *was* injected is always reported, a clean
signal reports nothing, and the discriminations that decide how a quantity should
be measured are kept apart -- a harmonic is never reported as an interharmonic or
the reverse, a frequency ramp is never reported as a phase step, and a step is
never reported as a dip.

Where the fundamental itself is moving, a spectrum taken by fitting and
subtracting one fixed tone cannot separate the movement of the fundamental from
genuine extra content. Those findings are then marked ``assessable = False``
rather than given a number, and a caller deciding whether something is safe must
read that as *possibly present*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trimes import fft as _fft
from trimes import phasor as _phasor
from trimes.base import FrameOrSeries, get_sampling_rate, is_uniformly_sampled
from trimes.transforms import abc_2_symmetrical_components

__all__ = [
    "CHARACTERISTICS",
    "DEFAULT_THRESHOLDS",
    "characterise",
    "characteristics_table",
]

_PHASES = ("a", "b", "c")
#: Column order of :func:`trimes.transforms.abc_2_symmetrical_components`.
_ZERO, _POSITIVE, _NEGATIVE = 0, 1, 2

#: The characteristics :func:`characterise` reports, in table order.
CHARACTERISTICS = (
    "off_nominal_frequency",
    "frequency_ramp",
    "phase_step",
    "magnitude_step",
    "dip",
    "swell",
    "interruption",
    "harmonics",
    "interharmonic",
    "negative_sequence",
    "zero_sequence",
    "dc_offset",
    "noise",
    "am_modulation",
    "pm_modulation",
)

DEFAULT_THRESHOLDS = {
    # IEC 61000-4-30 measures the frequency to 10 mHz; ten times that is safely
    # above the resolution of a window of a few periods.
    "off_nominal_frequency": 0.1,
    # A quarter of the 2 Hz/s that grid codes treat as a severe rate of change.
    "frequency_ramp": 0.5,
    # Well above the jitter of a phase track on a clean signal, well below the
    # 10 degrees of the smallest step the synchrophasor standard tests.
    "phase_step": 2.0,
    "magnitude_step": 0.02,
    # IEC 61000-4-30: a dip is below 0.9 pu and a swell above 1.1 pu.
    "dip": 0.9,
    "swell": 1.1,
    # IEC 61000-4-30 uses 0.1 pu as the interruption threshold for a dip that is
    # counted as an interruption.
    "interruption": 0.1,
    # IEC 61000-4-7 and most connection codes put the planning level for the
    # total harmonic distortion of a transmission voltage near this.
    "harmonics": 0.01,
    "interharmonic": 0.005,
    # EN 50160 limits the negative-sequence unbalance of a supply to 2 percent.
    "negative_sequence": 0.01,
    "zero_sequence": 0.01,
    "dc_offset": 0.01,
    "noise": 0.005,
    "am_modulation": 0.01,
    "pm_modulation": 0.01,
}
"""Default thresholds for the boolean verdicts, as fractions or in native units.

Where a standard fixes a value it is used: the dip and swell levels and the
interruption level come from IEC 61000-4-30, the unbalance limit from EN 50160
and the distortion level from IEC 61000-4-7. The rest are round numbers chosen
to sit above the numerical floor of the measurement on a clean signal.
"""

_UNITS = {
    "off_nominal_frequency": "Hz",
    "frequency_ramp": "Hz/s",
    "phase_step": "deg",
    "dip": "pu",
    "swell": "pu",
    "interruption": "pu",
    "pm_modulation": "rad",
}
"""Unit of a characteristic where it is not a fraction of the fundamental."""

#: Verdicts that fire when the value falls *below* the threshold rather than above.
_BELOW = frozenset({"dip", "interruption"})

#: Fundamental periods excluded either side of a detected event before the steady
#: characteristics are measured. Five is the longest window any method in this
#: package uses, so a window that straddles an event is excluded with it.
_EXCLUSION_PERIODS = 5.0

#: Baseline over which a frequency ramp is measured, in seconds. Long enough that
#: the jitter of a frequency track averages out, short enough to catch a ramp
#: that lasts only a few hundred milliseconds.
_RAMP_BASELINE_S = 0.2

#: Fundamental periods per block for the spectral characteristics. IEC 61000-4-7
#: aggregates harmonics over ten cycles, but ten cycles resolve only 5 Hz and an
#: interharmonic a few hertz from the fundamental is exactly the case that matters
#: most, so the block is longer: twenty-five periods is half a second at 50 Hz and
#: resolves 2 Hz, which separates a tone 3 Hz away from the fundamental.
_BLOCK_PERIODS = 25

#: Beyond this many detected events, the record is taken to carry a *periodic*
#: disturbance rather than a sequence of discrete events. A steady interharmonic
#: beats against the fundamental and makes a one-period phasor oscillate
#: continuously, which a level-change detector reports as an event per beat. Left
#: uncorrected, the exclusion around each of them removes the whole disturbed
#: region and the disturbance is measured only where it is absent.
_MAX_EVENTS = 12

#: How far a spectral peak must stand above the median of the residual spectrum
#: to count as a discrete tone rather than as the extreme of a noise floor. A
#: sinusoid in a Hann-windowed spectrum rises orders of magnitude above the
#: surrounding bins, so this is not a demanding test, but without it a record
#: carrying only broadband noise reports its noisiest bin as an interharmonic.
_TONE_PROMINENCE = 8.0


def _ramp(track: np.ndarray, times: np.ndarray, baseline_s: float) -> tuple:
    """Steepest sustained slope of a track, and where it was found.

    Returns ``(hz_per_s, {"interval": ...})`` ready to splat into ``record``. The
    slope is taken over ``baseline_s`` rather than between neighbouring samples,
    because the jitter of a frequency track differentiates into a slope far
    larger than any real ramp while contributing almost nothing over a baseline.
    """
    if track.size < 4:
        return 0.0, {}
    step = float(np.median(np.diff(times))) if times.size > 1 else 0.0
    span = max(1, int(round(baseline_s / step))) if step > 0.0 else 1
    if track.size <= span:
        span = track.size - 1
    # Medians over a third of the baseline at each end, not the endpoints
    # themselves. An interharmonic beating against the fundamental makes the
    # frequency track oscillate, and a difference of two instantaneous values
    # reports that oscillation as a ramp. Over a third of a baseline the
    # oscillation averages out and a genuine ramp does not.
    third = max(1, span // 3)
    if track.size > span + third:
        windows = np.lib.stride_tricks.sliding_window_view(track, third)
        level = np.median(windows, axis=1)
        track = level
        times = times[: level.size]
        if track.size <= span:
            span = max(1, track.size - 1)
    elapsed = times[span:] - times[:-span]
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = (track[span:] - track[:-span]) / elapsed
    finite = np.isfinite(slope)
    if not finite.any():
        return 0.0, {}
    index = int(np.nanargmax(np.abs(np.where(finite, slope, 0.0))))
    return float(abs(slope[index])), {
        "interval": (float(times[index]), float(times[index + span])),
        "detail": {"span": float(np.ptp(track[np.isfinite(track)]))},
    }


def _local_fundamental(window: pd.DataFrame, nominal: float, fallback: float) -> float:
    """The fundamental frequency within one quiet stretch of a record."""
    try:
        estimate = _phasor.estimate_frequency(window["a"], nominal=nominal)
        value = float(np.asarray(estimate["frequency"], dtype=float).ravel()[0])
    except Exception:  # pragma: no cover - a degenerate window
        return fallback
    # a estimate far from nominal is more likely a failed fit than a real
    # excursion, so fall back rather than size a window from it
    return value if np.isfinite(value) and abs(value - nominal) < 0.2 * nominal else fallback


def _whole_periods(frame: pd.DataFrame, lo: float, hi: float, samples: int) -> pd.DataFrame:
    """The part of ``frame`` in ``[lo, hi]``, trimmed to whole fundamental periods.

    Otherwise the fundamental falls between two spectral bins and leaks across the
    spectrum, which shows up as a distortion of a percent or two and as a spurious
    tone a few hertz away. This is the reason the power-quality standards specify
    a window of a whole number of cycles.
    """
    inside = frame.loc[(frame.index >= lo) & (frame.index <= hi)]
    whole = (len(inside) // samples) * samples
    return inside.iloc[:whole] if whole >= samples else inside


def _reference_amplitude(magnitude, times, events, period) -> float:
    """The operating point a dip or a swell is measured against.

    IEC 61000-4-30 measures a dip against a declared or a sliding reference
    voltage, not against the average of the record that contains the dip. The
    level before the first event serves as that reference here, because a record
    that spends half its length dipped has a median part way into the dip and
    would report the dip as shallower than it is. With no event, the median of
    the whole record is the operating point.
    """
    if not magnitude.size:
        return 0.0
    if events:
        before = times < events[0] - period
        if before.sum() >= 4:
            return float(np.median(magnitude[before]))
    return float(np.median(magnitude))


def _as_frame(abc) -> pd.DataFrame:
    """Coerce a phase mapping or a three-column frame to an ``abc`` frame."""
    if isinstance(abc, pd.DataFrame):
        missing = [p for p in _PHASES if p not in abc.columns]
        if missing:
            if abc.shape[1] != 3:
                raise ValueError(
                    f"expected columns {list(_PHASES)} or exactly three columns, "
                    f"got {list(abc.columns)}"
                )
            abc = abc.set_axis(list(_PHASES), axis=1)
        return abc.loc[:, list(_PHASES)].astype(float)
    if hasattr(abc, "keys"):
        missing = [p for p in _PHASES if p not in abc]
        if missing:
            raise ValueError(f"missing phases {missing} in {sorted(abc)}")
        columns = {}
        for phase in _PHASES:
            values = abc[phase]
            columns[phase] = (
                values.to_numpy(dtype=float)
                if isinstance(values, pd.Series)
                else np.asarray(values, dtype=float)
            )
        first = abc["a"]
        index = first.index if isinstance(first, pd.Series) else None
        return pd.DataFrame(columns, index=index)
    raise TypeError(f"expected a mapping of phases or a DataFrame, got {type(abc).__name__}")


def _prepare(abc, fs: float | None) -> tuple[pd.DataFrame, float]:
    """Validate the record and resolve the sampling rate."""
    frame = _as_frame(abc)
    if frame.empty:
        raise ValueError("the record is empty")
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError("the record contains nan or inf, which this module does not fill")
    if fs is None:
        fs = get_sampling_rate(frame, check_uniform=True)
    else:
        fs = float(fs)
        if fs <= 0.0:
            raise ValueError(f"fs must be positive, got {fs}")
    if not isinstance(frame.index, pd.RangeIndex) and frame.index.size > 2:
        if not is_uniformly_sampled(frame):
            raise ValueError("the record is not uniformly sampled")
    times = np.arange(frame.index.size, dtype=float) / fs
    return frame.set_axis(pd.Index(times, name="time")), fs


def _sliding_phasor(values: np.ndarray, samples: int) -> np.ndarray:
    """Rolling fundamental Fourier coefficient, one value per sample.

    Numerically identical to :func:`trimes.fourier.get_fourier_coef_rolling` with
    ``k = 1`` (to about 1e-15 relative), but evaluated as a single correlation
    rather than one window at a time, which is some eight hundred times faster
    and is what makes this module usable on a record of any length.

    The leading samples that have no full window repeat the first full value, so
    the result has the length of the input. Callers drop them.
    """
    kernel = (np.exp(-2j * np.pi * np.arange(samples) / samples) * (np.sqrt(2.0) / samples))[::-1]
    # the trailing factor aligns the phase reference with the window convention
    # of trimes.fourier, which is one sample further on than a plain correlation
    valid = np.convolve(values, kernel, mode="valid") * np.exp(-2j * np.pi / samples)
    out = np.empty(values.size, dtype=complex)
    out[: samples - 1] = valid[0]
    out[samples - 1 :] = valid
    return out


def _sequence_traces(frame: pd.DataFrame, fs: float, fundamental: float) -> pd.DataFrame:
    """Sliding symmetrical-component phasors of the fundamental.

    One window of a fundamental period per sample, so the traces follow the
    record. The window is sized from the *measured* fundamental rather than the
    nominal one: a window of a nominal period does not span a whole period of an
    off-nominal signal, and the leakage that follows appears as a negative
    sequence that the signal does not have.

    Returns a frame with the complex columns ``zero``, ``positive`` and
    ``negative``, the leading partial windows dropped.
    """
    samples = max(2, int(round(fs / fundamental)))
    # abc_2_symmetrical_components takes the phases as columns, unlike the
    # Clarke and Park transforms in the same module, which take them as rows
    stacked = np.column_stack(
        [_sliding_phasor(frame[phase].to_numpy(dtype=float), samples) for phase in _PHASES]
    )
    sequence = abc_2_symmetrical_components(stacked)
    result = pd.DataFrame(
        {
            "zero": sequence[:, _ZERO],
            "positive": sequence[:, _POSITIVE],
            "negative": sequence[:, _NEGATIVE],
        },
        index=frame.index,
    )
    return result.iloc[samples:]


def _away_from_events(times: np.ndarray, events, half_width: float) -> np.ndarray:
    """Mask of the instants that are more than ``half_width`` from any event."""
    keep = np.ones(np.shape(times), dtype=bool)
    for event in events:
        keep &= np.abs(times - event) > half_width
    return keep


def _largest_jump(
    values: np.ndarray,
    times: np.ndarray,
    span: int,
    *,
    detrend: bool = False,
    lag: int | None = None,
) -> tuple[float, float | None]:
    """Largest *sustained* change of ``values``, and where it happens.

    The level each side of a candidate instant is the median over ``span``
    samples, not the instantaneous value. That matters whenever the record
    carries a steady oscillation: an interharmonic makes a one-period phasor beat
    at the difference frequency, and comparing instantaneous values across a
    period reports that beat as a step. A median over a period averages the beat
    away and leaves a genuine step untouched.

    With ``detrend``, the second difference is used instead of the first. A
    quantity that ramps has a constant first difference and therefore a second
    difference of zero, while a step of size ``S`` has a second difference of
    ``S`` whatever the trend it sits on. This is what keeps a frequency ramp,
    which moves the phase continuously, from being reported as a phase step.

    ``lag`` separates the two median windows by more than their own length,
    leaving a gap between them. A one-period window responds to *any* abrupt
    event, so a phase step briefly disturbs the magnitude as well, and comparing
    levels a single period apart reports that disturbance as a magnitude step. A
    gap places the disturbance between the two windows, where it is ignored,
    while a settled change of level is still seen in full.
    """
    lag = span if lag is None else lag
    if values.size <= span + lag:
        return 0.0, None
    # rolling median of length `span`, aligned so that entry i covers [i, i+span)
    windows = np.lib.stride_tricks.sliding_window_view(values, span)
    level = np.median(windows, axis=1)
    if level.size <= lag:
        return 0.0, None
    if detrend:
        if level.size <= 2 * lag:
            return 0.0, None
        difference = level[2 * lag :] - 2.0 * level[lag:-lag] + level[: -2 * lag]
        offset = 2 * lag
    else:
        difference = level[lag:] - level[:-lag]
        offset = lag
    if not np.isfinite(difference).any():
        return 0.0, None
    index = int(np.nanargmax(np.abs(difference)))
    return float(np.abs(difference[index])), float(times[min(index + offset, times.size - 1)])


def _fit_tone(channel: np.ndarray, t: np.ndarray, frequency: float) -> np.ndarray:
    """Least-squares fit of a single tone at a known frequency, as a waveform."""
    design = np.column_stack(
        [np.cos(2.0 * np.pi * frequency * t), np.sin(2.0 * np.pi * frequency * t)]
    )
    coefficients, *_ = np.linalg.lstsq(design, channel, rcond=None)
    return design @ coefficients


def _dominant_tone(
    channel: np.ndarray, fs: float, fundamental: float, *, guard: float, max_order: int
) -> tuple[float, float]:
    """Largest tone that is neither the fundamental nor one of its harmonics.

    Returns ``(frequency, amplitude)`` with the amplitude relative to the
    fundamental's, or ``(nan, 0.0)`` when nothing is left once the fundamental
    and its harmonics have been taken out.

    This is the measurement that separates an interharmonic from a harmonic, and
    it is the distinction that matters most when choosing how to measure a
    quantity from the record: a window of a whole number of fundamental periods
    rejects a harmonic exactly and an interharmonic not at all.

    The fundamental and every harmonic within range are **subtracted in the time
    domain** rather than masked out of the spectrum. Masking cannot work here: the
    fundamental is two orders of magnitude larger than the tone being looked for,
    and unless the window happens to hold an exact number of its periods it leaks
    across far more bins than any guard band would cover. Subtracting a fitted
    tone removes that leakage with it. This is the same reasoning that leads the
    iterative interpolated DFT of the synchrophasor literature to subtract each
    component it identifies.
    """
    if channel.size < 8:
        return float("nan"), 0.0
    t = np.arange(channel.size, dtype=float) / fs
    reference_wave = _fit_tone(channel, t, fundamental)
    reference = float(np.sqrt(2.0 * np.mean(reference_wave**2)))
    if reference <= 0.0:
        return float("nan"), 0.0

    residual = channel - reference_wave
    nyquist = 0.5 * fs
    for order in range(2, max_order + 1):
        centre = order * fundamental
        if centre >= nyquist:
            break
        residual = residual - _fit_tone(residual, t, centre)

    window = np.hanning(residual.size)
    spectrum = np.abs(np.fft.rfft(residual * window))
    frequencies = np.fft.rfftfreq(residual.size, 1.0 / fs)
    if spectrum.size < 4:
        return float("nan"), 0.0
    resolution = float(frequencies[1])
    # drop DC, which dc_offset reports on its own, and the bins the subtraction
    # itself cannot clean because a fit leaves a residual at its own frequency
    mask = frequencies > max(guard * resolution, 0.5 * resolution)
    if not mask.any():
        return float("nan"), 0.0
    index = int(np.argmax(np.where(mask, spectrum, -np.inf)))
    # A discrete tone has to stand out from whatever else is in the residual, or
    # it is not a tone. Broadband noise also has a largest bin, and without this
    # test a record carrying nothing but noise reports its noisiest bin as an
    # interharmonic. A tone rises an order of magnitude above the surrounding
    # floor, while the extreme of a noise spectrum rises only a few times above
    # its own median.
    floor = float(np.median(spectrum[mask]))
    if floor > 0.0 and spectrum[index] < _TONE_PROMINENCE * floor:
        return float("nan"), 0.0

    # The peak bin is not the tone. A tone between two bins loses up to about a
    # sixth of its height to the shape of the window, so its frequency is refined
    # by the three-point interpolation of the interpolated-DFT literature and its
    # amplitude is then fitted in the time domain at that frequency, where the
    # bin alignment no longer matters at all.
    refined = frequencies[index] + resolution * _interpolate_peak(spectrum, index)

    # Fit the fundamental, its harmonics and the candidate *together* rather than
    # one after another. Sequential subtraction assumes the components are
    # orthogonal over the window, which holds only when the window spans a whole
    # number of beat periods between them. For a tone a few hertz from the
    # fundamental it generally does not, and the fundamental's fit then absorbs a
    # part of the tone -- about a tenth of it at three hertz -- which is the very
    # case this measurement exists to catch. A joint fit needs no orthogonality.
    present = [fundamental]
    for order in range(2, max_order + 1):
        centre = order * fundamental
        if centre >= nyquist:
            break
        present.append(centre)
    if min(abs(refined - f) for f in present) < 0.5 * resolution:
        return float("nan"), 0.0  # the candidate is the fundamental or a harmonic
    amplitudes = _joint_fit(channel, t, present + [refined])
    return float(refined), float(amplitudes[-1] / reference)


def _joint_fit(channel: np.ndarray, t: np.ndarray, frequencies) -> np.ndarray:
    """Peak amplitude of each of several tones, fitted simultaneously.

    Peak rather than RMS, to match the scaling of the reference this module
    divides by, which comes from :func:`_fit_tone`.
    """
    columns = []
    for frequency in frequencies:
        columns.append(np.cos(2.0 * np.pi * frequency * t))
        columns.append(np.sin(2.0 * np.pi * frequency * t))
    design = np.column_stack(columns)
    coefficients, *_ = np.linalg.lstsq(design, channel, rcond=None)
    pairs = coefficients.reshape(-1, 2)
    return np.hypot(pairs[:, 0], pairs[:, 1])


def _interpolate_peak(spectrum: np.ndarray, index: int) -> float:
    """Fractional-bin offset of a peak, from the three bins around it.

    The estimator of a Hann-windowed interpolated DFT,
    ``d = 2 (m+ - m-) / (m- + 2 m0 + m+)``, clipped to half a bin.
    """
    if index <= 0 or index >= spectrum.size - 1:
        return 0.0
    lower, centre, upper = spectrum[index - 1], spectrum[index], spectrum[index + 1]
    denominator = lower + 2.0 * centre + upper
    if denominator <= 0.0:
        return 0.0
    return float(np.clip(2.0 * (upper - lower) / denominator, -0.5, 0.5))


def _nearest(frequencies: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(frequencies - value)))


def _event_times(magnitude, phase_deg, times, span) -> list:
    """Instants where the amplitude or the phase track moves abruptly.

    Used only to decide which part of the record is quiet enough for the steady
    characteristics to be measured on, so it is deliberately coarser than the
    measurements themselves. The thresholds are set above what a steady
    modulation reaches over one period -- a depth of 0.1 at 2 Hz moves the
    amplitude by about 2.5 percent per period -- because a modulation is a steady
    characteristic and excluding it from its own measurement window would leave
    nothing to measure. A small step missed here is still reported by
    ``magnitude_step`` or ``phase_step``, which have their own thresholds. All
    that is lost is a little of the record.
    """
    events: list[float] = []
    reference = float(np.median(magnitude)) or 1.0
    for track, threshold in ((magnitude / reference, 0.05), (phase_deg, 5.0)):
        if track.size <= span:
            continue
        difference = np.abs(track[span:] - track[:-span])
        moving = difference > threshold
        # keep the start of each contiguous run, which is where the event is
        starts = np.flatnonzero(moving & ~np.r_[False, moving[:-1]])
        events.extend(float(times[index]) for index in starts)
    return sorted(events)


def _spectral_blocks(frame: pd.DataFrame, events, samples: int, exclusion_s: float) -> list:
    """Blocks of whole fundamental periods on which a spectrum may be taken.

    The record is cut into blocks of :data:`_BLOCK_PERIODS` periods and the blocks
    that straddle an event are dropped. Each surviving block is measured on its
    own and the caller keeps the worst result.

    Blocks rather than one long stretch, for two reasons. A disturbance switched on
    for part of a record is diluted in a spectrum of the whole thing, so measuring
    the whole record would understate it, while measuring only the longest quiet
    stretch can land on the part where it is absent and miss it entirely. And a
    block of a whole number of periods puts the fundamental exactly on a bin,
    which is what keeps its leakage from being mistaken for an interharmonic.
    """
    times = frame.index.to_numpy(dtype=float)
    size = _BLOCK_PERIODS * samples
    if times.size < 2 * samples:
        return [frame]
    size = min(size, (times.size // 2) * 1) if size > times.size else size
    blocks = []
    for start in range(0, times.size - size + 1, size):
        stop = start + size
        block = frame.iloc[start:stop]
        span = (float(times[start]), float(times[stop - 1]))
        if any(span[0] - exclusion_s <= event <= span[1] + exclusion_s for event in events):
            continue
        blocks.append(block)
    if not blocks:
        # every block straddles an event, so measure them all and let the caller's
        # worst-of aggregation stand: an inflated number is better than none
        blocks = [
            frame.iloc[start : start + size]
            for start in range(0, max(1, times.size - size + 1), size)
        ]
    return blocks or [frame]


def _steady_intervals(times, events, span_s, edge_s, minimum_s) -> list:
    """Every stretch of the record with no event in it, longest first.

    The spectral characteristics are measured on these rather than over the whole
    record. A step or a ramp spreads energy across the spectrum, and measured
    over a record that contains one, a clean signal looks as though it carried an
    interharmonic and a noise floor. Each stretch is taken on its own rather than
    stitched to the others, because a seam between two stretches is a
    discontinuity and would recreate the artefact being avoided.

    Every stretch is measured and the worst result is kept, not just the longest
    one. A disturbance that is switched on for part of a record leaves the rest
    of it clean, and a measurement confined to the longest quiet stretch can land
    on the clean part and report that the record contains nothing.
    """
    start, stop = float(times[0]) + edge_s, float(times[-1])
    boundaries = [start]
    for event in events:
        boundaries.extend([event - span_s, event + span_s])
    boundaries.append(stop)
    intervals = []
    for lo, hi in zip(boundaries[::2], boundaries[1::2]):
        lo, hi = max(lo, start), min(hi, stop)
        if hi - lo >= minimum_s:
            intervals.append((lo, hi))
    intervals.sort(key=lambda pair: pair[1] - pair[0], reverse=True)
    return intervals


def _modulation(track: np.ndarray, fs_track: float, *, reference: float) -> tuple[float, float]:
    """Depth and frequency of a periodic variation of a track.

    ``track`` is an amplitude or a phase trace and ``reference`` scales the depth
    (the mean amplitude, or one for a phase in radians). The depth is reported in
    the convention of equation 25 of IEC/IEEE 60255-118-1, where the modulated
    amplitude is ``A (1 + kx cos wm t)``, so the depth is the peak deviation of
    the fundamental of the track and not its peak-to-peak value.
    """
    centred = track - np.mean(track)
    if centred.size < 8 or reference <= 0.0:
        return 0.0, float("nan")
    window = np.hanning(centred.size)
    # A cosine of amplitude A through a Hann window of N samples has a peak of
    # A*N/4 in the one-sided transform: N/2 for the one-sided scaling and 1/2
    # again for the window's coherent gain. Dividing by N/2 alone would report
    # half the true depth.
    spectrum = np.abs(np.fft.rfft(centred * window)) / (0.25 * centred.size)
    frequencies = np.fft.rfftfreq(centred.size, 1.0 / fs_track)
    if spectrum.size < 2:
        return 0.0, float("nan")
    index = 1 + int(np.argmax(spectrum[1:]))  # skip the residual mean
    return float(spectrum[index] / reference), float(frequencies[index])


def characterise(
    abc,
    fs: float | None = None,
    *,
    nominal: float = 50.0,
    thresholds: dict | None = None,
    reference_amplitude: float | None = None,
    harmonic_orders: int = 40,
    interharmonic_guard: float = 2.0,
) -> dict:
    """Measure which characteristics a three-phase record contains.

    Parameters
    ----------
    abc : mapping or pandas.DataFrame
        The three phase voltages or currents, either as the
        ``{"a", "b", "c"}`` mapping that
        :func:`trimes.synchrophasor.three_phase` returns, or as a frame with
        those three columns (or exactly three columns in that order).
    fs : float, optional
        Sampling rate in Hz. Taken from the index when omitted, which then has
        to be uniform seconds.
    nominal : float, default 50.0
        Nominal system frequency in Hz.
    thresholds : dict, optional
        Overrides for :data:`DEFAULT_THRESHOLDS`, merged on top of it.
    reference_amplitude : float, optional
        The operating point that a dip or a swell is measured against, as a
        positive-sequence RMS value. Supply it whenever it is known, because it
        cannot always be inferred: a record that is dipped for half its length
        has no majority level, and the level before the first event is the best
        guess available but is only a guess. With ``None``, the level before the
        first detected event is used, falling back to the median of the record.
    harmonic_orders : int, default 40
        Highest harmonic order considered, both for the distortion and as a
        guard band for the interharmonic search. The default is the range
        IEC 61000-4-7 defines.
    interharmonic_guard : float, default 2.0
        Half-width of the guard band around each harmonic, in spectral bins.

    Returns
    -------
    dict
        With the keys

        ``nominal``, ``fs`` : float
            The nominal frequency and the sampling rate in Hz.
        ``duration`` : float
            Length of the record in seconds.
        ``fundamental_frequency`` : float
            Median of the measured fundamental frequency in Hz.
        ``fundamental_amplitude`` : float
            The operating point a dip or a swell is measured against:
            ``reference_amplitude`` if given, otherwise the median amplitude
            before the first event, otherwise the median over the record.
        ``events`` : list of float
            Onsets in seconds of the abrupt changes of the positive-sequence
            amplitude (more than 5 percent of its median within one period) or
            phase (more than 5 degrees within one period), in ascending order.
            Each is the start of the period over which the change was seen, so
            it lies up to one period before the change itself. These are the
            event instants, and not the ``interval`` of the findings, which for
            ``magnitude_step`` and ``phase_step`` is the instant of the largest
            step and can lie several periods after its onset.
        ``periodic_disturbance`` : bool
            ``True`` if there are more than :data:`_MAX_EVENTS` events. They are
            then taken for the beats of a steady disturbance, such as an
            interharmonic, rather than discrete events, and are not excluded
            from the steady measurements.
        ``steady_interval`` : (float, float)
            Start and end in seconds of the longest stretch without an event,
            on which the modulations are measured.
        ``blocks`` : int
            Number of spectral blocks the harmonics, interharmonics, noise and
            DC offset are measured on.
        ``stationary`` : bool
            ``False`` if a frequency ramp was found. The spectral findings are
            then not assessable, see ``assessable`` below.
        ``findings`` : dict
            One entry per name in :data:`CHARACTERISTICS`, each a dict with
            ``value``, ``unit``, ``threshold``, ``present`` (the verdict of
            ``value`` against ``threshold``), ``assessable`` and ``interval``.
            ``assessable`` is ``False`` where the verdict could not be decided,
            which a caller must read as *possibly present*, and the entry then
            carries ``limited_by``, the characteristics that prevented it.
            ``interval`` is ``None`` for a characteristic without a location
            and otherwise a ``(start, end)`` pair in seconds: the instant of
            the largest step for the steps, of the most extreme settled
            amplitude for ``dip`` and ``swell``, of the largest deviation for
            ``off_nominal_frequency``, the baseline of the steepest slope for
            ``frequency_ramp``, and the spectral block of the largest value for
            ``harmonics``, ``interharmonic``, ``noise`` and ``dc_offset``.
            Some entries carry a ``detail`` dict, for example the order of the
            dominant harmonic or the frequency of the modulation.

    Raises
    ------
    ValueError
        If the record is empty, is not uniformly sampled, or contains ``nan``.

    See Also
    --------
    characteristics_table : the same result as a table.
    trimes.synchrophasor.three_phase : the forward model of most of these.

    Examples
    --------
    >>> from trimes.synchrophasor import steady_state, three_phase
    >>> abc, _ = three_phase(steady_state(5000.0, duration=1.0), harmonics={5: 0.04})
    >>> result = characterise(abc, 5000.0)
    >>> result["findings"]["harmonics"]["present"]
    True
    >>> result["findings"]["dip"]["present"]
    False
    """
    limits = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    frame, fs = _prepare(abc, fs)
    times = frame.index.to_numpy(dtype=float)
    duration = float(times[-1] - times[0]) if times.size > 1 else 0.0

    findings: dict[str, dict] = {}

    def record(name, value, *, interval=None, detail=None):
        threshold = limits[name]
        below = name in _BELOW
        present = bool(value < threshold) if below else bool(value > threshold)
        findings[name] = {
            "value": float(value),
            "unit": _UNITS.get(name, "fraction"),
            "threshold": float(threshold),
            "present": present,
            "assessable": True,
            "interval": interval,
        }
        if detail:
            findings[name]["detail"] = detail

    # -- the fundamental, which everything below is measured relative to ----- #
    frequency = _phasor.frequency_trace(
        frame, nominal=nominal, window_cycles=3.0, step_cycles=1.0, rocof=True
    )
    per_channel = frequency.xs("frequency", axis=1, level=-1)
    f_track = per_channel.mean(axis=1).to_numpy(dtype=float)
    f_times = per_channel.index.to_numpy(dtype=float)
    finite = np.isfinite(f_track)
    fundamental = float(np.median(f_track[finite])) if finite.any() else nominal

    period = 1.0 / fundamental
    samples_per_period = max(2, int(round(fs / fundamental)))
    sequence = _sequence_traces(frame, fs, fundamental)
    positive = sequence["positive"].to_numpy()
    magnitude = np.abs(positive)
    track_times = sequence.index.to_numpy(dtype=float)
    # The phase of a rolling Fourier coefficient advances at the *full*
    # fundamental frequency, because each window is referenced to its own start.
    # De-rotating at the measured fundamental rather than at the nominal one
    # leaves a constant, so neither the nominal rotation nor a steady off-nominal
    # offset can masquerade as a phase step or as a phase modulation.
    de_rotated = np.unwrap(np.angle(positive)) - 2.0 * np.pi * fundamental * track_times
    phase_deg = np.degrees(de_rotated)

    # -- events, and the blocks the steady characteristics are measured on ---- #
    events = _event_times(magnitude, phase_deg, track_times, samples_per_period)
    # A steady interharmonic beats against the fundamental and makes a one-period
    # phasor oscillate continuously, which the level-change detector above reports
    # as an event per beat. Excluding all of those would remove the whole disturbed
    # region and measure the disturbance only where it is absent, so beyond a small
    # count the "events" are recognised as one periodic condition instead.
    periodic = len(events) > _MAX_EVENTS
    for_exclusion = [] if periodic else events
    # Two window sets, because the characteristics need different things of them.
    # A spectrum wants a whole number of periods and is better taken in several
    # short blocks, so that a disturbance present for part of the record is not
    # diluted. The track characteristics -- a dip, an unbalance, a modulation --
    # want one long contiguous stretch instead: a modulation at 2 Hz has a period
    # of half a second and cannot be seen at all in a block of ten cycles.
    windows = _spectral_blocks(
        frame, for_exclusion, samples_per_period, _EXCLUSION_PERIODS * period
    )
    intervals = _steady_intervals(
        track_times,
        for_exclusion,
        span_s=_EXCLUSION_PERIODS * period,
        edge_s=period,
        minimum_s=4.0 * period,
    ) or [(float(times[0]), float(times[-1]))]
    quiet_track = np.zeros(track_times.shape, dtype=bool)
    for lo, hi in intervals:
        quiet_track |= (track_times >= lo) & (track_times <= hi)
    # the longest stretch, which is where a modulation is looked for and which is
    # what "steady_interval" reports: a contiguous span with no event in it
    longest = max(intervals, key=lambda pair: pair[1] - pair[0])
    on_track = (track_times >= longest[0]) & (track_times <= longest[1])
    steady_from, steady_to = longest

    # -- frequency ramp ------------------------------------------------------ #
    # measured as a change over a baseline rather than as a point-to-point
    # gradient, because the jitter of a frequency track on a noisy signal
    # differentiates into a slope far larger than any real ramp
    # a phase or magnitude step makes the frequency estimator spike for about one
    # window, which over a 0.2 s baseline still reads as a steep slope, so the
    # neighbourhood of every detected event is taken out first. A genuine ramp
    # produces no event: at 1 Hz/s the phase moves by a negligible amount per
    # period, so the ramp itself survives this.
    away = _away_from_events(f_times, events, _EXCLUSION_PERIODS * period) & finite
    ramp_value, ramp_where = _ramp(f_track[away], f_times[away], _RAMP_BASELINE_S)
    record("frequency_ramp", ramp_value, **ramp_where)

    # The *largest* departure from nominal, not the departure of the median. A
    # record that holds 50 Hz, then 52, then 48 has a median at nominal and is
    # nowhere near nominal for most of its length.
    if away.any():
        deviation = np.abs(f_track[away] - nominal)
        worst = int(np.argmax(deviation))
        record(
            "off_nominal_frequency",
            float(deviation[worst]),
            interval=(float(f_times[away][worst]), float(f_times[away][worst])),
            detail={"median": fundamental, "range": [float(f_track[away].min()), float(f_track[away].max())]},
        )
    else:
        record("off_nominal_frequency", abs(fundamental - nominal))

    # -- steps --------------------------------------------------------------- #
    step, when = _largest_jump(phase_deg, track_times, samples_per_period, detrend=True)
    record("phase_step", step, interval=None if when is None else (when, when))

    reference = (
        float(reference_amplitude)
        if reference_amplitude
        else _reference_amplitude(magnitude, track_times, events, period)
    )
    relative = magnitude / reference if reference > 0.0 else magnitude
    jump, when = _largest_jump(
        relative, track_times, samples_per_period, lag=3 * samples_per_period
    )
    record("magnitude_step", jump, interval=None if when is None else (when, when))

    # -- dip, swell, interruption -------------------------------------------- #
    # Measured against the pre-event level, not the median: a record that spends
    # half its length dipped has a median part way into the dip. Measured only
    # where the record is settled, because a one-period phasor dips briefly at any
    # abrupt event, and a phase step -- which does not change the amplitude at all
    # -- would otherwise be reported as a shallow dip.
    if magnitude.size and reference > 0.0 and quiet_track.any():
        settled = np.flatnonzero(quiet_track)
        lowest = int(settled[np.argmin(magnitude[quiet_track])])
        highest = int(settled[np.argmax(magnitude[quiet_track])])
        low = float(magnitude[lowest] / reference)
        high = float(magnitude[highest] / reference)
        record("dip", low, interval=(float(track_times[lowest]), float(track_times[lowest])))
        record("swell", high, interval=(float(track_times[highest]), float(track_times[highest])))
        record("interruption", low)
    else:
        record("dip", 1.0)
        record("swell", 1.0)
        record("interruption", 1.0)

    # -- unbalance, over every quiet stretch --------------------------------- #
    with np.errstate(divide="ignore", invalid="ignore"):
        negative_ratio = np.abs(sequence["negative"].to_numpy()) / magnitude
        zero_ratio = np.abs(sequence["zero"].to_numpy()) / magnitude
    record("negative_sequence", _safe_max(negative_ratio[quiet_track]))
    record("zero_sequence", _safe_max(zero_ratio[quiet_track]))

    # -- distortion, the worst over every quiet stretch ---------------------- #
    # The worst rather than the longest. A disturbance switched on for part of a
    # record leaves the rest clean, and a measurement confined to one stretch can
    # land on the clean part and report that the record contains nothing.
    orders = tuple(range(2, harmonic_orders + 1))
    best_harmonic = (-1.0, None, None)
    best_tone = (-1.0, float("nan"), None)
    best_noise = (-1.0, None)
    for window in windows:
        # Each stretch is measured at *its own* fundamental. One frequency for the
        # whole record is not good enough: a record that holds 50 Hz, then 52, then
        # 48 has a median of 50, and a spectrum taken at 50 Hz of a stretch that is
        # actually at 52 leaks so badly that it reports an interharmonic of several
        # thousand percent. The stretches contain no event by construction, so the
        # frequency is nearly constant within each one.
        local = _local_fundamental(window, nominal, fundamental)
        samples_local = max(2, int(round(fs / local)))
        window = _whole_periods(window, window.index[0], window.index[-1], samples_local)
        if len(window) < 2 * samples_local:
            continue
        channel = window["a"]
        span = (float(window.index[0]), float(window.index[-1]))
        spectrum = _fft.harmonics(channel, fundamental=local, orders=(1,) + orders, window="hann")
        amplitudes = spectrum.iloc[:, 0]
        fundamental_level = float(amplitudes.get(1, np.nan))
        distortion = float(_fft.thd(channel, fundamental=local, max_order=harmonic_orders).iloc[0])
        dominant = amplitudes.drop(index=1, errors="ignore").dropna()
        detail = None
        if not dominant.empty and np.isfinite(fundamental_level) and fundamental_level > 0.0:
            detail = {
                "order": int(dominant.idxmax()),
                "level": float(dominant.max() / fundamental_level),
            }
        if distortion > best_harmonic[0]:
            best_harmonic = (distortion, detail, span)

        values = channel.to_numpy(dtype=float)
        tone_hz, tone_level = _dominant_tone(
            values, fs, local, guard=interharmonic_guard, max_order=harmonic_orders
        )
        if tone_level > best_tone[0]:
            best_tone = (tone_level, tone_hz, span)

        level = _noise_fraction(channel, fs, local, harmonic_orders, tone_hz)
        if level > best_noise[0]:
            best_noise = (level, span)

    record("harmonics", best_harmonic[0], interval=best_harmonic[2], detail=best_harmonic[1])
    tone_level, tone_hz, tone_span = best_tone
    record(
        "interharmonic",
        tone_level,
        interval=tone_span,
        detail=None
        if tone_level <= limits["interharmonic"] or not np.isfinite(tone_hz)
        else {"frequency": tone_hz},
    )
    record("noise", best_noise[0], interval=best_noise[1])

    # -- DC offset, the worst over every quiet stretch ----------------------- #
    # a common-mode offset is what a shared measurement chain produces, and it is
    # the only kind that a three-phase sum does not cancel
    best_offset = (-1.0, None, None)
    for window in windows:
        values = window.to_numpy(dtype=float)
        means = values.mean(axis=0)
        peak = float(np.max(np.abs(values))) or 1.0
        level = float(np.abs(np.mean(means))) / peak
        if level > best_offset[0]:
            best_offset = (
                level,
                {"per_phase": [float(m) for m in means]},
                (float(window.index[0]), float(window.index[-1])),
            )
    record("dc_offset", best_offset[0], interval=best_offset[2], detail=best_offset[1])

    # -- modulation, on the quiet stretch ------------------------------------ #
    am_depth, am_hz = _modulation(magnitude[on_track], fs, reference=reference)
    record("am_modulation", am_depth, detail=_modulation_detail(am_depth, am_hz, limits))
    # the phase modulation of equation 25 is a deviation in radians, so its
    # reference is one rather than the operating point
    pm_depth, pm_hz = _modulation(de_rotated[on_track], fs, reference=1.0)
    record("pm_modulation", pm_depth, detail=_modulation_detail(pm_depth, pm_hz, limits, "pm"))

    # -- what could not be assessed ------------------------------------------ #
    # The spectral characteristics are measured by fitting and subtracting one
    # fundamental at one frequency. That is only meaningful while the fundamental
    # holds still. Where it ramps or is modulated, the residual of the fit is the
    # movement of the fundamental itself, and reporting it as an interharmonic or
    # as noise would be an artefact of the method rather than a property of the
    # signal. Such a finding is marked as not assessable, and a caller deciding
    # whether something is safe must read that as *possibly present*.
    # Only a moving *frequency* blocks the fit, and only that is tested for. An
    # apparent amplitude or phase modulation must not be used here even though it
    # often accompanies a moving fundamental, because an interharmonic produces
    # one as a side effect of beating against the fundamental: suppressing the
    # spectral findings on that evidence would let a tone suppress its own
    # detection. A genuine modulation moves the frequency as well and is still
    # caught, through the ramp.
    moving = [name for name in ("frequency_ramp",) if findings[name]["present"]]
    if moving:
        for name in ("harmonics", "interharmonic", "noise"):
            findings[name]["assessable"] = False
            findings[name]["limited_by"] = moving

    return {
        "nominal": float(nominal),
        "fs": float(fs),
        "duration": duration,
        "fundamental_frequency": fundamental,
        "fundamental_amplitude": reference,
        "steady_interval": (float(steady_from), float(steady_to)),
        "events": events,
        "periodic_disturbance": bool(periodic),
        "blocks": len(windows),
        "stationary": not moving,
        "findings": {name: findings[name] for name in CHARACTERISTICS},
    }


def _modulation_detail(depth: float, hertz: float, limits: dict, kind: str = "am") -> dict | None:
    """The modulation frequency, but only where there is a modulation to have one.

    The peak of a spectrum exists whatever the signal, so a frequency reported
    beside a depth of order 1e-17 would be noise dressed as a measurement.
    """
    threshold = limits[f"{kind}_modulation"]
    if depth <= threshold or not np.isfinite(hertz):
        return None
    return {"frequency": float(hertz)}


def _safe_max(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(finite.max()) if finite.size else 0.0


def _noise_fraction(
    channel: pd.Series, fs: float, fundamental: float, max_order: int, tone_hz: float
) -> float:
    """Broadband content as a fraction of the fundamental.

    The fundamental, its harmonics and the dominant interharmonic are masked out
    of the power spectral density and what remains is integrated, so a harmonic
    or an interharmonic is not counted twice as noise.
    """
    density = _fft.power_spectral_density(channel, window="hann", detrend="constant")
    frequencies = density.index.to_numpy(dtype=float)
    power = density.iloc[:, 0].to_numpy(dtype=float)
    if frequencies.size < 4:
        return 0.0
    resolution = float(frequencies[1] - frequencies[0])
    mask = np.ones(power.shape, dtype=bool)
    band = 3.0 * resolution
    for order in range(0, max_order + 1):
        centre = order * fundamental
        if centre > frequencies[-1]:
            break
        mask &= np.abs(frequencies - centre) > band
    if np.isfinite(tone_hz):
        mask &= np.abs(frequencies - tone_hz) > band
    broadband = float(np.trapezoid(np.where(mask, power, 0.0), frequencies))
    total = float(np.trapezoid(power, frequencies))
    if total <= 0.0:
        return 0.0
    # both are variances, so the amplitude ratio is the square root
    return float(np.sqrt(max(broadband, 0.0) / total))


def characteristics_table(result: dict) -> pd.DataFrame:
    """Lay a :func:`characterise` result out as one row per characteristic.

    Parameters
    ----------
    result : dict
        The output of :func:`characterise`.

    Returns
    -------
    pandas.DataFrame
        Indexed by characteristic, with columns ``present``, ``value``,
        ``unit``, ``threshold``, ``interval`` and ``detail``. The row order is
        :data:`CHARACTERISTICS`.

    See Also
    --------
    trimes.synchrophasor.metrics.compliance : the same shape for a pass or fail
        against the limits of a standard.
    """
    rows = []
    for name, finding in result["findings"].items():
        interval = finding.get("interval")
        rows.append(
            {
                "characteristic": name,
                "present": finding["present"],
                "assessable": finding["assessable"],
                "value": finding["value"],
                "unit": finding["unit"],
                "threshold": finding["threshold"],
                "at": None if interval is None else float(interval[0]),
                "detail": finding.get("detail"),
            }
        )
    return pd.DataFrame(rows).set_index("characteristic")
