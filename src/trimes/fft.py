"""Frequency-domain analysis for uniformly sampled time series.

All public functions accept a :class:`pandas.Series` or
:class:`pandas.DataFrame` whose index represents time and return objects
indexed by frequency in Hz (or by harmonic order, in the case of
:func:`harmonics`).

Conventions
-----------
Time index
    May be float seconds, a :class:`pandas.TimedeltaIndex` or a
    :class:`pandas.DatetimeIndex`. It is converted to float seconds relative
    to the first sample by :func:`time_to_seconds`.
Uniform sampling
    Every transform assumes equidistant samples. This is verified and a
    ``ValueError`` is raised otherwise; use :func:`resample_uniform` to fix
    non-uniform data explicitly rather than relying on silent interpolation.
Frequency resolution
    ``df = fs / n``, where ``n`` is the number of samples and ``fs`` the
    sampling rate. The highest represented frequency is the Nyquist frequency
    ``fs / 2``.
Column handling
    Columns are treated as independent channels and their names are preserved.
    A ``Series`` input is processed as a single channel; the returned object is
    still a ``DataFrame`` with one column (except for :func:`thd`).
Missing data
    ``NaN`` values are rejected rather than zero-filled, because zero-filling
    corrupts spectra silently.
"""

from __future__ import annotations

from typing import Literal, Sequence

import numpy as np
import pandas as pd

try:
    from scipy.signal import get_window, resample
except ModuleNotFoundError as _exc:  # pragma: no cover - depends on env
    raise ModuleNotFoundError(
        "trimes.fft needs scipy; install it with: pip install 'trimes[dsp]'"
    ) from _exc

from trimes.base import time_to_seconds, get_sampling_rate

__all__ = [
    "rfft",
    "amplitude_spectrum",
    "power_spectral_density",
    "phase_spectrum",
    "at_frequency",
    "harmonics",
    "thd",
]

FrameOrSeries = pd.Series | pd.DataFrame


# --------------------------------------------------------------------------- #
# Core transform
# --------------------------------------------------------------------------- #
def _prepare(
    data: FrameOrSeries,
    window: str | tuple | None,
    detrend: Literal["constant", "linear"] | None,
) -> tuple[np.ndarray, np.ndarray, float, pd.Index, bool]:
    """Validate the input and apply detrending and windowing.

    Internal helper shared by all transforms, so that they behave identically
    with respect to validation, detrending and window normalisation.

    Parameters
    ----------
    data : pandas.Series or pandas.DataFrame
        Uniformly sampled input time series.
    window : str or tuple or None
        Window specification passed to :func:`scipy.signal.get_window`
        (e.g. ``"hann"``, ``"blackman"``, ``("kaiser", 8.6)``). ``None``
        selects a rectangular window (no tapering).
    detrend : {'constant', 'linear'} or None
        Trend removal applied per column *before* windowing:

        * ``"constant"`` — subtract the column mean (removes the DC
          component).
        * ``"linear"`` — subtract a least-squares straight line.
        * ``None`` — no detrending.

    Returns
    -------
    xw : numpy.ndarray
        Detrended and windowed samples, shape ``(n, n_columns)``, ``float``.
    w : numpy.ndarray
        The window applied, shape ``(n,)``. Needed by the callers for
        amplitude and power normalisation.
    fs : float
        Sampling rate in Hz.
    columns : pandas.Index
        Column labels of the input, for reconstructing the output.
    was_series : bool
        ``True`` if the input was a ``Series``.

    Raises
    ------
    ValueError
        If the input is not uniformly sampled, contains ``NaN``, or if
        ``detrend`` is an unknown option.

    Notes
    -----
    Detrending is deliberately applied before windowing: doing it afterwards
    would reintroduce a spectral bias, because the window itself has non-zero
    mean.
    """
    was_series = isinstance(data, pd.Series)
    frame = data.to_frame() if was_series else data

    fs = get_sampling_rate(frame)
    x = frame.to_numpy(dtype=float)

    if np.isnan(x).any():
        raise ValueError("Input contains NaN. Interpolate or drop them first.")

    if detrend == "constant":
        x = x - x.mean(axis=0)
    elif detrend == "linear":
        t = time_to_seconds(frame.index)
        coeffs = np.polynomial.polynomial.polyfit(t, x, deg=1)
        x = x - np.polynomial.polynomial.polyval(t, coeffs).T
    elif detrend is not None:
        raise ValueError(f"Unknown detrend option {detrend!r}.")

    n = x.shape[0]
    w = np.ones(n) if window is None else get_window(window, n, fftbins=True)

    return x * w[:, None], w, fs, frame.columns, was_series


def rfft(
    data: FrameOrSeries,
    window: str | tuple | None = None,
    detrend: Literal["constant", "linear"] | None = None,
) -> pd.DataFrame:
    """Compute the raw one-sided complex FFT of a real-valued time series.

    Thin pandas wrapper around :func:`numpy.fft.rfft`. No amplitude scaling is
    applied, which makes this the right entry point if you need the
    unnormalised coefficients (e.g. for filtering and a subsequent inverse
    transform). For physically meaningful magnitudes use
    :func:`amplitude_spectrum`.

    Parameters
    ----------
    data : pandas.Series or pandas.DataFrame
        Uniformly sampled, real-valued time series without ``NaN``.
    window : str or tuple or None, default None
        Window specification for :func:`scipy.signal.get_window`. ``None``
        means a rectangular window.
    detrend : {'constant', 'linear'} or None, default None
        Trend removal applied before windowing. See :func:`_prepare`.

    Returns
    -------
    pandas.DataFrame
        Complex FFT coefficients with ``n // 2 + 1`` rows. The index is named
        ``"frequency"`` and holds the bin centres in Hz, from ``0`` (DC) to the
        Nyquist frequency ``fs / 2``. Columns are those of the input.

    Raises
    ------
    ValueError
        Propagated from :func:`_prepare` for non-uniform sampling, ``NaN``
        values or an unknown ``detrend`` option.

    See Also
    --------
    amplitude_spectrum : Scaled single-sided spectrum in signal units.
    power_spectral_density : Power density estimate in units²/Hz.
    phase_spectrum : Phase angles of these coefficients.

    Notes
    -----
    Because the input is real, negative frequencies carry no additional
    information and are omitted. Consequently the coefficient magnitudes are
    *not* the physical amplitudes: for a bin-centred sine of amplitude ``A``,
    ``abs(coefficient) == A * n / 2``.
    """
    xw, _, fs, columns, _ = _prepare(data, window, detrend)
    spectrum = np.fft.rfft(xw, axis=0)
    freq = np.fft.rfftfreq(xw.shape[0], d=1.0 / fs)
    return pd.DataFrame(spectrum, index=pd.Index(freq, name="frequency"), columns=columns)


def amplitude_spectrum(
    data: FrameOrSeries,
    window: str | tuple | None = None,
    detrend: Literal["constant", "linear"] | None = None,
    scaling: Literal["amplitude", "rms"] = "amplitude",
    complex_output: bool = False,
) -> pd.DataFrame:
    """Compute the single-sided amplitude spectrum in units of the input signal.

    A sinusoid of amplitude ``A`` whose frequency coincides with a bin centre
    yields exactly ``A`` (``scaling="amplitude"``) or ``A / sqrt(2)``
    (``scaling="rms"``), independently of the record length, sampling rate and
    window used.

    Parameters
    ----------
    data : pandas.Series or pandas.DataFrame
        Uniformly sampled, real-valued time series without ``NaN``.
    window : str or tuple or None, default None
        Window specification for :func:`scipy.signal.get_window`. ``None``
        (rectangular) is optimal when the record covers an integer number of
        signal periods; otherwise ``"hann"`` reduces spectral leakage at the
        cost of a wider main lobe. The window's coherent gain is compensated
        automatically.
    detrend : {'constant', 'linear'} or None, default None
        Trend removal applied before windowing. Use ``"constant"`` to suppress
        a large DC offset that would otherwise leak into low-frequency bins.
    scaling : {'amplitude', 'rms'}, default 'amplitude'
        Output convention:

        * ``"amplitude"`` — peak amplitude of the corresponding sinusoid.
        * ``"rms"`` — root-mean-square value, i.e. amplitude divided by
          ``sqrt(2)``. This convention makes the spectrum satisfy Parseval's
          theorem: the root sum of squares over all bins equals the RMS value
          of the (detrended, unwindowed) signal.
    complex_output : bool, default False
        If ``True``, return complex values whose magnitude is the scaled
        amplitude and whose argument is the phase in radians. Useful for
        harmonic phase analysis or for reconstructing individual components.
        If ``False``, return real magnitudes.

    Returns
    -------
    pandas.DataFrame
        Spectrum with ``n // 2 + 1`` rows, index named ``"frequency"`` in Hz,
        columns as in the input. Dtype is ``float`` unless ``complex_output``
        is ``True``.

    Raises
    ------
    ValueError
        If ``scaling`` is unknown, or propagated from :func:`_prepare` for
        non-uniform sampling, ``NaN`` values or an unknown ``detrend`` option.

    See Also
    --------
    rfft : Unnormalised complex coefficients.
    at_frequency : Robust lookup of specific frequencies in the result.
    harmonics : Amplitudes at integer multiples of a fundamental frequency.

    Notes
    -----
    Scaling is performed in three steps: division by the window sum (coherent
    gain), doubling of all bins except DC to account for the omitted negative
    frequencies, and — for even ``n`` — undoing that doubling for the Nyquist
    bin, which has no mirror partner.

    The frequency resolution is ``fs / n``. Components between bin centres are
    spread over neighbouring bins (spectral leakage) and their apparent
    amplitude is reduced by up to 36 % for a rectangular window; choose the
    record length to be an integer multiple of the signal period whenever
    possible.

    Examples
    --------
    >>> fs, t = 1000.0, np.arange(0, 1, 1 / 1000.0)
    >>> s = pd.Series(3.0 * np.sin(2 * np.pi * 50 * t), index=t)
    >>> float(at_frequency(amplitude_spectrum(s), 50.0).iloc[0, 0])
    3.0
    """
    xw, w, fs, columns, _ = _prepare(data, window, detrend)
    n = xw.shape[0]

    spectrum = np.fft.rfft(xw, axis=0) / w.sum()
    spectrum[1:] *= 2.0
    if n % 2 == 0:
        spectrum[-1] /= 2.0  # Nyquist bin is not mirrored

    if scaling == "rms":
        spectrum = spectrum / np.sqrt(2.0)
    elif scaling != "amplitude":
        raise ValueError(f"Unknown scaling {scaling!r}.")

    values = spectrum if complex_output else np.abs(spectrum)
    freq = np.fft.rfftfreq(n, d=1.0 / fs)
    return pd.DataFrame(values, index=pd.Index(freq, name="frequency"), columns=columns)


def power_spectral_density(
    data: FrameOrSeries,
    window: str | tuple | None = "hann",
    detrend: Literal["constant", "linear"] | None = "constant",
) -> pd.DataFrame:
    """Estimate the one-sided power spectral density (periodogram).

    Uses the same normalisation as :func:`scipy.signal.welch` with a single
    segment, so results are directly comparable. Appropriate for broadband or
    noisy signals, where the power *density* rather than a discrete amplitude
    is the quantity of interest.

    Parameters
    ----------
    data : pandas.Series or pandas.DataFrame
        Uniformly sampled, real-valued time series without ``NaN``.
    window : str or tuple or None, default 'hann'
        Window specification for :func:`scipy.signal.get_window`. Tapering is
        the default here because PSD estimation is normally applied to
        non-periodic records. The window's noise power gain is compensated.
    detrend : {'constant', 'linear'} or None, default 'constant'
        Trend removal applied before windowing. Mean removal is the default to
        prevent a DC offset from dominating the low-frequency estimate.

    Returns
    -------
    pandas.DataFrame
        PSD in units² per Hz (e.g. V²/Hz for a voltage signal), with
        ``n // 2 + 1`` rows, index named ``"frequency"`` in Hz, and the columns
        of the input. Values are real and non-negative.

    Raises
    ------
    ValueError
        Propagated from :func:`_prepare` for non-uniform sampling, ``NaN``
        values or an unknown ``detrend`` option.

    See Also
    --------
    amplitude_spectrum : Discrete amplitudes for deterministic components.
    scipy.signal.welch : Segment-averaged estimate with lower variance.

    Notes
    -----
    A single-segment periodogram has a high variance (roughly 100 % standard
    error per bin) that does *not* decrease with record length. For long
    records prefer Welch's method, which averages over overlapping segments
    and trades frequency resolution for a smoother estimate.

    Integrating the returned density over frequency recovers the variance of
    the detrended signal (Parseval's theorem), e.g. via
    ``numpy.trapezoid(psd.values, psd.index, axis=0)``.
    """
    xw, w, fs, columns, _ = _prepare(data, window, detrend)
    n = xw.shape[0]

    psd = np.abs(np.fft.rfft(xw, axis=0)) ** 2 / (fs * np.sum(w**2))
    psd[1:] *= 2.0
    if n % 2 == 0:
        psd[-1] /= 2.0

    freq = np.fft.rfftfreq(n, d=1.0 / fs)
    return pd.DataFrame(psd, index=pd.Index(freq, name="frequency"), columns=columns)


def phase_spectrum(
    data: FrameOrSeries,
    window: str | tuple | None = None,
    detrend: Literal["constant", "linear"] | None = None,
    deg: bool = False,
    unwrap: bool = False,
) -> pd.DataFrame:
    """Compute the phase angles of the one-sided spectrum.

    Parameters
    ----------
    data : pandas.Series or pandas.DataFrame
        Uniformly sampled, real-valued time series without ``NaN``.
    window : str or tuple or None, default None
        Window specification for :func:`scipy.signal.get_window`. Symmetric
        windows do not bias the phase, so ``None`` is usually fine.
    detrend : {'constant', 'linear'} or None, default None
        Trend removal applied before windowing.
    deg : bool, default False
        If ``True``, return degrees instead of radians.
    unwrap : bool, default False
        If ``True``, remove discontinuities of more than ``pi`` along the
        frequency axis using :func:`numpy.unwrap`, yielding a continuous phase
        curve. Applied before conversion to degrees.

    Returns
    -------
    pandas.DataFrame
        Phase angles with ``n // 2 + 1`` rows, index named ``"frequency"`` in
        Hz, columns as in the input. Values lie in ``(-pi, pi]`` (or
        ``(-180, 180]`` degrees) unless ``unwrap`` is ``True``.

    Raises
    ------
    ValueError
        Propagated from :func:`rfft`.

    Notes
    -----
    Phase is referenced to a cosine at the start of the record: a signal
    ``cos(2*pi*f*t)`` has phase ``0`` at ``f``, whereas ``sin(2*pi*f*t)`` has
    ``-pi/2``.

    Bins whose amplitude is near zero carry no meaningful phase information —
    their angle is determined by numerical noise. Mask them by amplitude
    (e.g. using :func:`amplitude_spectrum`) before interpreting or plotting
    the result.
    """
    spectrum = rfft(data, window=window, detrend=detrend)
    phase = np.angle(spectrum.to_numpy())
    if unwrap:
        phase = np.unwrap(phase, axis=0)
    if deg:
        phase = np.degrees(phase)
    return pd.DataFrame(phase, index=spectrum.index, columns=spectrum.columns)


# --------------------------------------------------------------------------- #
# Frequency lookup
# --------------------------------------------------------------------------- #
def at_frequency(
    spectrum: pd.DataFrame,
    frequencies: float | Sequence[float],
    tol: float | None = None,
) -> pd.DataFrame:
    """Extract spectrum values at the bins nearest to the requested frequencies.

    Parameters
    ----------
    spectrum : pandas.DataFrame
        Frequency-indexed result of :func:`amplitude_spectrum`, :func:`rfft`,
        :func:`power_spectral_density` or :func:`phase_spectrum`. The index
        must be numeric and equidistant.
    frequencies : float or sequence of float
        Requested frequency or frequencies in Hz. Need not coincide exactly
        with bin centres.
    tol : float, optional
        Maximum accepted deviation between a requested frequency and the
        nearest bin centre, in Hz. Defaults to half the bin width, i.e. the
        nearest bin is always accepted for frequencies inside the spectrum's
        range. Requested frequencies further away yield ``NaN`` rows.

    Returns
    -------
    pandas.DataFrame
        One row per requested frequency, indexed by the *requested* values
        (not the actual bin centres) under the index name ``"frequency"``, with
        the columns of ``spectrum``. Dtype is ``complex`` if ``spectrum`` is
        complex, otherwise ``float``.

    Raises
    ------
    ValueError
        If ``spectrum`` contains fewer than two bins, so that the bin width
        cannot be determined.

    Notes
    -----
    This helper exists because the frequency index originates from
    floating-point arithmetic, so ``spectrum.loc[50.0]`` may raise a
    ``KeyError`` even though a bin exists at nominally 50 Hz. For example,
    ``np.fft.rfftfreq(2000, d=1 / 10_000)`` places the tenth bin at
    ``49.99999999999999`` rather than ``50.0``.

    Only the single nearest bin is returned; no interpolation between bins is
    performed. If a component lies between bin centres, its energy is spread
    over several bins and the returned value underestimates the true
    amplitude.

    Examples
    --------
    >>> spec = amplitude_spectrum(df)
    >>> at_frequency(spec, [50, 250, 350]).round(3)
    """
    freqs = np.atleast_1d(np.asarray(frequencies, dtype=float))
    index = spectrum.index.to_numpy(dtype=float)
    if index.size < 2:
        raise ValueError("Spectrum must contain at least two bins.")

    bin_width = float(index[1] - index[0])
    tol = 0.5 * bin_width if tol is None else float(tol)

    idx = np.argmin(np.abs(index[None, :] - freqs[:, None]), axis=1)
    within_tol = np.abs(index[idx] - freqs) <= tol

    source = spectrum.to_numpy()
    dtype = complex if np.iscomplexobj(source) else float
    values = source[idx].astype(dtype)
    values[~within_tol] = np.nan

    return pd.DataFrame(
        values, index=pd.Index(freqs, name="frequency"), columns=spectrum.columns
    )


# --------------------------------------------------------------------------- #
# Harmonic analysis
# --------------------------------------------------------------------------- #
def harmonics(
    data: FrameOrSeries,
    fundamental: float = 50.0,
    orders: Sequence[int] | None = None,
    window: str | tuple | None = None,
    interpolate_peak: bool = False,
    complex_output: bool = False,
) -> pd.DataFrame:
    """Extract amplitudes at integer multiples of a fundamental frequency.

    Parameters
    ----------
    data : pandas.Series or pandas.DataFrame
        Uniformly sampled, real-valued time series without ``NaN``.
    fundamental : float, default 50.0
        Fundamental frequency in Hz (50 for European power systems, 60 for
        North American ones).
    orders : sequence of int, optional
        Harmonic orders to evaluate. Order ``1`` is the fundamental, ``0``
        would be DC. Defaults to ``1 .. floor(f_nyquist / fundamental)``,
        i.e. all harmonics representable at the given sampling rate.
    window : str or tuple or None, default None
        Window specification for :func:`scipy.signal.get_window`. Leave at
        ``None`` (rectangular) if the record covers an integer number of
        fundamental periods; otherwise combine ``"hann"`` with
        ``interpolate_peak=True``.
    interpolate_peak : bool, default False
        If ``True``, take the value with the largest magnitude among the three
        bins closest to each harmonic frequency instead of the single nearest
        bin. Mitigates the amplitude error caused by spectral leakage when the
        harmonic falls between bin centres. Despite the name, no interpolation
        between bins is performed.
    complex_output : bool, default False
        If ``True``, return complex values carrying both amplitude
        (magnitude) and phase (argument, in radians, referenced to a cosine at
        the start of the record). If ``False``, return real magnitudes.

    Returns
    -------
    pandas.DataFrame
        Amplitudes in units of the input signal (peak convention, see
        :func:`amplitude_spectrum`). The index is named ``"order"`` and holds
        the requested orders; columns are those of the input. Orders whose
        frequency exceeds the Nyquist frequency yield ``NaN`` rows. Dtype is
        ``float`` unless ``complex_output`` is ``True``.

    Raises
    ------
    ValueError
        Propagated from :func:`amplitude_spectrum`.

    See Also
    --------
    thd : Total harmonic distortion derived from these amplitudes.
    amplitude_spectrum : Full spectrum underlying this function.

    Notes
    -----
    For accurate results the record length should be an integer multiple of
    the fundamental period (coherent sampling); e.g. at ``fs = 10 kHz`` and
    ``fundamental = 50 Hz``, use multiples of 200 samples. The frequency
    resolution ``fs / n`` should also divide ``fundamental`` so that harmonics
    land exactly on bin centres.

    Off-nominal fundamental frequencies (e.g. 49.8 Hz during a frequency
    excursion) violate this assumption. In such cases estimate the actual
    fundamental first and pass it as ``fundamental``, or use a
    parametric method instead of the FFT.

    Examples
    --------
    >>> harmonics(df, fundamental=50.0, orders=[1, 5, 7]).round(3)
               u_a
    order
    1        100.0
    5         10.0
    7          5.0
    """
    spec = amplitude_spectrum(data, window=window, complex_output=complex_output)
    freq = spec.index.to_numpy()
    f_nyq = freq[-1]

    if orders is None:
        orders = range(1, int(f_nyq // fundamental) + 1)
    orders = [int(o) for o in orders]

    rows = []
    for order in orders:
        f_target = order * fundamental
        if f_target > f_nyq:
            rows.append(np.full(spec.shape[1], np.nan))
            continue
        idx = int(np.argmin(np.abs(freq - f_target)))
        if interpolate_peak:
            lo, hi = max(idx - 1, 0), min(idx + 2, len(freq))
            block = spec.iloc[lo:hi].to_numpy()
            sel = np.argmax(np.abs(block), axis=0)
            rows.append(block[sel, np.arange(block.shape[1])])
        else:
            rows.append(spec.iloc[idx].to_numpy())

    return pd.DataFrame(
        np.asarray(rows), index=pd.Index(orders, name="order"), columns=spec.columns
    )


def thd(
    data: FrameOrSeries,
    fundamental: float = 50.0,
    max_order: int = 40,
    window: str | tuple | None = None,
    interpolate_peak: bool = False,
) -> pd.Series:
    """Compute the total harmonic distortion (THD) per channel.

    Follows the IEC definition, which references the harmonic content to the
    fundamental amplitude:

    .. math:: 
        `\\mathrm{THD} = \\frac{\\sqrt{\\sum_{h=2}^{H} A_h^2}}{A_1}`

    Parameters
    ----------
    data : pandas.Series or pandas.DataFrame
        Uniformly sampled, real-valued time series without ``NaN``.
    fundamental : float, default 50.0
        Fundamental frequency in Hz.
    max_order : int, default 40
        Highest harmonic order ``H`` included in the sum. The IEC 61000-4-7
        default is 40. Orders above the Nyquist frequency contribute ``NaN``
        and therefore propagate ``NaN`` into the result — reduce ``max_order``
        or increase the sampling rate if that occurs.
    window : str or tuple or None, default None
        Window specification, forwarded to :func:`harmonics`.
    interpolate_peak : bool, default False
        Forwarded to :func:`harmonics`. Recommended together with a tapered
        window when sampling is not coherent.

    Returns
    -------
    pandas.Series
        THD as a dimensionless ratio (multiply by 100 for percent), indexed by
        the column names of the input and named ``"thd"``. Channels whose
        fundamental amplitude is exactly zero yield ``NaN`` instead of raising
        a division error.

    Raises
    ------
    ValueError
        Propagated from :func:`harmonics` / :func:`amplitude_spectrum`.

    See Also
    --------
    harmonics : Individual harmonic amplitudes.

    Notes
    -----
    This is the "THD-F" (fundamental-referenced) definition. The alternative
    "THD-R" convention divides by the total RMS value and yields slightly
    smaller numbers; the two agree closely for low distortion levels.

    Interharmonics and broadband noise are not captured — only exact integer
    multiples of ``fundamental`` are summed. Spectral leakage from
    non-coherent sampling can bias the value in either direction.

    Examples
    --------
    >>> float(thd(df, fundamental=50.0).round(5))   # 10 % 5th, 5 % 7th
    0.1118
    """
    amps = harmonics(
        data,
        fundamental=fundamental,
        orders=range(1, max_order + 1),
        window=window,
        interpolate_peak=interpolate_peak,
    ).abs()

    fund = amps.loc[1]
    higher = np.sqrt((amps.loc[2:] ** 2).sum(axis=0))
    result = higher / fund.replace(0.0, np.nan)
    return result.rename("thd")