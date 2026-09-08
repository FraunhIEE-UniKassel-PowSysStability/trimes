import numpy as np
import pandas as pd

from trimes.base import is_uniformly_sampled, get_sampling_rate, time_to_seconds


def _window_kernel(x: np.ndarray | float, w: np.ndarray) -> np.ndarray:
    """Evaluate the normalised DFT kernel of a window at fractional bin offsets.

    For a window ``w`` of length ``n`` this returns

    .. math:: \\tilde{W}(x) = \\frac{1}{\\sum_m w[m]}
              \\sum_{m=0}^{n-1} w[m]\\, e^{-2\\pi j x m / n},

    i.e. the window's spectrum evaluated at the (generally non-integer) bin
    position ``x``, scaled so that ``kernel(0) == 1``.

    A real tone ``A * cos(2*pi*nu*m/n + phi)`` — with ``nu`` the frequency in
    bins — then produces the amplitude spectrum

    .. math:: S[k] = A\\left[e^{j\\phi}\\tilde{W}(k-\\nu)
                           + e^{-j\\phi}\\tilde{W}(k+\\nu)\\right],

    where the second term is the leakage of the negative-frequency image that
    :func:`estimate_frequency` removes iteratively.

    Parameters
    ----------
    x : float or numpy.ndarray
        Bin offsets at which to evaluate the kernel. May be non-integer.
    w : numpy.ndarray
        One-dimensional window samples (periodic convention).

    Returns
    -------
    numpy.ndarray
        Complex kernel values with the shape of ``numpy.atleast_1d(x)``.

    Notes
    -----
    Evaluated by direct summation rather than via a closed-form Dirichlet
    expression. This is exact for *any* window, avoids the removable
    singularities of the analytic form, and costs ``O(n)`` per offset — which is
    negligible because only a few offsets are needed per iteration.
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    m = np.arange(w.size)
    return (np.exp(-2j * np.pi * np.outer(x, m) / w.size) @ w) / w.sum()


def _ipdft_step(
    spectrum: np.ndarray, k: int, w: np.ndarray
) -> tuple[float, complex]:
    """Perform one three-point Hann IpDFT interpolation around bin ``k``.

    Parameters
    ----------
    spectrum : numpy.ndarray
        One-dimensional normalised amplitude spectrum (complex).
    k : int
        Index of the peak bin; must satisfy ``1 <= k <= spectrum.size - 2``.
    w : numpy.ndarray
        Window samples, used to invert the kernel for amplitude and phase.

    Returns
    -------
    delta : float
        Fractional bin offset of the true peak, clipped to ``[-0.5, 0.5]``.
    phasor : complex
        Complex amplitude of the tone, obtained by dividing the peak bin by the
        kernel value at ``-delta``. Its modulus is the amplitude, its argument
        the phase referenced to a cosine at the record start.

    Notes
    -----
    The interpolation formula

    .. math:: \\delta = \\frac{2(|S_{k+1}| - |S_{k-1}|)}
                             {|S_{k-1}| + 2|S_k| + |S_{k+1}|}

    is exact for the Hann window in the absence of leakage from other
    components; for other windows it is only an approximation.
    """
    left, centre, right = (abs(spectrum[k - 1]), abs(spectrum[k]), abs(spectrum[k + 1]))
    denominator = left + 2.0 * centre + right
    delta = 0.0 if denominator == 0.0 else 2.0 * (right - left) / denominator
    delta = float(np.clip(delta, -0.5, 0.5))
    phasor = spectrum[k] / _window_kernel(-delta, w)[0]
    return delta, phasor


def estimate_frequency(
    data: pd.DataFrame | pd.Series,
    *,
    nominal: float | None = 50.0,
    search_range: tuple[float, float] | None = None,
    iterations: int = 3,
    window: str | np.ndarray = "hann",
) -> pd.DataFrame:
    """Estimate frequency, amplitude and phase of the dominant tone per channel.

    Implements the enhanced interpolated DFT (e-IpDFT): a three-point Hann
    interpolation of the spectral peak, followed by iterative removal of the
    leakage caused by the tone's own negative-frequency image. This is the
    estimator class used by P-class phasor measurement units and stays accurate
    for records as short as two to three fundamental cycles.

    Parameters
    ----------
    data : pandas.DataFrame or pandas.Series
        Uniformly sampled, real-valued, NaN-free time-domain signal indexed by
        time. Each column is treated as an independent channel. A ``Series`` is
        promoted to a single-column frame.
    nominal : float or None, default 50.0
        Nominal system frequency in Hz. Used only to derive a default
        ``search_range`` of ``(0.8 * nominal, 1.2 * nominal)``, which prevents
        the estimator from locking onto a harmonic or onto a dominant DC
        component. Pass ``None`` to search the full spectrum.
    search_range : tuple of float, optional
        Explicit inclusive ``(f_min, f_max)`` band in Hz within which the peak
        bin is sought. Overrides the band derived from ``nominal``.
    iterations : int, default 3
        Number of image-leakage removal iterations. ``0`` reduces the method to
        plain IpDFT; ``2`` to ``3`` is sufficient in practice, and further
        iterations change the result by less than the noise floor.
    window : str or numpy.ndarray, default 'hann'
        Window applied before the transform. Only ``"hann"`` matches the
        interpolation formula used internally; a rectangular window
        (``"boxcar"``) or an explicit array of length ``len(data)`` is accepted
        for experimentation but degrades accuracy under spectral leakage.

    Returns
    -------
    pandas.DataFrame
        One row per channel, indexed by column name, with columns:

        ``frequency``
            Estimated frequency in Hz.
        ``amplitude``
            Estimated peak amplitude in units of the input signal.
        ``phase``
            Estimated phase in radians, referenced to a cosine at the first
            sample of the record.
        ``delta``
            Final fractional bin offset. Values approaching ``+-0.5`` indicate
            that the peak sits midway between bins, where accuracy is lowest.

    Raises
    ------
    ValueError
        If the signal is not uniformly sampled, contains NaN, has fewer than
        four samples, if ``search_range`` contains no usable bin, or if the peak
        falls on the first or last bin, where three-point interpolation is
        impossible.

    See Also
    --------
    frequency_trace : Applies this estimator to a sliding window.
    amplitude_spectrum : Underlying spectrum, with the same normalisation.

    Notes
    -----
    Accuracy is limited by the tone's isolation, not by the bin spacing. For a
    clean 50 Hz signal sampled at 10 kHz over three cycles, the frequency error
    is typically below 1 mHz. Interharmonics or a second tone inside the
    ``search_range`` bias the result, because only one component is modelled.

    The reported phase depends on the record's start time. Only phase
    *differences* between channels are meaningful unless the time reference is
    fixed deliberately, e.g. to a GPS second boundary as in a real PMU.

    Examples
    --------
    >>> fs = 10_000.0
    >>> t = np.arange(0.0, 0.06, 1.0 / fs)          # three cycles
    >>> x = 100.0 * np.cos(2 * np.pi * 49.37 * t + 0.5)
    >>> df = pd.DataFrame({"u_a": x}, index=pd.Index(t, name="time"))
    >>> result = estimate_frequency(df)
    >>> round(float(result.loc["u_a", "frequency"]), 3)
    49.37
    """
    if isinstance(data, pd.Series):
        data = data.to_frame()
    if not is_uniformly_sampled(data):
        raise ValueError(
            "estimate_frequency requires uniform sampling; use resample_uniform first."
        )

    values = data.to_numpy(dtype=float)
    if values.shape[0] < 4:
        raise ValueError("At least four samples are required for interpolation.")
    if not np.isfinite(values).all():
        raise ValueError("Input contains NaN or inf.")

    n = values.shape[0]
    fs = get_sampling_rate(data)
    bin_width = fs / n

    if isinstance(window, str):
        w = np.ones(n) if window in ("boxcar", "rectangular") else np.hanning(n + 1)[:-1]
    else:
        w = np.asarray(window, dtype=float)
        if w.shape != (n,):
            raise ValueError(f"Window must have length {n}, got {w.shape}.")

    spectrum = np.fft.rfft(values * w[:, None], axis=0) * 2.0 / w.sum()
    frequencies = np.fft.rfftfreq(n, 1.0 / fs)

    if search_range is None and nominal is not None:
        search_range = (0.8 * nominal, 1.2 * nominal)
    if search_range is None:
        candidates = np.ones(frequencies.size, dtype=bool)
    else:
        candidates = (frequencies >= search_range[0]) & (frequencies <= search_range[1])
    candidates[0] = False
    candidates[-1] = False
    if not candidates.any():
        raise ValueError(
            f"No interpolatable bin inside search_range={search_range}; "
            f"the bin width is {bin_width:g} Hz."
        )

    records = []
    for column in range(values.shape[1]):
        channel = spectrum[:, column]
        k = int(np.flatnonzero(candidates)[np.argmax(np.abs(channel[candidates]))])

        delta, phasor = _ipdft_step(channel, k, w)
        for _ in range(max(iterations, 0)):
            # Model and subtract the negative-frequency image of the estimated
            # tone, then re-interpolate on the corrected neighbourhood.
            nu = k + delta
            neighbourhood = np.array([k - 1, k, k + 1], dtype=float)
            image = np.conj(phasor) * _window_kernel(neighbourhood + nu, w)
            corrected = channel.copy()
            corrected[k - 1 : k + 2] -= image
            delta, phasor = _ipdft_step(corrected, k, w)

        records.append(
            {
                "frequency": (k + delta) * bin_width,
                "amplitude": abs(phasor),
                "phase": float(np.angle(phasor)),
                "delta": delta,
            }
        )

    return pd.DataFrame(records, index=pd.Index(data.columns, name="channel"))


def frequency_trace(
    data: pd.DataFrame | pd.Series,
    *,
    nominal: float = 50.0,
    window_cycles: float = 3.0,
    step_cycles: float = 1.0,
    rocof: bool = True,
    **estimator_kwargs,
) -> pd.DataFrame:
    """Track frequency over time with a sliding e-IpDFT estimator.

    Slides a window of ``window_cycles`` nominal cycles across the record,
    applies :func:`estimate_frequency` to each position, and timestamps the
    result at the window centre — the reporting convention of IEEE/IEC
    60255-118-1 synchrophasors.

    Parameters
    ----------
    data : pandas.DataFrame or pandas.Series
        Uniformly sampled, real-valued time-domain signal indexed by time.
    nominal : float, default 50.0
        Nominal frequency in Hz, defining the window and step length in
        seconds and the default search band of the estimator.
    window_cycles : float, default 3.0
        Window length in nominal cycles. Shorter windows react faster to
        transients; longer windows suppress noise and interharmonics. Below
        about two cycles, accuracy degrades even with image compensation.
    step_cycles : float, default 1.0
        Advance between consecutive windows in nominal cycles. Determines the
        reporting rate: ``1.0`` yields 50 estimates per second at 50 Hz.
    rocof : bool, default True
        If ``True``, add a ``rocof`` column with the rate of change of
        frequency in Hz/s, obtained from :func:`numpy.gradient` of the
        frequency trace. Differentiation amplifies noise, so consider
        smoothing the frequency column first for noisy measurements.
    **estimator_kwargs
        Forwarded to :func:`estimate_frequency`, e.g. ``iterations``,
        ``search_range`` or ``window``.

    Returns
    -------
    pandas.DataFrame
        Frame indexed by window-centre time, with a two-level column index
        ``(channel, quantity)`` where ``quantity`` is ``"frequency"``,
        ``"amplitude"``, ``"phase"`` and optionally ``"rocof"``. For
        single-channel input the channel level can be dropped with
        ``result.droplevel(0, axis=1)``.

    Raises
    ------
    ValueError
        If the record is shorter than one window, or if ``window_cycles`` or
        ``step_cycles`` is not positive.

    See Also
    --------
    estimate_frequency : Single-window estimator applied here.

    Notes
    -----
    The first and last estimates are centred half a window inside the record,
    so the trace is shorter than the input on both ends. During a frequency
    step, estimates whose window straddles the step are a weighted mix of both
    frequencies; the transition therefore smears over roughly one window
    length, which is the intrinsic response-time limit of any DFT estimator.

    Examples
    --------
    >>> trace = frequency_trace(df, nominal=50.0, window_cycles=3.0)
    >>> trace.droplevel(0, axis=1)[["frequency", "rocof"]].head()  # doctest: +SKIP
    """
    if isinstance(data, pd.Series):
        data = data.to_frame()
    if window_cycles <= 0.0 or step_cycles <= 0.0:
        raise ValueError("window_cycles and step_cycles must be positive.")

    time = time_to_seconds(data.index)
    fs = get_sampling_rate(data)
    n_window = int(round(window_cycles / nominal * fs))
    n_step = max(int(round(step_cycles / nominal * fs)), 1)
    if n_window < 4 or n_window > len(data):
        raise ValueError(
            f"Window of {window_cycles} cycles corresponds to {n_window} samples, "
            f"which does not fit into a record of {len(data)} samples."
        )

    starts = range(0, len(data) - n_window + 1, n_step)
    centres, rows = [], []
    for start in starts:
        segment = data.iloc[start : start + n_window]
        estimate = estimate_frequency(segment, nominal=nominal, **estimator_kwargs)
        centres.append(0.5 * (time[start] + time[start + n_window - 1]))
        rows.append(estimate[["frequency", "amplitude", "phase"]].stack())

    result = pd.DataFrame(rows, index=pd.Index(centres, name="time"))
    result.columns = pd.MultiIndex.from_tuples(
        result.columns, names=("channel", "quantity")
    )

    if rocof:
        for channel in result.columns.get_level_values(0).unique():
            frequency = result[(channel, "frequency")].to_numpy()
            gradient = (
                np.gradient(frequency, result.index.to_numpy(dtype=float))
                if frequency.size > 1
                else np.zeros_like(frequency)
            )
            result[(channel, "rocof")] = gradient
        result = result.sort_index(axis=1)

    return result


def referenced_phase(
    trace: pd.DataFrame, *, nominal: float = 50.0, unwrap: bool = False
) -> pd.DataFrame:
    """De-rotate a phase trace onto a common nominal reference.

    Each estimate in a sliding trace is referenced to its own window centre, so
    a constant-phase signal appears as a phase that rotates at ``nominal``.
    Subtracting ``2*pi*nominal*t`` yields a phase that is constant for a
    nominal-frequency signal and therefore makes angular steps directly visible.

    Parameters
    ----------
    trace : pandas.DataFrame
        Output of :func:`frequency_trace`, with a ``(channel, quantity)``
        column index containing a ``"phase"`` quantity.
    nominal : float, default 50.0
        Reference frequency in Hz.
    unwrap : bool, default False
        If ``True``, unwrap the result instead of wrapping it to
        ``(-pi, pi]``. Useful for off-nominal signals, whose referenced phase
        ramps linearly.

    Returns
    -------
    pandas.DataFrame
        Frame of referenced phases in radians, indexed like ``trace``, with one
        column per channel.
    """
    t = trace.index.to_numpy(dtype=float)
    phases = trace.xs("phase", axis=1, level="quantity")
    referenced = phases.to_numpy() - 2.0 * np.pi * nominal * t[:, None]
    if unwrap:
        referenced = np.unwrap(referenced, axis=0)
    else:
        referenced = np.angle(np.exp(1j * referenced))
    return pd.DataFrame(referenced, index=trace.index, columns=phases.columns)

