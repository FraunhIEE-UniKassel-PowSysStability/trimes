from typing import Literal, Sequence

import pandas as pd
import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.axes._axes import Axes
    from cycler import cycler
except ModuleNotFoundError as _exc:  # pragma: no cover - depends on env
    raise ModuleNotFoundError(
        "trimes.plots needs matplotlib; install it with: pip install 'trimes[viz]'"
    ) from _exc


from trimes.base import resample


def plot_2y(
    ts_axis1: pd.DataFrame | pd.Series,
    ts_axis2: pd.DataFrame | pd.Series,
    kwargs_ax1: dict | None = None,
    kwargs_ax2: dict | None = None,
    colors: list | None = None,
) -> tuple[Axes, Axes]:
    """Plot two time series on a shared x-axis with independent y-axes.

    ``ts_axis2`` is drawn against a secondary y-axis on the right, and its
    colour cycle is offset so its curves do not clash with ``ts_axis1``'s.

    Parameters
    ----------
    ts_axis1 : pandas.DataFrame or pandas.Series
        Time series for the left y-axis.
    ts_axis2 : pandas.DataFrame or pandas.Series
        Time series for the right y-axis.
    kwargs_ax1, kwargs_ax2 : dict, optional
        Keyword arguments forwarded to the respective ``.plot`` call.
    colors : list, optional
        Colour cycle to use. Defaults to Matplotlib's current cycle.

    Returns
    -------
    tuple of matplotlib.axes.Axes
        The left and right axes.
    """
    kwargs_ax1 = {} if kwargs_ax1 is None else kwargs_ax1
    kwargs_ax2 = {} if kwargs_ax2 is None else kwargs_ax2
    ts_axis1.plot(**kwargs_ax1)
    ax1 = plt.gca()
    plt.legend(loc="upper left")

    ax2 = ax1.twinx()
    if colors is None:
        colors = get_pyplot_default_colormap_hex()
    n_curves_ax1 = ts_axis1.shape[1] if ts_axis1.ndim == 2 else 1
    color_index = n_curves_ax1 % len(colors)
    color_cycler = cycler(color=colors[color_index:] + colors[:color_index])
    ax2.set_prop_cycle(color_cycler)
    ts_axis2.plot(ax=ax2, **kwargs_ax2)
    plt.legend(loc="upper right")
    return ax1, ax2


def fill_between(
    ts1: pd.Series,
    ts2: pd.Series,
    ax: Axes | None = None,
    resample_ts: bool = True,
    **kwargs,
) -> None:
    """Shade the area between two time series.

    Parameters
    ----------
    ts1, ts2 : pandas.Series
        The two boundary curves.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. Defaults to the current axes.
    resample_ts : bool, default True
        If True, resample ``ts2`` onto ``ts1``'s index first (the two series
        must otherwise already share a sample grid).
    **kwargs
        Forwarded to :meth:`~matplotlib.axes.Axes.fill_between`.
    """
    if resample_ts:
        ts2 = resample(ts2, ts1.index.values)
    if not ax:
        ax = plt.gca()
    ax.fill_between(ts1.index.values, ts1.to_numpy(), ts2.to_numpy(), **kwargs)


def add_vertical_line_to_plot(x: float | int, **kwargs):
    plt.axvline(
        x=x,  # vertical line
        **kwargs,
    )


def add_point_to_plot(
    x: float, y: float, label: str = "", ha="center", va="top", **kwargs
):
    """Mark a single point on the current axes, optionally with a text label.

    Parameters
    ----------
    x, y : float
        Point coordinates.
    label : str, optional
        Text drawn at the point; omitted if empty.
    ha : {'center', 'right', 'left'}, default 'center'
        Horizontal alignment of the label.
    va : {'top', 'bottom', 'center', 'baseline', 'center_baseline'}, default 'top'
        Vertical alignment of the label.
    **kwargs
        Forwarded to :meth:`~matplotlib.axes.Axes.plot` for the marker.
    """
    ax = plt.gca()
    ax.plot(x, y, "o", **kwargs)
    if label:
        plt.gca().text(x, y, label, ha=ha, va=va)


def get_pyplot_default_colormap_hex():
    return plt.rcParams["axes.prop_cycle"].by_key()["color"]


def get_pyplot_default_color_cycler(start_index=0):
    cmap = get_pyplot_default_colormap_hex()
    return cycler(color=cmap[start_index:] + cmap[:start_index])


def plot_bode(
    spectrum: pd.DataFrame,
    *,
    magnitude: Literal["db", "linear"] = "db",
    db_reference: float = 1.0,
    freq_scale: Literal["log", "linear"] = "log",
    phase_unit: Literal["deg", "rad"] = "deg",
    unwrap: bool = True,
    mask_below: float | None = 1e-3,
    style: Literal["line", "stem"] = "line",
    freq_range: tuple[float | None, float | None] | None = None,
    mark_frequencies: Sequence[float] | None = None,
    axes: tuple | None = None,
    legend: bool | None = None,
    **plot_kwargs,
):
    """Plot a complex spectrum as a Bode diagram (magnitude and phase).

    Creates two vertically stacked, x-axis-linked subplots: magnitude over
    frequency on top, phase over frequency below.

    Parameters
    ----------
    spectrum : pandas.DataFrame
        Complex, frequency-indexed spectrum as returned by
        :func:`amplitude_spectrum` with ``complex_output=True`` or by
        :func:`rfft`. Each column is drawn as a separate curve using the column
        name as its label.
    magnitude : {'db', 'linear'}, default 'db'
        Scaling of the magnitude axis:

        * ``"db"`` — ``20 * log10(abs(X) / db_reference)``, i.e. the amplitude
          (not power) convention. Bins with exactly zero magnitude become
          ``NaN`` and are left blank instead of plotted at ``-inf``.
        * ``"linear"`` — ``abs(X)`` in units of the input signal, drawn on a
          linear y-axis.
    db_reference : float, default 1.0
        Reference value for the dB conversion, in units of the input signal.
        Ignored unless ``magnitude="db"``. Use e.g. the fundamental amplitude
        to obtain harmonic levels relative to it (dBc).
    freq_scale : {'log', 'linear'}, default 'log'
        Scaling of the shared frequency axis. With ``"log"`` the DC bin
        (0 Hz) is dropped, since it cannot be represented on a logarithmic
        axis. Use ``"linear"`` for harmonic spectra of power-system signals,
        where equally spaced harmonics are easier to read.
    phase_unit : {'deg', 'rad'}, default 'deg'
        Unit of the phase axis. With ``"deg"`` and ``unwrap=False`` the y-ticks
        are set to multiples of 90 degrees.
    unwrap : bool, default True
        If ``True``, remove jumps larger than ``pi`` along the frequency axis
        via :func:`numpy.unwrap` before masking, producing a continuous phase
        curve. If ``False``, the phase stays wrapped to ``(-180, 180]`` degrees.
    mask_below : float or None, default 1e-3
        Relative magnitude threshold for the phase plot. Bins whose magnitude
        is below ``mask_below * max(abs(X))`` of their own column are replaced
        by ``NaN`` in the phase subplot, because their angle is determined by
        numerical noise only. Set to ``None`` to plot all phases unmasked. The
        magnitude subplot is never masked.
        Bins below the threshold become ``NaN``. With ``style="line"`` and a
        sparse spectrum this can leave isolated valid bins that no line segment
        connects; these are drawn as individual markers.
    style : {'line', 'stem'}, default 'line'
        Rendering of the magnitude curve. ``"line"`` connects the bins and
        suits broadband spectra; ``"stem"`` draws one vertical line with a
        marker per bin and suits sparse harmonic spectra. The phase subplot
        always uses markers when ``style="stem"``, since a connecting line
        across masked bins would be misleading.
    freq_range : tuple of (float or None, float or None), optional
        Inclusive ``(f_min, f_max)`` limits in Hz applied to the data *before*
        plotting, so that dB ranges and the ``mask_below`` threshold refer to
        the visible band only. ``None`` for either entry leaves that side
        unbounded. If ``None``, the full spectrum is used.
    mark_frequencies : sequence of float, optional
        Frequencies in Hz at which a thin vertical guide line is drawn in both
        subplots, e.g. the expected harmonic frequencies. Purely cosmetic; no
        values are read from the spectrum.
    axes : tuple of matplotlib.axes.Axes, optional
        Pre-existing ``(ax_magnitude, ax_phase)`` pair to draw into, for
        embedding the diagram into a larger figure. If ``None``, a new figure
        with two subplots is created.
    legend : bool, optional
        Whether to draw a legend on the magnitude axis. If ``None``, a legend
        is drawn only when ``spectrum`` has more than one column.
    **plot_kwargs
        Additional keyword arguments forwarded to every
        :meth:`~matplotlib.axes.Axes.plot` call (e.g. ``linewidth``,
        ``alpha``, ``marker``). Applied to both subplots.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing the diagram — the parent figure of ``axes`` if
        one was supplied.
    axes : tuple of matplotlib.axes.Axes
        The ``(ax_magnitude, ax_phase)`` pair, for further customisation such
        as setting limits or adding annotations.

    Raises
    ------
    ImportError
        If ``matplotlib`` is not installed.
    ValueError
        If ``spectrum`` is not complex (magnitude and phase cannot both be
        derived from real magnitudes), if ``freq_range`` or ``freq_scale``
        leaves no bins to plot, or if an unknown option is passed to
        ``magnitude``, ``freq_scale``, ``phase_unit`` or ``style``.

    See Also
    --------
    amplitude_spectrum : Produces a suitable complex input.
    phase_spectrum : Phase values without the magnitude subplot.

    Notes
    -----
    The phase of an amplitude spectrum is referenced to a cosine at the start
    of the record, so ``sin(2*pi*f*t)`` appears at ``-90`` degrees. Since a
    time shift of the record rotates all phases linearly with frequency, only
    phase *differences* between channels or between harmonics are meaningful
    unless the time reference is fixed deliberately.

    Windowing is not compensated here beyond what
    :func:`amplitude_spectrum` already does. A tapered window widens every
    peak, so narrow features in the plot reflect the window's main lobe rather
    than the true bandwidth of the component.

    Examples
    --------
    >>> spec = amplitude_spectrum(df, complex_output=True)
    >>> fig, _ = plot_bode(
    ...     spec,
    ...     freq_scale="linear",
    ...     freq_range=(0.0, 1000.0),
    ...     style="stem",
    ...     mark_frequencies=[50.0, 250.0, 350.0],
    ... )
    >>> fig.savefig("spectrum.png", dpi=150)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "plot_bode requires matplotlib. Install it with "
            "`pip install matplotlib`."
        ) from exc

    if not np.iscomplexobj(spectrum.to_numpy()):
        raise ValueError(
            "plot_bode requires a complex spectrum. Call amplitude_spectrum "
            "with complex_output=True, or use rfft."
        )
    if magnitude not in ("db", "linear"):
        raise ValueError(f"Unknown magnitude option {magnitude!r}.")
    if freq_scale not in ("log", "linear"):
        raise ValueError(f"Unknown freq_scale option {freq_scale!r}.")
    if phase_unit not in ("deg", "rad"):
        raise ValueError(f"Unknown phase_unit option {phase_unit!r}.")
    if style not in ("line", "stem"):
        raise ValueError(f"Unknown style option {style!r}.")

    freq = spectrum.index.to_numpy(dtype=float)
    keep = np.ones(freq.size, dtype=bool)
    if freq_scale == "log":
        keep &= freq > 0.0
    if freq_range is not None:
        f_min, f_max = freq_range
        if f_min is not None:
            keep &= freq >= f_min
        if f_max is not None:
            keep &= freq <= f_max
    if not keep.any():
        raise ValueError("No frequency bins left to plot; check freq_range.")

    freq = freq[keep]
    values = spectrum.to_numpy()[keep]
    columns = list(spectrum.columns)

    # --- magnitude -------------------------------------------------------- #
    mag = np.abs(values)
    if magnitude == "db":
        with np.errstate(divide="ignore"):
            mag_plot = 20.0 * np.log10(mag / db_reference)
        mag_plot[~np.isfinite(mag_plot)] = np.nan
        mag_label = f"magnitude in dB (ref. {db_reference:g})"
    else:
        mag_plot = mag
        mag_label = "magnitude"

    # --- phase ------------------------------------------------------------ #
    phase = np.angle(values)
    if unwrap:
        phase = np.unwrap(phase, axis=0)
    if phase_unit == "deg":
        phase = np.degrees(phase)
    if mask_below is not None:
        peak = np.nanmax(mag, axis=0, keepdims=True)
        phase = np.where(mag >= mask_below * peak, phase, np.nan)

    # --- axes ------------------------------------------------------------- #
    if axes is None:
        fig, (ax_mag, ax_phase) = plt.subplots(
            2, 1, sharex=True, figsize=(8.0, 6.0), height_ratios=(2, 1)
        )
    else:
        ax_mag, ax_phase = axes
        fig = ax_mag.get_figure()

    baseline = np.nanmin(mag_plot) if magnitude == "db" else 0.0
    phase_kwargs = dict(plot_kwargs)
    if style == "stem":
        phase_kwargs.setdefault("linestyle", "none")
        phase_kwargs.setdefault("marker", "o")
        phase_kwargs.setdefault("markersize", 3.0)

    for i, name in enumerate(columns):
        label = str(name)
        if style == "stem":
            color = ax_mag._get_lines.get_next_color()
            ax_mag.vlines(freq, baseline, mag_plot[:, i], color=color, linewidth=1.0)
            ax_mag.plot(
                freq, mag_plot[:, i], color=color, linestyle="none",
                marker="o", markersize=3.0, label=label, **plot_kwargs
            )
            ax_phase.plot(freq, phase[:, i], color=color, label=label, **phase_kwargs)
        else:
            (line,) = ax_mag.plot(freq, mag_plot[:, i], label=label, **plot_kwargs)
            color = line.get_color()
            ax_phase.plot(freq, phase[:, i], color=color, label=label, **phase_kwargs)

            # A masked phase can leave isolated bins that no line segment
            # reaches; mark them explicitly so they stay visible.
            for ax, y in ((ax_mag, mag_plot[:, i]), (ax_phase, phase[:, i])):
                lonely = _isolated_points(y)
                if lonely.any():
                    ax.plot(
                        freq[lonely],
                        y[lonely],
                        color=color,
                        linestyle="none",
                        marker="o",
                        markersize=3.0,
                    )

    for f_mark in mark_frequencies or ():
        for ax in (ax_mag, ax_phase):
            ax.axvline(f_mark, color="0.6", linewidth=0.8, linestyle=":", zorder=0)

    ax_mag.set_xscale(freq_scale)
    ax_mag.set_ylabel(mag_label)
    ax_mag.grid(True, which="both", alpha=0.3)
    if magnitude == "linear":
        ax_mag.set_ylim(bottom=0.0)

    ax_phase.set_xlabel("frequency in Hz")
    ax_phase.set_ylabel(f"phase in {'degrees' if phase_unit == 'deg' else 'rad'}")
    ax_phase.grid(True, which="both", alpha=0.3)
    if phase_unit == "deg" and not unwrap:
        ax_phase.set_ylim(-180.0, 180.0)
        ax_phase.set_yticks([-180.0, -90.0, 0.0, 90.0, 180.0])

    ax_phase.set_xlim(freq[0], freq[-1])

    if legend is None:
        legend = len(columns) > 1
    if legend:
        ax_mag.legend(loc="best", fontsize="small")

    fig.align_ylabels((ax_mag, ax_phase))
    return fig, (ax_mag, ax_phase)


def _isolated_points(y: np.ndarray) -> np.ndarray:
    """Return a mask of finite samples whose neighbours are both non-finite.

    Such samples cannot be rendered by a plain line plot, because a line
    segment requires two adjacent finite values. They are drawn as markers
    instead.

    Parameters
    ----------
    y : numpy.ndarray
        One-dimensional array of plot values, possibly containing ``NaN``.

    Returns
    -------
    numpy.ndarray
        Boolean mask of the same shape as ``y``.
    """
    valid = np.isfinite(y)
    has_left = np.concatenate(([False], valid[:-1]))
    has_right = np.concatenate((valid[1:], [False]))
    return valid & ~has_left & ~has_right
