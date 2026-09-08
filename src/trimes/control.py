"""Step-response analysis (needs the ``control`` extra: python-control +
matplotlib).
"""

try:
    import control
except ModuleNotFoundError as _exc:  # pragma: no cover - depends on env
    raise ModuleNotFoundError(
        "trimes.control needs the optional 'control' package. "
        "Install it with: pip install 'trimes[control]'"
    ) from _exc

import matplotlib.pyplot as plt
import pandas as pd

from trimes.plots import (  # raises a helpful error if matplotlib is missing
    add_point_to_plot,
    add_vertical_line_to_plot,
)

__all__ = ["step_info_series", "plot_step_info"]


def step_info_series(
    ts: pd.Series,
    yinitial: float | None = None,
    yfinal: float | None = None,
    **kwargs,
) -> dict:
    """Step-response characteristics of ``ts``.

    Thin interface to :func:`control.step_info`
    (python-control.readthedocs.io/en/latest/generated/control.step_info.html).
    The series is shifted so the step starts at zero before it is passed on.

    Parameters
    ----------
    ts : pandas.Series
        The step response over time.
    yinitial : float, optional
        Value before the step. Defaults to the first sample.
    yfinal : float, optional
        Steady-state value after the step. Defaults to letting
        :func:`control.step_info` infer it.
    **kwargs
        Forwarded to :func:`control.step_info`.

    Returns
    -------
    dict
        The step-info dictionary (``RiseTime``, ``SettlingTime``,
        ``Overshoot``, ...).
    """
    ts_np = ts.to_numpy()
    if yinitial is None:
        yinitial = ts_np[0]
    ts_np = ts_np - yinitial
    return control.step_info(ts_np, ts.index.values, yfinal=yfinal, **kwargs)


def plot_step_info(
    ts: pd.Series,
    step_info: dict,
    yinitial: float | None = None,
    alpha: float = 0.7,
    linestyle: str = "--",
) -> None:
    """Annotate the current axes with markers for a step-info dictionary.

    Draws vertical lines for the settling and peak times, horizontal lines for
    the settling band, over/undershoot and steady-state value, and a marker at
    the peak.

    Parameters
    ----------
    ts : pandas.Series
        The step response that ``step_info`` describes.
    step_info : dict
        Output of :func:`step_info_series` (or :func:`control.step_info`).
    yinitial : float, optional
        Value before the step. Defaults to the first sample of ``ts``.
    alpha : float, default 0.7
        Line alpha.
    linestyle : str, default "--"
        Line style.
    """
    if yinitial is None:
        yinitial = ts.iloc[0]
    y_final = ts.iloc[-1]

    ax = plt.gca()
    plt_args = {
        "alpha": alpha,
        "linestyle": linestyle,
    }
    for info, val in step_info.items():
        if info in ["SettlingTime", "PeakTime"]:
            add_vertical_line_to_plot(
                val,
                label=info,
                color=ax._get_lines.get_next_color(),
                **plt_args,
            )
        elif info in ["SettlingMin", "SettlingMax", "SteadyStateValue"]:
            val += yinitial
            plt.axhline(
                y=val,
                label=info,
                color=ax._get_lines.get_next_color(),
                **plt_args,
            )
        elif info in ["Overshoot"]:
            if val != 0:
                val = y_final + val / 100 * (y_final - yinitial)
                plt.axhline(
                    y=val,
                    label=info,
                    color=ax._get_lines.get_next_color(),
                    **plt_args,
                )
        elif info in ["Undershoot"]:
            if val != 0:
                val = yinitial - val / 100 * (y_final - yinitial)
                plt.axhline(
                    y=val,
                    label=info,
                    color=ax._get_lines.get_next_color(),
                    **plt_args,
                )
        elif info in ["Peak"]:
            t = step_info.get("PeakTime")
            if t:
                val += yinitial
                add_point_to_plot(t, val, va="bottom", label=info, color="red")
            else:
                raise Exception("'PeakTime' must be available to plot 'Peak'")
