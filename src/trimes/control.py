import control
import pandas as pd
import matplotlib.pyplot as plt
from icecream import ic

from trimes.plots import (
    add_point_to_plot,
    add_vertical_line_to_plot,
)


def step_info_series(ts: pd.Series, y_initial: float | None = None, **kwargs) -> dict:
    """Get information on step response. Interface to 'control' package's function 'step_info'.

    See python-control.readthedocs.io/en/latest/generated/control.step_info.html

    Args:
        ts (pd.Series): time series

        y_initial (float | None, optional): Initial value before step. Defaults to None (first value in 'ts' is used).

        kwargs: kwargs for control.step_info

    Returns:
        dict: step information
    """
    ts_np = ts.to_numpy()
    if y_initial is None:
        y_initial = ts_np[0]
    ts_np = ts_np - y_initial
    return control.step_info(ts_np, ts.index.values, **kwargs)


def plot_step_info(
    ts: pd.Series,
    step_info: dict,
    y_initial: float | None = None,
    alpha=0.7,
    linestyle="--",
) -> None:
    """Plot step information.

    Args:
        ts (pd.Series): time series

        step_info (dict): corresponding step info

        y_initial (float | None, optional): Initial value before step. Defaults to None (first value in 'ts' is used).

        alpha (float, optional): plot argument. Defaults to 0.7.

        linestyle (str, optional): plot argument. Defaults to "--".
    """
    if y_initial is None:
        y_initial = ts.iloc[0]
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
            val += y_initial
            plt.axhline(
                y=val,  # vertical line
                label=info,
                color=ax._get_lines.get_next_color(),
                **plt_args,
            )
        elif info in ["Overshoot"]:
            if val != 0:
                val = y_final + val / 100 * (y_final - y_initial)
                plt.axhline(
                    y=val,  # vertical line
                    label=info,
                    color=ax._get_lines.get_next_color(),
                    **plt_args,
                )
        elif info in ["Undershoot"]:
            if val != 0:
                val = y_initial - val / 100 * (y_final - y_initial)
                print(val)
                plt.axhline(
                    y=val,  # vertical line
                    label=info,
                    color=ax._get_lines.get_next_color(),
                    **plt_args,
                )
        elif info in ["Peak"]:
            t = step_info.get("PeakTime")
            if t:
                val += y_initial
                add_point_to_plot(t, val, va="bottom", label=info, color="red")
            else:
                raise Exception("'PeakTime' must be available to plot 'Peak'")
