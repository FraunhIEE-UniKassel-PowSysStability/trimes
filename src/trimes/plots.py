import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes._axes import Axes
from cycler import cycler
from icecream import ic

from trimes.base import resample


def plot_2y(
    ts_axis1: pd.DataFrame | pd.Series,
    ts_axis2: pd.DataFrame | pd.Series,
    kwargs_ax1: dict = {},
    kwargs_ax2: dict = {},
    colors: list | None = None,
) -> tuple[Axes, Axes]:
    """Plot with two y-axes (i.e. a secondary y-axis on the right).

    Args:
        ts_axis1 (pd.DataFrame | pd.Series): time series for left axis
        ts_axis2 (pd.DataFrame | pd.Series): time series for right axis
        kwargs_ax1 (dict, optional): kwargs for left axis. Defaults to {}.
        kwargs_ax2 (dict, optional): kwargs for right axis. Defaults to {}.
        colors (list | None, optional): colors used. Defaults to None (default colormap is used).

    Returns:
        tuple[Axes, Axes]: left and right axes
    """
    ts_axis1.plot(**kwargs_ax1)
    ax1 = plt.gca()
    plt.legend(loc="upper left")

    ax2 = ax1.twinx()
    if colors is None:
        colors = get_pyplot_default_colormap_hex()
    color_index = ts_axis1.shape[1] % len(colors)
    color_cycler = cycler(color=colors[color_index:] + colors[:color_index])
    ax2.set_prop_cycle(color_cycler)
    ts_axis2.plot(ax=ax2, **kwargs_ax2)
    plt.legend(loc="upper right")
    return ax1, ax2


def fill_between(
    ax: Axes, ts1: pd.Series, ts2: pd.Series, resample_ts: bool = True, **kwargs
) -> None:
    """Fill area between time series 'ts1' and 'ts2'.

    Args:
        ax (Axes): matpotlib axes
        ts1 (pd.Series): time series
        ts2 (pd.Series): time series
        resample_ts (bool): If True, 'ts2' is resampled according to the index of 'ts1'. Default is True.
        kwargs: keyword arguments for 'ax.fill_between'
    """
    if resample_ts:
        ts2 = resample(ts2, ts1.index.values)
    ax.fill_between(ts1.index.values, ts1.to_numpy(), ts2.to_numpy(), **kwargs)


def add_vertical_line_to_plot(x: float | int, **kwargs):
    plt.axvline(
        x=x,  # vertical line
        **kwargs,
    )


def add_point_to_plot(
    x: float, y: float, label: str = "", ha="center", va="top", **kwargs
):
    """Add point to plot.

    Args:
        x (float): x coordinate
        y (float): y coordinate
        label (str, optional): Defaults to "".
        ha: 'center', 'right', 'left'
        va: 'top', 'bottom', 'center', 'baseline', 'center_baseline'
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
