# utils.py
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import numpy as np
import pandas as pd
import seaborn as sns


def minutes_to_hms(minutes):
    """
    transforms a number of minutes to a str of hms on 2020-10-13
    for datetime formatting
    """
    if np.isnan(minutes):
        hms = np.nan
    else:
        hms = (
            "2020-10-13"
            f" {int((minutes % 1440) // 60):0=2d}"
            f":{int(minutes % 60):0=2d}"
            f":{int((minutes % 1) * 60):0=2d}"
        )
    return hms


def day_graph():
    """returns fig, ax with standard formatting"""
    sns.set_theme(style="whitegrid")
    # plot param
    xmin = pd.to_datetime("2020-10-13 00:00:00")
    xmax = pd.to_datetime("2020-10-14 00:00:00")
    hours = mdates.HourLocator(interval=1)
    half_hours = mdates.MinuteLocator(byminute=[0, 30], interval=1)
    h_fmt = mdates.DateFormatter("%H:%M")
    k_com = tick.FuncFormatter(lambda x, p: format(int(x), ","))

    # formatting
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim((xmin, xmax))
    ax.set_xticks(pd.date_range(xmin, xmax, freq="30min"))
    ax.set_xticklabels(ax.get_xticks(), rotation=45, **{"horizontalalignment": "right"})
    ax.set(
        ylabel="Pax/hr",
    )
    ax.xaxis.set_major_locator(hours)
    ax.xaxis.set_major_formatter(h_fmt)
    ax.xaxis.set_minor_locator(half_hours)
    ax.yaxis.set_major_formatter(k_com)

    return fig, ax
