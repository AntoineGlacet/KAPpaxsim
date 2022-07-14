# KIX_T1d_CUSBD.py
# includes:
# - KIX_T1_CUSBD_departure_sim_function
# - univariate_cost_function_generator_t1d_CUSBD
# - cost_function_t1d_CUSBD_EBS

import datetime
import heapq
import os
import random

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import simpy
from tqdm import tqdm
from math import ceil

from src.utils.profiles_from_schedule_new import SimParam

FREQ = "5min"
WINDOW = 1


def minutes_to_hms(minutes):
    if np.isnan(minutes):
        hms = np.nan
    else:
        hms = "{0:s} {1:0=2d}:{2:0=2d}:{3:0=2d}".format(
            "2020-10-13",
            int((minutes % 1440) // 60),
            int(minutes % 60),
            int((minutes % 1) * 60),
        )
    return hms


class Pax:
    """
    description
    describe row format and that we will use this for storing results
    """

    def __init__(self, simulation, row):
        self.simulation = simulation
        self.env = self.simulation.env
        self.row = row.copy()

        # data defining pax
        self.flight_number = row["Flight Number"]
        self.airline = self.flight_number.split(" ", 1)[0]
        self.std = row["Scheduled Time"]
        self.show_up_time = row["time"]
        # to put inside of simparam!
        self.type = "CUSBD"

    def depart(self):
        # advance sim time to pax show-up
        yield self.env.timeout(self.row["minutes"])
        if self.type == "CUSBD":
            # do kiosk process
            with self.simulation.resources["kiosk"].request() as request:
                self.row["kiosk_queue_length"] = len(
                    self.simulation.resources["kiosk"].queue
                )
                self.row["start_kiosk_queue"] = self.env.now
                yield request
                self.row["end_kiosk_queue"] = self.env.now
                yield self.env.process(self.checkin_2step_kiosk())
                self.row["end_kiosk_process"] = self.env.now

    def checkin_2step_kiosk(self):
        """check-in at Kiosk"""
        Pt_kiosk = self.simulation.simparam.dct_processes["kiosk"] / 60
        yield self.env.timeout(Pt_kiosk)


class Simulation:
    """
    description
    we need simparam to include:
        - dct_resources = { 'name': N}
        - dct_processes = { 'name': Pt} Pt in seconds
    """

    def __init__(self, simparam: SimParam):
        # init sim env
        env = simpy.Environment(initial_time=0)
        self.env = env
        self.simparam = simparam
        self.df_Counters = simparam.df_Counters

        # Create Simpy resources
        self.resources = {
            "kiosk": simpy.PriorityResource(
                self.env, self.simparam.dct_resource["kiosk"]
            ),
        }

        # Create Simpy resources specifically for check-in
        list_airlines = self.simparam.schedule["Airline Code"].unique()
        self.checkin = {
            airline: simpy.PriorityResource(self.env, 1) for airline in list_airlines
        }

    def generate_pax(self):
        # generate a Pax for each pax of df_Pax
        self.pax_list = [Pax(self, row) for _, row in self.simparam.df_Pax.iterrows()]
        for pax in self.pax_list:
            self.env.process(pax.depart())

    def format_df_result(self):
        # concatenate pax rows
        self.df_result = pd.concat(
            [pax.row for pax in self.pax_list], axis=1
        ).transpose()

        self.df_result.rename(
            columns={
                "checkin_kiosk_queue_length": "kiosk_queue_length",
                "start_checkin_kiosk_queue": "start_kiosk_queue",
                "end_checkin_kiosk_queue": "end_kiosk_queue",
                "end_checkin_kiosk_process": "end_kiosk_process",
            },
            inplace=True,
        )

        # different types of columns
        datetime_columns = [
            "start_kiosk_queue",
            "end_kiosk_queue",
            "end_kiosk_process",
        ]
        # list for iteration
        list_process_all = ["kiosk"]

        # change datetime columns to datetime
        for column in datetime_columns:
            self.df_result[column] = pd.to_datetime(
                self.df_result[column].apply(lambda x: minutes_to_hms(x))
            )

        # calculate waiting times
        for process in list_process_all:
            self.df_result["wait_time_{}".format(process)] = (
                self.df_result["end_{}_queue".format(process)]
                - self.df_result["start_{}_queue".format(process)]
            ).fillna(datetime.timedelta(0))

        # artificial high waiting time for pax who did not finish
        for process in list_process_all:
            mask = (pd.isna(self.df_result["end_{}_queue".format(process)])) & (
                pd.notna(self.df_result["start_{}_queue".format(process)])
            )

            self.df_result.loc[
                mask, "wait_time_{}".format(process)
            ] = datetime.timedelta(hours=8)

        # aggregates for plotting
        list_plot = [
            "start_{}_queue",
            "end_{}_process",
            "{}_queue_length",
            "wait_time_{}",
        ]
        self.dct_plot = {
            key: [plot.format(key) for plot in list_plot] for key in list_process_all
        }
        ratio_sampling = pd.to_timedelta("1H") / pd.to_timedelta(FREQ)

        # in
        self.plt_in = [
            (
                self.df_result.set_index(self.dct_plot[key][0], drop=False)["Pax"]
                .resample(FREQ)
                .agg(["sum"])
                .rolling(window=WINDOW, center=True)
                .mean()
                .dropna()
                .apply(lambda x: x * ratio_sampling)
            )
            for key in [*self.dct_plot]
        ]

        # out
        self.plt_out = [
            (
                self.df_result.set_index(self.dct_plot[key][1], drop=False)["Pax"]
                .resample(FREQ)
                .agg(["sum"])
                .rolling(window=WINDOW, center=True)
                .mean()
                .dropna()
                .apply(lambda x: x * ratio_sampling)
            )
            for key in [*self.dct_plot]
        ]

        # queue length
        self.plt_queue_length = [
            (
                self.df_result.set_index(self.dct_plot[key][0], drop=False)[
                    self.dct_plot[key][2]
                ]
                .resample(FREQ)
                .agg(["max"])
                .rolling(window=WINDOW, center=True)
                .mean()
            )
            for key in [*self.dct_plot]
        ]

        # queue duration
        self.plt_queue_duration = [
            (
                self.df_result.set_index(self.dct_plot[key][0], drop=False)[
                    self.dct_plot[key][3]
                ]
                .apply(lambda x: x.total_seconds() / 60)
                .resample(FREQ)
                .agg(["max"])
                .rolling(window=WINDOW, center=True)
                .mean()
            )
            for key in [*self.dct_plot]
        ]
        # histograms of queue duration and queue length

        self.plt_hist_wait_time = [
            (
                self.df_result[self.df_result[self.dct_plot[key][0]].notnull()][
                    self.dct_plot[key][3]
                ].apply(lambda x: x.total_seconds() / 60)
            )
            for key in [*self.dct_plot]
        ]

        self.dct_hist_wait_time = {
            key: (
                self.df_result[self.df_result[self.dct_plot[key][0]].notnull()][
                    self.dct_plot[key][3]
                ].apply(lambda x: x.total_seconds() / 60)
            )
            for key in [*self.dct_plot]
        }

        self.dct_hist_queue_length = {
            key: (
                self.df_result[self.df_result[self.dct_plot[key][0]].notnull()][
                    self.dct_plot[key][2]
                ]
            )
            for key in [*self.dct_plot]
        }

    def plot_result(self):
        n_graph = len([*self.dct_plot])

        # plot param
        xmin = pd.to_datetime("2020-10-13 00:00:00")
        xmax = pd.to_datetime("2020-10-14 00:00:00")
        plt.rcParams.update({"figure.autolayout": True})
        hours = mdates.HourLocator(interval=1)
        half_hours = mdates.MinuteLocator(byminute=[0, 30], interval=1)
        h_fmt = mdates.DateFormatter("%H:%M:%S")

        # plotting
        widths = [4, 1]
        gs_kw = dict(width_ratios=widths)

        fig = plt.figure(figsize=(16, 4 * n_graph))

        axs = fig.subplots(n_graph, 2, squeeze=True, gridspec_kw=gs_kw)
        ax2 = [axs[i, 0].twinx() for i in range(n_graph)]

        # plot for all processes, except 'wait for counter opening'
        for i in range(n_graph):
            axs[i, 0].plot(self.plt_in[i], label="in", lw=2)
            axs[i, 0].plot(self.plt_out[i], label="out", lw=2)
            axs[i, 0].plot(
                self.plt_queue_length[i], label="queue length", ls="--", lw=1
            )
            ax2[i].plot(
                self.plt_queue_duration[i],
                label="queue duration",
                color="r",
                ls="--",
                lw=1,
            )

            sns.histplot(self.plt_hist_wait_time[i], ax=axs[i, 1], bins=30)

            axs[i, 0].set(
                ylabel="Pax/hr or Pax",
                title=[*self.dct_plot][i],
                xlim=[xmin, xmax],
            )
            axs[i, 0].set_ylim(bottom=0)
            axs[i, 0].xaxis.set_major_locator(hours)
            axs[i, 0].xaxis.set_major_formatter(h_fmt)
            axs[i, 0].xaxis.set_minor_locator(half_hours)
            axs[i, 0].legend(loc="upper left")

            axs[i, 1].set_xlim(left=0)

            ax2[i].legend(loc="upper right")
            ax2[i].set(ylabel="waiting time [min]")
            ax2[i].set_ylim(bottom=0)
            ax2[i].spines["right"].set_color("r")
            ax2[i].tick_params(axis="y", colors="r")
            ax2[i].yaxis.label.set_color("r")

        # remove ticks labels for all grows except last
        for i in range(n_graph - 1):
            ax2[i].tick_params(
                axis="x",
                which="both",
                bottom=False,
                top=False,
                labelbottom=False,
            )
            axs[i, 0].tick_params(
                axis="x",
                which="both",
                bottom=False,
                top=False,
                labelbottom=False,
            )

        # format last row graphs xticks labels
        labels = axs[n_graph - 1, 0].get_xticklabels()
        plt.setp(labels, rotation=45, horizontalalignment="right")
        axs[n_graph - 1, 0].set(xlabel="time")

        plt.show()
