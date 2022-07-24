# KIX_T1d_CUSBD_new.py

import datetime

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import simpy
from tqdm import tqdm

from src.utils.simparam import SimParam


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


class CustomResource(simpy.PriorityResource):
    """
    define a custom resource with extra functions
    """

    def __init__(self, env, startup_capacity: int, airline):
        self.max_capacity = 200
        super().__init__(env, self.max_capacity)
        self.current_capacity = self.max_capacity
        self.dummy_requests_list = []
        self.env = env
        self.set_capacity(startup_capacity)

    def set_capacity(self, target_capacity):
        # use dummy priority 0 request to manage capacity
        # we need to store the request to be able to release them later
        diff_capa = self.current_capacity - target_capacity
        for i in range(abs(diff_capa)):
            if diff_capa > 0:
                dummy_request = self.request(priority=0)
                self.dummy_requests_list.append(dummy_request)
            else:
                self.release(self.dummy_requests_list[-1])
                self.dummy_requests_list.pop(-1)
        self.current_capacity = target_capacity

    def change_capa_per_schedule(self, serie_Counters_change):
        previous_time = 0
        for time, n_counters in serie_Counters_change.items():
            if time == 0:
                continue
            yield self.env.timeout((time - previous_time) * 5)
            previous_time = time
            self.set_capacity(int(n_counters))


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
        self.type = row["pax_type"]
        self.process_sequence = self.simulation.simparam.dct_process_sequence[
            self.type
        ][0]
        self.hour_to_std = self.simulation.simparam.dct_process_sequence[self.type][1][
            "hour_to_std"
        ]

    def go_to_process(self, process_str: str):
        """
        Pax queues and do process
        Result stored to pax.row
        """
        if process_str == "wait_opening":
            yield self.env.process(self.wait_opening())
        else:
            if process_str == "checkin":
                resource = self.simulation.checkin[self.airline]
            else:
                resource = self.simulation.resources[process_str]

            with resource.request(priority=2) as request:
                # store queue length and start of queue time
                self.row[f"{process_str}_queue_length"] = len(resource.queue)
                self.row[f"start_{process_str}_queue"] = self.env.now
                # request usage start queueing
                yield request
                # store end of the queue time
                self.row[f"end_{process_str}_queue"] = self.env.now
                # do the process
                yield self.env.process(self.do_process(process_str))
                # store end of process time
                self.row[f"end_{process_str}_process"] = self.env.now

    def do_process(self, process_str: str):
        """Pax does process, simple timeout"""
        Pt = self.simulation.simparam.dct_processes[process_str] / 60
        yield self.env.timeout(Pt)

    def wait_opening(self):
        """
        wait for time_to_std before STD before going to next step
        record results!
        """
        std = self.std.hour * 60 + self.std.minute + self.std.second / 60
        t = std - self.hour_to_std * 60 - self.env.now
        # if std < self.env.now is that flight
        # is after midnight and pax arrives before midnight
        if std - self.env.now < 0:
            t += 24 * 60
        if t > 0:
            yield self.env.timeout(t)

    def depart(self):
        # advance sim time to pax show-up
        yield self.env.timeout(self.row["minutes"])

        # do process successively
        for process_str in self.process_sequence:
            yield self.env.process(self.go_to_process(process_str))


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
        self.dct_Counters_change = simparam.dct_Counters_change

        # Create normal resources
        self.resources = {
            process_str: simpy.PriorityResource(
                self.env, self.simparam.dct_resource[process_str]
            )
            for process_str in self.simparam.dct_resource
        }

    def generate_checkin(self):
        """
        create the checkin resources and
        make their capacity vary according to df_Counters info
        we use dummy priority 0 (top) requests to block counters
        """
        # Create custom resources
        list_airlines = self.simparam.schedule["Airline Code"].unique()
        self.checkin = {
            airline: CustomResource(
                self.env, int(self.dct_Counters_change[airline][0]), airline=airline
            )
            for airline in list_airlines
        }

        # generate change of capacity events at the right time
        for airline in self.checkin:
            self.env.process(
                self.checkin[airline].change_capa_per_schedule(
                    self.dct_Counters_change[airline]
                )
            )

        return self

    def generate_pax(self):
        # generate a Pax for each pax of df_Pax
        self.pax_list = [Pax(self, row) for _, row in self.simparam.df_Pax.iterrows()]
        for pax in self.pax_list:
            self.env.process(pax.depart())

        return self

    def run(self, end_time=1440):
        # run simulation for a 24h
        with tqdm(total=end_time - 1, desc="Simulation running...") as runpbar:
            for i in range(1, end_time):
                self.env.run(until=i)
                runpbar.update(1)

    def format_df_result(
        self, filter_airline: str = None, freq: str = "5min", win: int = 3
    ):
        # concatenate pax rows
        self.df_result = pd.concat(
            [pax.row for pax in self.pax_list], axis=1
        ).transpose()

        # add airline col
        self.df_result["Airline"] = self.df_result["Flight Number"].apply(
            lambda x: x.split(" ", 1)[0]
        )

        # sampling ratio
        ratio_sampling = pd.to_timedelta("1H") / pd.to_timedelta(freq)

        # list for iteration
        list_process_all = ["kiosk", "checkin", "CUSBD", "security"]

        # different types of columns
        datetime_columns_types = [
            "start_{}_queue",
            "end_{}_queue",
            "end_{}_process",
        ]

        datetime_columns = [
            a.format(b) for b in list_process_all for a in datetime_columns_types
        ]

        # change datetime columns to datetime
        for column in datetime_columns:
            self.df_result[column] = pd.to_datetime(
                self.df_result[column].apply(lambda x: minutes_to_hms(x))
            )

        # calculate waiting times
        for process in list_process_all:
            self.df_result[f"wait_time_{process}"] = (
                self.df_result[f"end_{process}_queue"]
                - self.df_result[f"start_{process}_queue"]
            ).fillna(datetime.timedelta(0))

        # artificial high waiting time for pax who did not finish
        for process in list_process_all:
            mask = (pd.isna(self.df_result[f"end_{process}_queue"])) & (
                pd.notna(self.df_result[f"start_{process}_queue"])
            )

            self.df_result.loc[mask, f"wait_time_{process}"] = datetime.timedelta(
                hours=8
            )

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

        # option to filter by airline
        if filter_airline is not None:
            if type(filter_airline) != list:
                filter_airline = [filter_airline]
            mask = self.df_result["Airline"].isin(filter_airline)
            df_result = self.df_result[mask].copy()
            # put some dummy values to not bother plotting
            u = df_result.select_dtypes(include=["datetime"])
            df_result.loc[0, u.columns] = pd.Timestamp("2020-10-14 21:40:00")

            u = df_result.select_dtypes(include=object)
            df_result.loc[0, u.columns] = 0
        else:
            df_result = self.df_result.copy()

        # in
        self.plt_in = [
            (
                df_result.set_index(self.dct_plot[key][0], drop=False)["Pax"]
                .resample(freq)
                .agg(["sum"])
                .rolling(window=win, center=True)
                .mean()
                .dropna()
                .apply(lambda x: x * ratio_sampling)
            )
            for key in [*self.dct_plot]
        ]

        # out
        self.plt_out = [
            (
                df_result.set_index(self.dct_plot[key][1], drop=False)["Pax"]
                .resample(freq)
                .agg(["sum"])
                .rolling(window=win, center=True)
                .mean()
                .dropna()
                .apply(lambda x: x * ratio_sampling)
            )
            for key in [*self.dct_plot]
        ]

        # queue length
        self.plt_queue_length = [
            (
                df_result.set_index(self.dct_plot[key][0], drop=False)[
                    self.dct_plot[key][2]
                ]
                .dropna()
                .resample(freq)
                .agg(["max"])
                .rolling(window=win, center=True)
                .mean()
            )
            for key in [*self.dct_plot]
        ]

        # queue duration
        self.plt_queue_duration = [
            (
                df_result.set_index(self.dct_plot[key][0], drop=False)[
                    self.dct_plot[key][3]
                ]
                .apply(lambda x: x.total_seconds() / 60)
                .resample(freq)
                .agg(["max"])
                .rolling(window=win, center=True)
                .mean()
            )
            for key in [*self.dct_plot]
        ]

        # histograms of queue duration and queue length
        self.dct_hist_wait_time = {
            key: (
                df_result[df_result[self.dct_plot[key][0]].notnull()][
                    self.dct_plot[key][3]
                ].apply(lambda x: x.total_seconds() / 60)
            )
            for key in [*self.dct_plot]
        }

        self.dct_hist_queue_length = {
            key: (
                df_result[df_result[self.dct_plot[key][0]].notnull()][
                    self.dct_plot[key][2]
                ]
            )
            for key in [*self.dct_plot]
        }

        self.plt_hist_wait_time = [value for value in self.dct_hist_wait_time.values()]

        return self

    def plot_result(self):
        n_graph = len([*self.dct_plot])
        # we force n_graph>=2 for 2-dim axs (for indexing)
        dummy_graph = False
        n_graph_dummy = n_graph
        if n_graph == 1:
            n_graph_dummy = n_graph + 1
            dummy_graph = True

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

        fig = plt.figure(figsize=(16, 4 * n_graph_dummy))

        axs = fig.subplots(n_graph_dummy, 2, squeeze=True, gridspec_kw=gs_kw)
        ax2 = [axs[i, 0].twinx() for i in range(n_graph_dummy)]

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
            axs[i, 1].text(
                0.7,
                0.9,
                f"total = {self.plt_hist_wait_time[i].count():,} Pax",
                horizontalalignment="center",
                verticalalignment="center",
                transform=axs[i, 1].transAxes,
            )

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

        # remove dummy if needed
        if dummy_graph:
            axs[1, 0].remove()
            axs[1, 1].remove()
            ax2[1].remove()

        plt.show()
