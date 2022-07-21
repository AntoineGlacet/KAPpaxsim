# profiles_from_schedule.py

import datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from decouple import AutoConfig
from scipy.interpolate import interp1d
from scipy.stats import norm
from tqdm import tqdm


class SimParam:
    """
    Class containing the simulation parameters
    schedule, show-up, airline codes, Pt, N, etc...
    """

    def __init__(
        self,
        dotenv_file_path: Path = Path(__file__).parent / "../../data/secret/.env",
    ):
        # ref to config file path
        self.config = AutoConfig(search_path=dotenv_file_path)

        # load show_up_profiles
        self.path_param = (
            Path(__file__).parent / ".." / ".." / self.config("ADRM_param_full_path")
        )

        # import the airline_code
        self.airline_code = pd.read_excel(
            self.path_param,
            sheet_name=r"airline_code",
            header=0,
        )
        # LCC list (2 char A/L codes)
        mask_LCC = self.airline_code["FSC / LCC"] == "LCC"
        self.airline_code[mask_LCC]
        self.list_LCC = self.airline_code[mask_LCC]["airline code"].to_numpy(
            dtype="str"
        )

    def schedule_from_path(self, path: Path):
        """
        excel file should be formatted with following columns
        | A/D | T1/T2(MM/9C/7C/TW) | Intl Regions | Category(P/C/O) | Sector |
        | Flight Number | SEATS FC | PAX_SUM FC | Flight Date | Scheduled Time |
        """
        self.path = path
        self.schedule = pd.read_excel(
            self.path,
            sheet_name=r"schedule",
            header=0,
        )
        self.schedule_origin = self.schedule.copy()

        return self

    def schedule_from_df(self, dataframe: pd.DataFrame):
        """
        table should be formatted with following columns
        | A/D | T1/T2(MM/9C/7C/TW) | Intl Regions | Category(P/C/O) | Sector |
        | Flight Number | SEATS FC | PAX_SUM FC | Flight Date | Scheduled Time |
        """
        self.schedule = dataframe.copy()
        self.schedule_origin = self.schedule

        return self

    def schedule_cleanup(self):
        """
        some cleaning operations on the schedule to correct inputs mistakes
        """
        # for easy handling of timestamps later (to be reviewed)
        self.schedule["Scheduled Time"] = "2020-10-13 " + self.schedule[
            "Scheduled Time"
        ].astype(str)
        self.schedule["Scheduled Time"] = pd.to_datetime(
            self.schedule["Scheduled Time"]
        )
        # to catch up some formatting mistakes from beontra extracts...
        self.schedule["Flight Number"] = self.schedule["Flight Number"].replace(
            ["JX821"], "JX 821"
        )
        self.schedule["Flight Number"] = self.schedule["Flight Number"].replace(
            ["NS*****"], "NS *****"
        )
        # split Airline Code
        self.schedule["Airline Code"] = self.schedule["Flight Number"].str.split(
            " ", 1, expand=True
        )[0]
        # bad formatting of 0 pax flight I guess? for JAL 8126
        self.schedule["PAX_SUM FC"].replace("-", 0, inplace=True)
        # store for reference
        self.schedule_clean = self.schedule

        return self

    def schedule_filter(
        self,
        direction: str = "D",
        sector: str = "I",
        terminal: str = "T1",
        date_str: str = "2017-03-19",
    ):
        filtered_data = self.schedule[
            (
                (self.schedule["A/D"] == direction)
                & (self.schedule["Sector"] == sector)
                & (self.schedule["Category(P/C/O)"] == "P")
                & (self.schedule["T1/T2(MM/9C/7C/TW)"] == terminal)
                & (self.schedule["Flight Date"] == pd.Timestamp(date_str))
                # remove 0 Pax flights
                & (self.schedule["PAX_SUM FC"] != 0)
            )
        ]
        filtered_data = filtered_data.reset_index()
        self.schedule = filtered_data
        self.schedule_filtered = self.schedule

        return self

    def assign_flight_show_up_category_default(self):
        """
        adds 'show_up_category' col to self.schedule
        assigns EARLY, China, LCC or FSC category
        """
        for i in range(len(self.schedule)):
            std = self.schedule.loc[i, "Scheduled Time"]
            if std < pd.to_datetime("2020-10-13 08:00:00") and std >= pd.to_datetime(
                "2020-10-13 02:00:00"
            ):
                category = "EARLY"

            elif self.schedule.loc[i, "Intl Regions"] == "China":
                category = "CHINA"

            elif self.schedule.loc[i, "Flight Number"][0:2] in self.list_LCC:
                category = "LCC"

            else:
                category = "FSC"

            self.schedule.loc[i, "show_up_category"] = category

        self.schedule_show_up_cat = self.schedule

        return self

    def define_norm_show_up(self, df_show_up_profiles: pd.DataFrame):
        # import show-up from a df
        # | name | loc | scale |
        # | FSC  | 60  |  30   |
        # loc = mean
        # scale = standard deviation
        # /!\ requires a 'show_up_category' col in self.schedule
        self.df_show_up_profiles = df_show_up_profiles

        self.dct_f_show_up = {
            name: lambda x, name=name: 1
            - norm.cdf(
                x,
                loc=self.df_show_up_profiles.loc[
                    self.df_show_up_profiles["name"] == name, "loc"
                ].iat[0],
                scale=self.df_show_up_profiles.loc[
                    self.df_show_up_profiles["name"] == name, "scale"
                ].iat[0],
            )
            for name in self.df_show_up_profiles["name"]
        }

        x = np.linspace(0, 360, 100)

        self.dct_f_inv_linear = {
            name: interp1d(self.dct_f_show_up[name](x), x, kind="linear")
            for name in self.dct_f_show_up
        }

        return self

    def show_up_from_file(self):
        """
        import from the ADRM param excel file
        # import show-up from a df
        # | name | loc | scale |
        # | FSC  | 60  |  30   |
        # loc = mean
        # scale = standard deviation
        # /!\ requires a 'show_up_category' col in self.schedule
        """
        show_up_ter = pd.read_excel(
            self.path_param,
            sheet_name=r"terminal",
            header=1,
        )

        list_cat = ["EARLY", "CHINA", "LCC", "FSC"]

        dct_show_up_profiles = {
            "name": [cat for cat in list_cat],
            "loc": [
                show_up_ter.loc[0, "cumulative distribution {}".format(cat)]
                for cat in list_cat
            ],
            "scale": [
                show_up_ter.loc[1, "cumulative distribution {}".format(cat)]
                for cat in list_cat
            ],
        }

        self.df_show_up_profiles = pd.DataFrame.from_dict(dct_show_up_profiles)
        self.define_norm_show_up(self.df_show_up_profiles)

        return self

    def assign_show_up(self):
        # we need a column "show_up_category" in the schedule dataframe
        list_time_Pax = []
        list_flights = []
        list_ST = []
        list_category = []

        for i in range(len(self.schedule)):
            N_flight_pax = int(self.schedule.loc[i, "PAX_SUM FC"])
            STD = self.schedule.loc[i, "Scheduled Time"]
            y = np.linspace(0.01, 0.99, N_flight_pax)
            show_up_category = self.schedule.loc[i, "show_up_category"]
            f_ter_inv_linear = self.dct_f_inv_linear[show_up_category]

            temps_Terminal = (
                self.schedule.loc[i, "Scheduled Time"].hour * 60
                + self.schedule.loc[i, "Scheduled Time"].minute
                - f_ter_inv_linear(y)
            )

            for t in temps_Terminal:
                t = datetime.datetime(
                    year=2020,
                    month=10,
                    day=13,
                    hour=int((t % (24 * 60)) / 60),
                    minute=int(t % 60),
                    second=int(t % 1 * 60),
                )
                list_time_Pax.append(t)
                list_flights.append(self.schedule.loc[i, "Flight Number"])
                list_ST.append(self.schedule.loc[i, "Scheduled Time"])
                list_category.append(show_up_category)

        dct_Pax = {
            "Flight Number": list_flights,
            "time": list_time_Pax,
            "Scheduled Time": list_ST,
            "Category": list_category,
        }
        df_Pax = pd.DataFrame(dct_Pax)

        self.df_Pax = df_Pax
        self.df_Pax["Pax"] = 1
        self.prepare_schedule_for_simulation()

        return self

    def prepare_schedule_for_simulation(self):
        self.df_Pax["minutes"] = (
            self.df_Pax["time"].dt.hour.astype(int) * 60
            + self.df_Pax["time"].dt.minute.astype(int)
            + self.df_Pax["time"].dt.second.astype(int) / 60
        )

        self.df_Pax["Airline"] = self.df_Pax["Flight Number"].apply(
            lambda x: x.split(" ", 1)[0]
        )

        return self

    def assign_check_in(
        self,
        start_time: float = 2.5,
        onecounter_time: float = 0.75,
        base_n_counter: float = 4,
        seats_per_add_counter: float = 60,
    ):
        # change to 5-minutes slot unit
        onecounter_slot = -int(((onecounter_time) * 60) // 5)
        start_slot = -int(((start_time) * 60) // 5)

        # create a dictionnary of airline and seats per 5 minutes
        dct_Seats = {
            airline: [0 for i in range(int(24 * 60 / 5))]
            for airline in self.schedule["Airline Code"]
        }

        # go through all flights
        for flight_number in self.schedule["Flight Number"]:
            mask = self.schedule["Flight Number"] == flight_number
            airline = self.schedule.loc[mask, "Airline Code"].iat[0]
            STD_5min = (
                self.schedule.loc[mask, "Scheduled Time"]
                .apply(lambda x: (x.hour * 60 + x.minute) // 5)
                .iat[0]
            )
            # we add number of seats to 5-minute rounded STD position in dct_Seats
            dct_Seats[airline][STD_5min] += self.schedule[(mask)]["SEATS FC"].iat[0]

        # create a df over 3 days to avoid errors for flights close to midnight
        df_Seats = pd.DataFrame.from_dict(dct_Seats)
        df_Counters = df_Seats * 0
        df_Counters_3d = (
            df_Counters.append(df_Counters).append(df_Counters).reset_index(drop=True)
        )

        offset = 288

        # First we add the seats for 2.5=>0.75 hours before STD
        for col in range(len(df_Seats.columns)):
            for i in range(len(df_Seats.index)):
                # When we see a cell with Seats for a flight
                if df_Seats.iloc[i, col] != 0:
                    # Wee check from 45 minutes to 2.5 hours before STD
                    for j in range(start_slot, onecounter_slot):
                        # for each cell, if there is already a number, we put add the seats
                        df_Counters_3d.iloc[i + offset + j, col] = (
                            df_Counters_3d.iloc[i + offset + j, col]
                            + df_Seats.iloc[i, col]
                        )
        # now we have a table with seats, let's apply the rule
        # valid on that period
        for col in range(len(df_Counters_3d.columns)):
            for i in range(len(df_Counters_3d.index)):
                if df_Counters_3d.iloc[i, col] > 0:
                    df_Counters_3d.iloc[i, col] = max(
                        base_n_counter,
                        base_n_counter
                        + 1
                        + (
                            (df_Counters_3d.iloc[i, col] - 201) // seats_per_add_counter
                        ),
                    )

        # Then we do the last 45 minutes
        for col in range(len(df_Seats.columns)):
            for i in range(len(df_Seats.index)):
                # When we see a cell with Seats for a flight
                if df_Seats.iloc[i, col] != 0:
                    # we check from STD to 45 minutes before
                    for j in range(onecounter_slot, 1):
                        # only if no other flights are checking in, do we add a counter
                        if df_Counters_3d.iloc[i + offset + j, col] == 0:
                            df_Counters_3d.iloc[i + offset + j, col] = 1

        # merge into only 1d
        df_Counters_final = df_Counters.copy()
        for i in range(len(df_Counters_final.index)):
            df_Counters_final.iloc[i, :] = (
                df_Counters_3d.iloc[i, :]
                + df_Counters_3d.iloc[i + offset, :]
                + df_Counters_3d.iloc[i + 2 * offset, :]
            )
        df_Counters_final["total"] = df_Counters_final.sum(axis=1)

        self.df_Counters = df_Counters_final

        # we need dct_Counters_change
        # { airline : series of time/n_counter when change }
        self.dct_Counters_change = {
            airline: self.df_Counters[airline].loc[
                self.df_Counters[airline].shift() != self.df_Counters[airline]
            ]
            for airline in self.schedule["Airline Code"].unique()
        }

        return self

    def plot_show_up_categories_profiles(self):
        sns.set_theme(style="whitegrid")
        fig, ax = plt.subplots(figsize=(8, 6))
        x = np.linspace(0, 360, 100)

        for key in self.dct_f_show_up:
            ax.plot(
                x,
                self.dct_f_show_up[key](x),
                label="show-up {}".format(key),
            )

        ax.invert_xaxis()
        ax.legend()

        plt.show()

    def plot_df_Pax(self, by_pax_type: bool = False, freq: str = "5min", win=12):
        sns.set_theme(style="whitegrid")
        ratio_sampling = pd.to_timedelta("1H") / pd.to_timedelta(freq)

        if by_pax_type == True:
            pax_types = self.df_Pax["pax_type"].unique()
            plot = {
                pax_types[i]: self.df_Pax.loc[
                    self.df_Pax["pax_type"].isin(pax_types[0 : i + 1])
                ]
                .set_index("time", drop=False)["Pax"]
                .resample(freq)
                .agg(["sum"])
                .rolling(window=win, center=True)
                .mean()
                .dropna()
                .apply(lambda x: x * ratio_sampling)
                for i in range(len(pax_types))
            }

        else:
            plot = (
                self.df_Pax.set_index("time", drop=False)["Pax"]
                .resample(freq)
                .agg(["sum"])
                .rolling(window=win, center=True)
                .mean()
                .dropna()
                .apply(lambda x: x * ratio_sampling)
            )

        fig, ax = plt.subplots(figsize=(8, 5))

        if by_pax_type == True:
            previous_plot = plot[pax_types[0]] * 0
            for pax_type in plot:
                ax.plot(plot[pax_type], label="{} show-up".format(pax_type))
                ax.fill_between(
                    plot[pax_type].index,
                    plot[pax_type]["sum"],
                    previous_plot["sum"],
                    alpha=0.2,
                )
                previous_plot = plot[pax_type]
        else:
            ax.plot(plot, label="total show-up")

        # plot param
        xmin = pd.to_datetime("2020-10-13 00:00:00")
        xmax = pd.to_datetime("2020-10-14 00:00:00")
        plt.rcParams.update({"figure.autolayout": True})
        hours = mdates.HourLocator(interval=1)
        half_hours = mdates.MinuteLocator(byminute=[0, 30], interval=1)
        h_fmt = mdates.DateFormatter("%H:%M")

        # formatting
        ax.set_xlim((xmin, xmax))
        ax.set_xticks(pd.date_range(xmin, xmax, freq="5min"))
        ax.set_xticklabels(
            ax.get_xticks(), rotation=45, **{"horizontalalignment": "right"}
        )
        ax.xaxis.set_major_locator(hours)
        ax.xaxis.set_major_formatter(h_fmt)
        ax.xaxis.set_minor_locator(half_hours)
        ax.legend(loc="upper left", frameon=False)

        plt.show()

    def plot_std(self, freq: str = "1H", win=1):
        sns.set_theme(style="whitegrid")

        ratio_sampling = pd.to_timedelta("1H") / pd.to_timedelta(freq)

        # plot param
        xmin = pd.to_datetime("2020-10-13 00:00:00")
        xmax = pd.to_datetime("2020-10-14 00:00:00")
        hours = mdates.HourLocator(interval=1)
        half_hours = mdates.MinuteLocator(byminute=[0, 30], interval=1)
        h_fmt = mdates.DateFormatter("%H:%M")
        width = 0.7 * (xmax - xmin) / 24
        offset = pd.Timedelta("30min")

        # formatting
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_xlim((xmin, xmax))
        ax.set_xticks(pd.date_range(xmin, xmax, freq="30min"))
        ax.set_xticklabels(
            ax.get_xticks(), rotation=45, **{"horizontalalignment": "right"}
        )
        ax.xaxis.set_major_locator(hours)
        ax.xaxis.set_major_formatter(h_fmt)
        ax.xaxis.set_minor_locator(half_hours)

        # calculation
        plot = (
            self.schedule.set_index("Scheduled Time", drop=False)["PAX_SUM FC"]
            .resample(freq)
            .agg(["sum"])
            .rolling(window=win, center=True)
            .mean()
            .dropna()
            .apply(lambda x: x * ratio_sampling)
        )
        # plot
        ax.bar(x=plot.index + offset, height=plot["sum"], width=width, align="center")
        ax.text(
            0.7,
            0.9,
            "total = {:,} Pax".format(plot["sum"].sum()),
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
        )
        plt.show()

        return plot

    def plot_std_profiles(self):
        pass

    def plot_show_up_profiles(self):
        pass
