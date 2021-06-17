# import the libraries required to do the work
import datetime
import os
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from decouple import AutoConfig
from matplotlib.ticker import FuncFormatter
from scipy.interpolate import interp1d
from scipy.stats import norm
from tqdm import tqdm

def show_up_function(
    target_peak=2900,
    direction="D",
    system="security",
    ratio=1,
    terminal="T1",
    CTG_type="A",
    custom_showup=False,
    custom_counter_rule=False,
    **kwargs,
):
    """
    target_peak : target_peak peak hour value (STA or STD). The load factor of each flight will be modified accordingly
    direction : A/D for arrival or departure
    system: from (terminal, security, CTG, boarding, deboarding), which show_up profiles to apply to the schedule
    ratio: ratio applied to each Load Factor for each flight (used to subdivide Pax. eg. Y/J; Jap/Foreign; lvl2/lvl3)

    The function will first identify which year to consider amongst the available schedule forecast
    Then, it will apply the relevant show-up profile to the selected year, considering the target_peak peak hour value
    and the ratio if any.

    custom_showup then requires **kwargs to define the norm.cdf values
    
    loc_FSC = kwargs["loc_FSC"]
    scale_FSC = kwargs["scale_FSC"]
    loc_LCC = kwargs["loc_LCC"]
    scale_LCC = kwargs["scale_LCC"]
    loc_CHINA = kwargs["loc_CHINA"]
    scale_CHINA = kwargs["scale_CHINA"]
    loc_EARLY = kwargs["loc_EARLY"]
    scale_EARLY = kwargs["scale_EARLY"]
    
    custom_counter_rule then requires **kwargs to define custom rule
    
    start_time = kwargs["start_time"]
    onecountertimer = kwargs["onecounter_time"]
    base_n_counter = kwargs["base_n_counter"]
    seats_per_add_counter = kwargs["seats_per_add_counter"]

    """

    # =============================== preparatory work for all peak hour extractions============================================
    
    # give the paths to schedule forecast and show-up profiles
    # we also use the "airline code" sheet for show-up profiles!
    # move all to ADRM_parameters_full (new) file on sharepoint
    # and download to project folder
    # get env variables (for schedule and show-up files paths)
    DOTENV_FILE_PATH = Path(__file__).parent / "../../data/secret/.env"
    config = AutoConfig(search_path=DOTENV_FILE_PATH)
    path_forecasts = Path(__file__).parent / ".."/ ".." / config('FY2019_FY2025_xlsx_path')
    path_show_up = Path(__file__).parent / ".."/ ".." / config('ADRM_param_full_xlsx_path')  

    # if custom showup, assign the mean and STD
    if custom_showup == True:
        loc_FSC = kwargs["loc_FSC"]
        scale_FSC = kwargs["scale_FSC"]
        loc_LCC = kwargs["loc_LCC"]
        scale_LCC = kwargs["scale_LCC"]
        loc_CHINA = kwargs["loc_CHINA"]
        scale_CHINA = kwargs["scale_CHINA"]
        loc_EARLY = kwargs["loc_EARLY"]
        scale_EARLY = kwargs["scale_EARLY"]

    # import the schedule from the excel file produced by Aero department    
    data = pd.read_excel(
        path_forecasts,
        sheet_name=r"IntlP_FY19-FY25",
        header=0,
    )

    # format a Schedules time column to make a Timeserie later on
    data["5min Interval"] = (
        data["5min Interval"].astype(str).str.pad(width=4, side="left", fillchar="0")
    )

    data["Scheduled Time"] = "2020-10-13 " + data["5min Interval"].astype(str)
    data["Scheduled Time"] = pd.to_datetime(data["Scheduled Time"])

    data["Flight Number"] = data["Flight Number"].replace(["JX821"], "JX 821")

    # fonction pour renvoyer les peak_hour
    def peak_hour(
        FY=2025, direction="D", weekday="Saturday", sector="I", terminal="T1"
    ):
        filtered_data = data[
            (
                (data["A/D"] == direction)
                & (data["Day Of Week"] == weekday)
                & (data["Int/Dom"] == sector)
                & (data["Category(P/C/O)"] == "Passenger")
                & (data["T1/T2(MM/9C/7C/TW)"] == terminal)
                & (data["FY"] == "FY{}".format(FY))
            )
        ]

        filtered_data = filtered_data.set_index("Scheduled Time")
        filtered_data = filtered_data.sort_index()

        peak_value = (
            filtered_data["PAX_SUM FC"]
            .resample("60S")
            .agg(["sum"])
            .rolling(window=60, center=True)
            .sum()
            .max()[0]
        )
        return peak_value

    # calcul des peak hours
    FY_list = [i for i in range(2019, 2026)]
    direction_list = ["STA", "STD"]

    df_peak = pd.DataFrame(index=FY_list, columns=direction_list)
    for dir in direction_list:
        for FY in FY_list:
            df_peak.loc[FY, dir] = peak_hour(
                FY=FY,
                direction=dir[-1:],
                weekday="Saturday",
                sector="I",
                terminal=terminal,
            )
    df_peak.replace(0, np.nan, inplace=True)
    # declare some constants (consider making it differently?)
    sector = "I"
    weekday = "Saturday"
    airline_code = pd.read_excel(
        path_show_up,
        sheet_name=r"airline_code",
        header=0,
    )

    # let's find the FY corresponding best to the target_peak
    if target_peak < df_peak["ST{}".format(direction)].min():
        FY = df_peak[
            (
                df_peak["ST{}".format(direction)]
                == df_peak["ST{}".format(direction)].min()
            )
        ].index[0]
        schedule_peak = df_peak["ST{}".format(direction)].min()
    else:
        maxmin_peak = max(
            i for i in df_peak["ST{}".format(direction)] if i <= target_peak
        )
        FY = df_peak[(df_peak["ST{}".format(direction)] == maxmin_peak)].index[0]
        schedule_peak = maxmin_peak
    # for later, let's store the selected year peak value
    # ===========================================  function start ================================================
    # filter
    filtered_data = data[
        (
            (data["A/D"] == direction)
            & (data["Day Of Week"] == weekday)
            & (data["Int/Dom"] == sector)
            & (data["Category(P/C/O)"] == "Passenger")
            & (data["T1/T2(MM/9C/7C/TW)"] == terminal)
            & (data["FY"] == "FY{}".format(FY))
        )
    ]
    filtered_data = filtered_data.reset_index()
    data = filtered_data
    # ====================================== Counters =====================================
    if system == "check-in":
        # NEW fix some input mistakes
        data["Flight Number"] = data["Flight Number"].replace(["JX821"], "JX 821")
        data["Flight Number"] = data["Flight Number"].replace(["NS*****"], "NS *****")
        # split Airline Code
        data["Airline Code"] = data["Flight Number"].str.split(" ", 1, expand=True)[0]

        # NEW
        start_time = 2.5 # hours before STD for check-in opening
        onecounter_time = 0.75 # hours before STD with only one counter
        base_n_counter = 4
        seats_per_add_counter = 60
        
        # in case we change checkin counter allocation rule
        if custom_counter_rule == True:
            start_time = kwargs["start_time"]
            onecounter_time = kwargs["onecounter_time"]
            base_n_counter  = kwargs["base_n_counter"]
            seats_per_add_counter = kwargs["seats_per_add_counter"]
            
        onecounter_slot = -int(((onecounter_time)*60)//5)
        start_slot = -int(((start_time)*60)//5)

        
        # create a dictionnary of airline and seats per 5 minutes
        # initialize with all {airline_code : [0...0]}
        dico = {
            airline_code: [0 for i in range(int(24 * 60 / 5))]
            for airline_code in data["Airline Code"]
        }

        # boucle sur les airlines
        for airline_code in data["Airline Code"].unique():

            # boucle sur les flight code
            for flight_number in data[(data["Airline Code"] == airline_code)][
                "Flight Number"
            ]:

                # round down 5 minutes le STD
                time = data[data["Flight Number"] == flight_number][
                    "Scheduled Time"
                ].iloc[0]
                STD_5interval = (time.hour * 60 + time.minute) // 5

                # on met le nombre de seats du vol à la position qui va bien dans les listes du dico
                dico[airline_code][STD_5interval] = (
                    dico[airline_code][STD_5interval]
                    + data[
                        (data["Scheduled Time"] == time)
                        & (data["Flight Number"] == flight_number)
                    ]["SEATS FC"].iloc[0]
                )

        df_Seats = pd.DataFrame.from_dict(dico)

        # initialize some dataframes
        df_Counters = pd.DataFrame().reindex_like(df_Seats)
        for col in df_Counters.columns:
            df_Counters[col].values[:] = int(0)

        # create a df over 3 days to avoid errors for flights close to midnight
        df_Counters_previous_day = df_Counters.copy()
        df_Counters_next_day = df_Counters.copy()
        df_Counters_previous_day = df_Counters_previous_day.reindex(
            index=["day-1 {}".format(i) for i in range(0, 288)]
        )
        df_Counters_next_day = df_Counters_next_day.reindex(
            index=["day+1 {}".format(i) for i in range(0, 288)]
        )

        df1 = df_Counters_previous_day
        df2 = df_Counters
        df3 = df_Counters_next_day

        df_Counters_3d = df1.append(df2).append(df3)
        df_Counters_3d = df_Counters_3d.fillna(0)

        offset = 288

        # First we add the seats for 2.5 hours before STD
        # to 45 min before STD
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
                if 0 < df_Counters_3d.iloc[i, col]:  
                    df_Counters_3d.iloc[i, col] = max(
                        base_n_counter,
                        df_Counters_3d.iloc[i, col] // seats_per_add_counter,
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

    # now we do all the show-up

    # ====================================== Terminal =====================================
    # For Terminal
    if system == "terminal":
        # import of the excel with the show up profiles
        show_up_ter = pd.read_excel(
            path_show_up,
            sheet_name=r"terminal",
            header=1,
        )
        show_up_ter = show_up_ter.drop([0, 1], axis=0)
        show_up_ter = show_up_ter.reset_index(drop=True)

        # interpolation of show_up profiles and inverse functions
        x = show_up_ter["time before STD"].to_numpy(dtype=float)

        yFSC = show_up_ter["cumulative distribution FSC"].to_numpy(dtype=float)
        yLCC = show_up_ter["cumulative distribution LCC"].to_numpy(dtype=float)
        yEARLY = show_up_ter["cumulative distribution EARLY"].to_numpy(dtype=float)
        yCHINA = show_up_ter["cumulative distribution CHINA"].to_numpy(dtype=float)

        f_ter_FSC = interp1d(x, yFSC, kind="linear")
        f_ter_LCC = interp1d(x, yLCC, kind="linear")
        f_ter_EARLY = interp1d(x, yEARLY, kind="linear")
        f_ter_CHINA = interp1d(x, yCHINA, kind="linear")

        if custom_showup == True:
            f_ter_FSC = lambda x: 1 - norm.cdf(x, loc=loc_FSC, scale=scale_FSC)
            f_ter_LCC = lambda x: 1 - norm.cdf(x, loc=loc_LCC, scale=scale_LCC)
            f_ter_EARLY = lambda x: 1 - norm.cdf(x, loc=loc_EARLY, scale=scale_EARLY)
            f_ter_CHINA = lambda x: 1 - norm.cdf(x, loc=loc_CHINA, scale=scale_CHINA)

        f_ter_FSC_inv_linear = interp1d(f_ter_FSC(x), x, kind="linear")
        f_ter_LCC_inv_linear = interp1d(f_ter_LCC(x), x, kind="linear")
        f_ter_EARLY_inv_linear = interp1d(f_ter_EARLY(x), x, kind="linear")
        f_ter_CHINA_inv_linear = interp1d(f_ter_CHINA(x), x, kind="linear")

        # let's allocate profiles to flight and apply the ratios to Pax
        list_time_Pax = []
        list_flights = []
        list_ST = []
        for i in range(len(filtered_data)):
            N_flight_pax = int(
                filtered_data.loc[i, "PAX_SUM FC"]
                * ratio
                * (target_peak / schedule_peak)
            )
            STD = filtered_data.loc[i, "Scheduled Time"]
            y = np.linspace(0.0001, 0.995, N_flight_pax)

            if filtered_data.loc[i, "Scheduled Time"] < pd.to_datetime(
                "2020-10-13 08:00:00"
            ) and filtered_data.loc[i, "Scheduled Time"] >= pd.to_datetime(
                "2020-10-13 02:00:00"
            ):
                temps_Terminal = (
                    filtered_data.loc[i, "Scheduled Time"].hour * 60
                    + filtered_data.loc[i, "Scheduled Time"].minute
                    - f_ter_EARLY_inv_linear(y)
                )

            elif filtered_data.loc[i, "Intl Regions"] == "China":
                temps_Terminal = (
                    filtered_data.loc[i, "Scheduled Time"].hour * 60
                    + filtered_data.loc[i, "Scheduled Time"].minute
                    - f_ter_CHINA_inv_linear(y)
                )

            elif filtered_data.loc[i, "Flight Number"][0:2] in airline_code[
                airline_code["FSC / LCC"] == "FSC"
            ]["airline code"].to_numpy(dtype="str"):
                temps_Terminal = (
                    filtered_data.loc[i, "Scheduled Time"].hour * 60
                    + filtered_data.loc[i, "Scheduled Time"].minute
                    - f_ter_LCC_inv_linear(y)
                )

            else:
                temps_Terminal = (
                    filtered_data.loc[i, "Scheduled Time"].hour * 60
                    + filtered_data.loc[i, "Scheduled Time"].minute
                    - f_ter_FSC_inv_linear(y)
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
                list_flights.append(filtered_data.loc[i, "Flight Number"])
                list_ST.append(filtered_data.loc[i, "Scheduled Time"])

    # ====================================== Security =====================================
    # For Security
    if system == "security":
        # import of the excel with the show up profiles
        show_up_sec = pd.read_excel(
            path_show_up,
            sheet_name=r"PRS",
            header=1,
        )
        show_up_sec = show_up_sec.drop([0, 1], axis=0)
        show_up_sec = show_up_sec.reset_index(drop=True)

        # interpolation of show_up profiles and inverse functions
        x = show_up_sec["time before STD"].to_numpy(dtype=float)

        yFSC = show_up_sec["cumulative distribution FSC"].to_numpy(dtype=float)
        f_sec_FSC = interp1d(x, yFSC, kind="linear")

        yLCC = show_up_sec["cumulative distribution LCC"].to_numpy(dtype=float)
        f_sec_LCC = interp1d(x, yLCC, kind="linear")

        yEARLY = show_up_sec["cumulative distribution EARLY"].to_numpy(dtype=float)
        f_sec_EARLY = interp1d(x, yEARLY, kind="linear")

        yCHINA = show_up_sec["cumulative distribution CHINA"].to_numpy(dtype=float)
        f_sec_CHINA = interp1d(x, yEARLY, kind="linear")

        yMORNING = show_up_sec["cumulative distribution MORNING"].to_numpy(dtype=float)
        f_sec_MORNING = interp1d(x, yEARLY, kind="linear")

        f_sec_FSC = interp1d(x, yFSC, kind="linear")
        f_sec_LCC = interp1d(x, yLCC, kind="linear")
        f_sec_EARLY = interp1d(x, yEARLY, kind="linear")
        f_sec_CHINA = interp1d(x, yCHINA, kind="linear")
        f_sec_MORNING = interp1d(x, yMORNING, kind="linear")

        f_sec_FSC_inv_linear = interp1d(f_sec_FSC(x), x, kind="linear")
        f_sec_LCC_inv_linear = interp1d(f_sec_LCC(x), x, kind="linear")
        f_sec_EARLY_inv_linear = interp1d(f_sec_EARLY(x), x, kind="linear")
        f_sec_CHINA_inv_linear = interp1d(f_sec_CHINA(x), x, kind="linear")
        f_sec_MORNING_inv_linear = interp1d(f_sec_MORNING(x), x, kind="linear")

        # let's allocate profiles to flight and apply the ratios to Pax
        list_time_Pax = []
        list_flights = []
        list_ST = []
        for i in range(len(filtered_data)):
            N_flight_pax = int(
                filtered_data.loc[i, "PAX_SUM FC"]
                * ratio
                * (target_peak / schedule_peak)
            )
            STD = filtered_data.loc[i, "Scheduled Time"]
            y = np.linspace(0.0001, 0.995, N_flight_pax)

            if filtered_data.loc[i, "Scheduled Time"] < pd.to_datetime(
                "2020-10-13 08:00:00"
            ) and filtered_data.loc[i, "Scheduled Time"] >= pd.to_datetime(
                "2020-10-13 02:00:00"
            ):
                temps_Security = (
                    filtered_data.loc[i, "Scheduled Time"].hour * 60
                    + filtered_data.loc[i, "Scheduled Time"].minute
                    - f_sec_EARLY_inv_linear(y)
                )

            elif filtered_data.loc[i, "Scheduled Time"] < pd.to_datetime(
                "2020-10-13 12:00:00"
            ) and filtered_data.loc[i, "Scheduled Time"] >= pd.to_datetime(
                "2020-10-13 08:00:00"
            ):
                temps_Security = (
                    filtered_data.loc[i, "Scheduled Time"].hour * 60
                    + filtered_data.loc[i, "Scheduled Time"].minute
                    - f_sec_MORNING_inv_linear(y)
                )

            elif filtered_data.loc[i, "Intl Regions"] == "China":
                temps_Security = (
                    filtered_data.loc[i, "Scheduled Time"].hour * 60
                    + filtered_data.loc[i, "Scheduled Time"].minute
                    - f_sec_CHINA_inv_linear(y)
                )

            elif filtered_data.loc[i, "Flight Number"][0:2] in airline_code[
                airline_code["FSC / LCC"] == "FSC"
            ]["airline code"].to_numpy(dtype="str"):
                temps_Security = (
                    filtered_data.loc[i, "Scheduled Time"].hour * 60
                    + filtered_data.loc[i, "Scheduled Time"].minute
                    - f_sec_LCC_inv_linear(y)
                )

            else:
                temps_Security = (
                    filtered_data.loc[i, "Scheduled Time"].hour * 60
                    + filtered_data.loc[i, "Scheduled Time"].minute
                    - f_sec_FSC_inv_linear(y)
                )

            for t in temps_Security:
                t = datetime.datetime(
                    year=2020,
                    month=10,
                    day=13,
                    hour=int((t % (24 * 60)) / 60),
                    minute=int(t % 60),
                    second=int(t % 1 * 60),
                )
                list_time_Pax.append(t)
                list_flights.append(filtered_data.loc[i, "Flight Number"])
                list_ST.append(filtered_data.loc[i, "Scheduled Time"])

    # ====================================== Call to Gate =====================================
    if system == "CTG":
        # import of the excel with the show up profiles
        show_up_CTG = pd.read_excel(
            path_show_up,
            sheet_name=r"CTG",
            header=1,
        )
        show_up_CTG = show_up_CTG.drop([0, 1, 2], axis=0)
        show_up_CTG = show_up_CTG.reset_index(drop=True)

        # interpolation of CTG profiles for specified type and inverse functions
        x = show_up_CTG["time before STD"].to_numpy(dtype=float)

        y_CTG_C = show_up_CTG[
            "cumulative distribution code C type {}".format(CTG_type)
        ].to_numpy(dtype=float)
        y_CTG_E = show_up_CTG[
            "cumulative distribution code E type {}".format(CTG_type)
        ].to_numpy(dtype=float)
        f_CTG_C = interp1d(x, y_CTG_C, kind="linear")
        f_CTG_E = interp1d(x, y_CTG_E, kind="linear")

        f_CTG_C_inv_linear = interp1d(f_CTG_C(x), x, kind="linear")
        f_CTG_E_inv_linear = interp1d(f_CTG_E(x), x, kind="linear")

        # let's allocate profiles to flight and apply the ratios to Pax
        list_time_Pax = []
        list_flights = []
        list_ST = []
        for i in range(len(filtered_data)):
            N_flight_pax = int(
                filtered_data.loc[i, "PAX_SUM FC"]
                * ratio
                * (target_peak / schedule_peak)
            )
            STD = filtered_data.loc[i, "Scheduled Time"]
            y = np.linspace(0.0001, 0.995, N_flight_pax)

            if filtered_data.loc[i, "Aircraft_Narrow/Wide"] == "Narrow body":
                temps_CTG = (
                    filtered_data.loc[i, "Scheduled Time"].hour * 60
                    + filtered_data.loc[i, "Scheduled Time"].minute
                    - f_CTG_C_inv_linear(y)
                )
            else:
                temps_CTG = (
                    filtered_data.loc[i, "Scheduled Time"].hour * 60
                    + filtered_data.loc[i, "Scheduled Time"].minute
                    - f_CTG_E_inv_linear(y)
                )

            for t in temps_CTG:
                t = datetime.datetime(
                    year=2020,
                    month=10,
                    day=13,
                    hour=int((t % (24 * 60)) / 60),
                    minute=int(t % 60),
                    second=int(t % 1 * 60),
                )
                list_time_Pax.append(t)
                list_flights.append(filtered_data.loc[i, "Flight Number"])
                list_ST.append(filtered_data.loc[i, "Scheduled Time"])

    # ====================================== Boarding =====================================
    if system == "boarding":
        # import of the excel with the show up profiles
        show_up_boarding = pd.read_excel(
            path_show_up,
            sheet_name=r"boarding",
            header=0,
        )
        show_up_boarding = show_up_boarding.reset_index(drop=True)

        # interpolation of boarding profiles for specified type and inverse functions
        x = show_up_boarding["time before STD"].to_numpy(dtype=float)

        y_boarding_C = show_up_boarding["cumulative distribution code C"].to_numpy(
            dtype=float
        )
        y_boarding_E = show_up_boarding["cumulative distribution code E"].to_numpy(
            dtype=float
        )
        f_boarding_C = interp1d(x, y_boarding_C, kind="linear")
        f_boarding_E = interp1d(x, y_boarding_E, kind="linear")

        f_boarding_C_inv_linear = interp1d(
            f_boarding_C(x)[0:10], x[0:10], kind="linear"
        )
        f_boarding_E_inv_linear = interp1d(
            f_boarding_E(x)[0:12], x[0:12], kind="linear"
        )

        # let's allocate profiles to flight and apply the ratios to Pax
        list_time_Pax = []
        list_flights = []
        list_ST = []
        for i in range(len(filtered_data)):
            N_flight_pax = int(
                filtered_data.loc[i, "PAX_SUM FC"]
                * ratio
                * (target_peak / schedule_peak)
            )
            STD = filtered_data.loc[i, "Scheduled Time"]
            y = np.linspace(0.0001, 0.995, N_flight_pax)

            if filtered_data.loc[i, "Aircraft_Narrow/Wide"] == "Narrow body":
                temps_boarding = (
                    filtered_data.loc[i, "Scheduled Time"].hour * 60
                    + filtered_data.loc[i, "Scheduled Time"].minute
                    - f_boarding_C_inv_linear(y)
                )
            else:
                temps_boarding = (
                    filtered_data.loc[i, "Scheduled Time"].hour * 60
                    + filtered_data.loc[i, "Scheduled Time"].minute
                    - f_boarding_E_inv_linear(y)
                )

            for t in temps_boarding:
                t = datetime.datetime(
                    year=2020,
                    month=10,
                    day=13,
                    hour=int((t % (24 * 60)) / 60),
                    minute=int(t % 60),
                    second=int(t % 1 * 60),
                )
                list_time_Pax.append(t)
                list_flights.append(filtered_data.loc[i, "Flight Number"])
                list_ST.append(filtered_data.loc[i, "Scheduled Time"])

    # ====================================== deboarding =====================================
    if system == "arrivals":
        # read the excel with show-up profiles
        show_up_arrival = pd.read_excel(
            path_show_up,
            sheet_name=r"deboarding",
            header=1,
        )

        # interpolate deboarding profiles to use on schedule
        x = show_up_arrival["time after STA"].to_numpy(dtype=float)
        yC = show_up_arrival["cumulative distribution code C"].to_numpy(dtype=float)
        yE = show_up_arrival["cumulative distribution code E"].to_numpy(dtype=float)
        fC = interp1d(x, yC, kind="linear")
        fE = interp1d(x, yE, kind="linear")
        fC_inv_linear = interp1d(fC(x)[0:3], x[0:3], kind="linear")
        fE_inv_linear = interp1d(fE(x)[0:4], x[0:4], kind="linear")

        # let's allocate profiles to flight and apply the ratios to Pax
        list_time_Pax = []
        list_flights = []
        list_ST = []
        for i in range(len(filtered_data)):
            N_flight_pax = int(
                filtered_data.loc[i, "PAX_SUM FC"]
                * ratio
                * (target_peak / schedule_peak)
            )
            STA = filtered_data.loc[i, "Scheduled Time"]
            y = np.linspace(0.0001, 0.995, N_flight_pax)

            if filtered_data.loc[i, "Aircraft_Narrow/Wide"] == "Narrow body":
                temps_deboarding = (
                    filtered_data.loc[i, "Scheduled Time"].hour * 60
                    + filtered_data.loc[i, "Scheduled Time"].minute
                    - fC_inv_linear(y)
                )
            else:
                temps_deboarding = (
                    filtered_data.loc[i, "Scheduled Time"].hour * 60
                    + filtered_data.loc[i, "Scheduled Time"].minute
                    - fE_inv_linear(y)
                )

            for t in temps_deboarding:
                t = datetime.datetime(
                    year=2020,
                    month=10,
                    day=13,
                    hour=int((t % (24 * 60)) / 60),
                    minute=int(t % 60),
                    second=int(t % 1 * 60),
                )
                list_time_Pax.append(t)
                list_flights.append(filtered_data.loc[i, "Flight Number"])
                list_ST.append(filtered_data.loc[i, "Scheduled Time"])

    if system == "check-in":
        return df_Counters_final
    else:
        dct_Pax = {
            "Flight Number": list_flights,
            "time": list_time_Pax,
            "Scheduled Time": list_ST,
        }
        df_Pax = pd.DataFrame(dct_Pax)
        return list_time_Pax, df_Pax

# use the function to generate Pax and counters
def generate_dep_Pax_Counters(
    target_peak=3900,
    terminal="T1", 
    custom_showup=False,
    custom_counter_rule=False,
    **kwargs,
):
    """
    returns df_Pax, df_Counters
    target peak: the value for target peak hour STD (double check the input is actually peak hour STD)
    terminal should be "T1" or "T2" ( T2 is corresponds to 4 AL, incl. TW)
    """
    with tqdm(total=2, desc="Pax and counter generation...") as pbar:
        _, df_Pax = show_up_function(
            target_peak=target_peak,
            direction="D",
            system="terminal",
            ratio=1,
            terminal=terminal,
            CTG_type="A",
            custom_showup=custom_showup,
            **kwargs,
        )
        pbar.update(1)

        df_Counters = show_up_function(
            target_peak=target_peak,
            direction="D",
            system="check-in",
            ratio=1,
            terminal=terminal,
            CTG_type="A",
            custom_showup=False,
            custom_counter_rule=custom_counter_rule,
            **kwargs,
        )
        pbar.update(1)

        if terminal == "T2":
            # apply the special T2 rule for counters
            def no_more_than_10(x):
                if x > 10:
                    x = 10
                return x

            def if_open_10(x):
                if x >= 1:
                    x = 10
                return x
            # specific allocation to T2: no more than 10 counters for each airline
            # and always 10 counters for peach (MM)
            df_Counters = df_Counters.applymap(lambda x: no_more_than_10(x))
            df_Counters["MM"] = df_Counters["MM"].apply(lambda x: if_open_10(x))
            df_Counters["total"] = df_Counters.drop(labels=['total'],axis=1).sum(axis=1)

    return df_Pax, df_Counters

# use the function to generate Pax and counters
def generate_arr_Pax(target_peak=3900, terminal="T1", custom_showup=False, **kwargs):
    """
    returns df_Pax
    target peak: the value for target peak hour STD (double check the input is actually peak hour STA)
    terminal should be "T1" or "T2" ( T2 is corresponds to 4 AL, incl. TW)
    """
    _, df_Pax = show_up_function(
        target_peak=target_peak,
        direction="A",
        system="arrivals",
        ratio=1,
        terminal=terminal,
        CTG_type="A",
        custom_showup=custom_showup,
        **kwargs,
    )

    return df_Pax
