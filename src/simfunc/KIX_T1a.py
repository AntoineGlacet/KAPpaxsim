# KIX_T1a
# no cost function generator yet
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


def KIX_T1a(
    path,
    df_Pax,
    N_quarantine,
    Pt_quarantine,
    N_immigration_counter,
    Pt_immigration_counter,
    N_immigration_self,
    Pt_immigration_self,
    Wt_bag_claim,
    N_customs_counter,
    Pt_customs_counter,
    N_customs_self,
    Pt_customs_self,
    traditional_pax_ratio,
    modern_pax_ratio,
    digital_pax_ratio,
    no_bag_pax_ratio,
    start_special_pax_ratio: float = 0,
    end_special_pax_ratio: float = 1,
    freq="10min",
    win=1,
    show_loading=True,
    show_graph=False,
    save_graph=False,
    save_xls=False,
    call_n_iter=None,
    totalpbar=None,
):
    """
    Function corresponding to one run of the simulation for KIX T1 arr int.
    returns df_result, list_KPI_run
    """
    random.seed(12)  # for reproductibility and smooth optimization

    # change units of Pt

    Pt_quarantine = Pt_quarantine / 60
    Pt_immigration_counter = Pt_immigration_counter / 60
    Pt_immigration_self = Pt_immigration_self / 60
    Pt_customs_counter = Pt_customs_counter / 60
    Pt_customs_self = Pt_customs_self / 60

    # Creating some useful data

    df_Pax["Flight Number"] = df_Pax["Flight Number"].replace(["JX821"], "JX 821")
    df_Pax["Flight Number"] = df_Pax["Flight Number"].replace(["NS*****"], "NS *****")

    df_Pax["airline"] = df_Pax["Flight Number"].apply(lambda x: x.split()[0])
    list_flight = df_Pax["Flight Number"].unique()
    list_airlines = [string for string in df_Pax["airline"].unique()]

    df_Pax["minutes"] = (
        df_Pax["time"].dt.hour.astype(int) * 60
        + df_Pax["time"].dt.minute.astype(int)
        + df_Pax["time"].dt.second.astype(int) / 60
    )
    df_Pax = df_Pax.sort_values(["minutes"]).reset_index(drop=True)

    FREQ = freq
    WINDOW = win

    """
    KIX T1 Int'l arrival
    """

    class arrival_creator(object):
        """
        description
        """

        def __init__(
            self,
            env,
        ):
            self.env = env
            # dummy_machine with infinite capacity
            self.dummy_machine = simpy.Resource(env, 999999999)
            self.dummy_machine2 = simpy.Resource(env, 999999999)

            self.quarantine = simpy.PriorityResource(env, N_quarantine)
            self.Pt_quarantine = Pt_quarantine

            self.immigration_counter = simpy.Resource(env, N_immigration_counter)
            self.Pt_immigration_counter = Pt_immigration_counter

            self.immigration_self = simpy.Resource(env, N_immigration_self)
            self.Pt_immigration_self = Pt_immigration_self

            self.bag_claim = simpy.PriorityResource(env, 999999999)
            self.Wt_bag_claim = Wt_bag_claim

            self.customs_counter = simpy.Resource(env, N_customs_counter)
            self.Pt_customs_counter = Pt_customs_counter

            self.customs_self = simpy.Resource(env, N_customs_self)
            self.Pt_customs_self = Pt_customs_self

        def quarantine_check(self, Pax):
            """quarantine check"""
            yield self.env.timeout(Pt_quarantine)

        def immigration_counter_check(self, Pax):
            """immigration counter check"""
            yield self.env.timeout(Pt_immigration_counter)

        def immigration_self_check(self, Pax):
            """immigration self check"""
            yield self.env.timeout(Pt_immigration_self)

        # to be improved, they should wait for flight STA + some duration + a little random
        # or this could be done with another show up profile
        def bag_claim_wait(self, Pax):
            """bag_claim check"""
            yield self.env.timeout(Wt_bag_claim)

        def customs_counter_check(self, Pax):
            """customs counter check"""
            yield self.env.timeout(Pt_customs_counter)

        def customs_self_check(self, Pax):
            """customs self check"""
            yield self.env.timeout(Pt_customs_self)

    # ======================================= Passenger journey for each type of Pax=======================================

    def Pax_traditional(env, name, arr):
        """
        quarantine => immigration_counter => bag_claim => customs_counter
        """
        df_result.loc[int(name.split("_")[1]), "Pax_ID"] = name
        df_result.loc[int(name.split("_")[1]), "terminal_show_up"] = env.now

        airline_code = name.split("_")[2].split()[0]
        index_airline = list_airlines.index(airline_code)
        index_Pax = int(name.split("_")[1])

        with arr.quarantine.request(priority=2) as request:
            df_result.loc[index_Pax, "quarantine_queue_length"] = len(
                arr.quarantine.queue
            )
            df_result.loc[index_Pax, "start_quarantine_queue"] = env.now
            yield request
            df_result.loc[index_Pax, "end_quarantine_queue"] = env.now
            yield env.process(arr.quarantine_check(name))
            df_result.loc[index_Pax, "end_quarantine_process"] = env.now

        with arr.immigration_counter.request() as request:
            df_result.loc[index_Pax, "immigration_counter_queue_length"] = len(
                arr.immigration_counter.queue
            )
            df_result.loc[index_Pax, "start_immigration_counter_queue"] = env.now
            yield request
            df_result.loc[index_Pax, "end_immigration_counter_queue"] = env.now
            yield env.process(arr.immigration_counter_check(name))
            df_result.loc[index_Pax, "end_immigration_counter_process"] = env.now

        # modified the end_claim_queue for good reporting
        with arr.bag_claim.request() as request:
            df_result.loc[index_Pax, "bag_claim_queue_length"] = arr.bag_claim.count
            df_result.loc[index_Pax, "start_bag_claim_queue"] = env.now
            yield request
            yield env.process(arr.bag_claim_wait(name))
            df_result.loc[index_Pax, "end_bag_claim_queue"] = env.now
            df_result.loc[index_Pax, "end_bag_claim_process"] = env.now

        with arr.customs_counter.request() as request:
            df_result.loc[index_Pax, "customs_counter_queue_length"] = len(
                arr.customs_counter.queue
            )
            df_result.loc[index_Pax, "start_customs_counter_queue"] = env.now
            yield request
            df_result.loc[index_Pax, "end_customs_counter_queue"] = env.now
            yield env.process(arr.customs_counter_check(name))
            df_result.loc[index_Pax, "end_customs_counter_process"] = env.now

    def Pax_modern(env, name, arr):
        """
        quarantine => immigration_self => bag_claim => customs_counter
        """
        df_result.loc[int(name.split("_")[1]), "Pax_ID"] = name
        df_result.loc[int(name.split("_")[1]), "terminal_show_up"] = env.now

        airline_code = name.split("_")[2].split()[0]
        index_airline = list_airlines.index(airline_code)
        index_Pax = int(name.split("_")[1])

        with arr.quarantine.request(priority=2) as request:
            df_result.loc[index_Pax, "quarantine_queue_length"] = len(
                arr.quarantine.queue
            )
            df_result.loc[index_Pax, "start_quarantine_queue"] = env.now
            yield request
            df_result.loc[index_Pax, "end_quarantine_queue"] = env.now
            yield env.process(arr.quarantine_check(name))
            df_result.loc[index_Pax, "end_quarantine_process"] = env.now

        with arr.immigration_self.request() as request:
            df_result.loc[index_Pax, "immigration_self_queue_length"] = len(
                arr.immigration_self.queue
            )
            df_result.loc[index_Pax, "start_immigration_self_queue"] = env.now
            yield request
            df_result.loc[index_Pax, "end_immigration_self_queue"] = env.now
            yield env.process(arr.immigration_self_check(name))
            df_result.loc[index_Pax, "end_immigration_self_process"] = env.now

        # modified the end_claim_queue for good reporting
        with arr.bag_claim.request() as request:
            df_result.loc[index_Pax, "bag_claim_queue_length"] = arr.bag_claim.count
            df_result.loc[index_Pax, "start_bag_claim_queue"] = env.now
            yield request
            yield env.process(arr.bag_claim_wait(name))
            df_result.loc[index_Pax, "end_bag_claim_queue"] = env.now
            df_result.loc[index_Pax, "end_bag_claim_process"] = env.now

        with arr.customs_counter.request() as request:
            df_result.loc[index_Pax, "customs_counter_queue_length"] = len(
                arr.customs_counter.queue
            )
            df_result.loc[index_Pax, "start_customs_counter_queue"] = env.now
            yield request
            df_result.loc[index_Pax, "end_customs_counter_queue"] = env.now
            yield env.process(arr.customs_counter_check(name))
            df_result.loc[index_Pax, "end_customs_counter_process"] = env.now

    def Pax_digital(env, name, arr):
        """
        quarantine => immigration_self => bag_claim => customs_self
        """
        df_result.loc[int(name.split("_")[1]), "Pax_ID"] = name
        df_result.loc[int(name.split("_")[1]), "terminal_show_up"] = env.now

        airline_code = name.split("_")[2].split()[0]
        index_airline = list_airlines.index(airline_code)
        index_Pax = int(name.split("_")[1])

        with arr.quarantine.request(priority=2) as request:
            df_result.loc[index_Pax, "quarantine_queue_length"] = len(
                arr.quarantine.queue
            )
            df_result.loc[index_Pax, "start_quarantine_queue"] = env.now
            yield request
            df_result.loc[index_Pax, "end_quarantine_queue"] = env.now
            yield env.process(arr.quarantine_check(name))
            df_result.loc[index_Pax, "end_quarantine_process"] = env.now

        with arr.immigration_self.request() as request:
            df_result.loc[index_Pax, "immigration_self_queue_length"] = len(
                arr.immigration_self.queue
            )
            df_result.loc[index_Pax, "start_immigration_self_queue"] = env.now
            yield request
            df_result.loc[index_Pax, "end_immigration_self_queue"] = env.now
            yield env.process(arr.immigration_self_check(name))
            df_result.loc[index_Pax, "end_immigration_self_process"] = env.now

        # modified the end_claim_queue for good reporting
        with arr.bag_claim.request() as request:
            df_result.loc[index_Pax, "bag_claim_queue_length"] = arr.bag_claim.count
            df_result.loc[index_Pax, "start_bag_claim_queue"] = env.now
            yield request
            yield env.process(arr.bag_claim_wait(name))
            df_result.loc[index_Pax, "end_bag_claim_queue"] = env.now
            df_result.loc[index_Pax, "end_bag_claim_process"] = env.now

        with arr.customs_self.request() as request:
            df_result.loc[index_Pax, "customs_self_queue_length"] = len(
                arr.customs_self.queue
            )
            df_result.loc[index_Pax, "start_customs_self_queue"] = env.now
            yield request
            df_result.loc[index_Pax, "end_customs_self_queue"] = env.now
            yield env.process(arr.customs_self_check(name))
            df_result.loc[index_Pax, "end_customs_self_process"] = env.now

    def Pax_no_bag(env, name, arr):
        """
        quarantine => immigration_counter => customs_counter
        """
        df_result.loc[int(name.split("_")[1]), "Pax_ID"] = name
        df_result.loc[int(name.split("_")[1]), "terminal_show_up"] = env.now

        airline_code = name.split("_")[2].split()[0]
        index_airline = list_airlines.index(airline_code)
        index_Pax = int(name.split("_")[1])

        with arr.quarantine.request(priority=2) as request:
            df_result.loc[index_Pax, "quarantine_queue_length"] = len(
                arr.quarantine.queue
            )
            df_result.loc[index_Pax, "start_quarantine_queue"] = env.now
            yield request
            df_result.loc[index_Pax, "end_quarantine_queue"] = env.now
            yield env.process(arr.quarantine_check(name))
            df_result.loc[index_Pax, "end_quarantine_process"] = env.now

        with arr.immigration_counter.request() as request:
            df_result.loc[index_Pax, "immigration_counter_queue_length"] = len(
                arr.immigration_counter.queue
            )
            df_result.loc[index_Pax, "start_immigration_counter_queue"] = env.now
            yield request
            df_result.loc[index_Pax, "end_immigration_counter_queue"] = env.now
            yield env.process(arr.immigration_counter_check(name))
            df_result.loc[index_Pax, "end_immigration_counter_process"] = env.now

        with arr.customs_counter.request() as request:
            df_result.loc[index_Pax, "customs_counter_queue_length"] = len(
                arr.customs_counter.queue
            )
            df_result.loc[index_Pax, "start_customs_counter_queue"] = env.now
            yield request
            df_result.loc[index_Pax, "end_customs_counter_queue"] = env.now
            yield env.process(arr.customs_counter_check(name))
            df_result.loc[index_Pax, "end_customs_counter_process"] = env.now

    def setup(env):
        # Create the arrival
        arrival = arrival_creator(env)

    # ======================================= Passenger generator by flight =======================================

    def Pax_generator(env, arrival, flight, df_Pax_flight, index_total):
        """
        create all the Pax types with their ratios
        """

        # Create initial Pax of the flight
        # global index_vol
        index_vol = 0
        index_total = index_total + index_vol
        N_pax_flight = len(df_Pax_flight["minutes"])
        yield env.timeout(df_Pax_flight["minutes"][index_vol])
        env.process(
            Pax_traditional(
                env,
                "pax_{}_{}_traditional".format(index_total, flight),
                arrival,
            )
        )

        # Create the other Paxes
        for index_vol in range(1, N_pax_flight):
            index_total += 1

            yield env.timeout(
                df_Pax_flight["minutes"][index_vol]
                - df_Pax_flight["minutes"][index_vol - 1]
            )
            # generate different types of Pax
            # first, randomly generate the list of index for each type of Pax
            # TO IMPROVE: DO THIS SECTION IN A LOOP OF PAX_TYPES_LIST
            def list_substract(list_1, list_2):
                for element in list_2:
                    if element in list_1:
                        list_1.remove(element)

            start_special_pax_index = int(N_pax_flight * start_special_pax_ratio)
            end_special_pax_index = int(N_pax_flight * end_special_pax_ratio)

            flight_index_list = [i for i in range(1, N_pax_flight)]
            flight_index_list_orig = flight_index_list.copy()
            flight_index_list = [
                i for i in range(start_special_pax_index, end_special_pax_index)
            ]

            random.shuffle(flight_index_list)
            digital_pax_list = flight_index_list[
                0 : int(N_pax_flight * digital_pax_ratio)
            ]
            list_substract(flight_index_list, digital_pax_list)

            random.shuffle(flight_index_list)
            modern_pax_list = flight_index_list[
                0 : int(N_pax_flight * modern_pax_ratio)
            ]
            list_substract(flight_index_list, modern_pax_list)

            random.shuffle(flight_index_list)
            no_bag_pax_list = flight_index_list[
                0 : int(N_pax_flight * no_bag_pax_ratio)
            ]
            list_substract(flight_index_list, no_bag_pax_list)

            # then, generate Pax accordingly
            if index_vol in modern_pax_list:
                env.process(
                    Pax_modern(
                        env,
                        "pax_{}_{}_modern".format(index_total, flight),
                        arrival,
                    )
                )
            elif index_vol in digital_pax_list:
                env.process(
                    Pax_digital(
                        env,
                        "pax_{}_{}_digital".format(index_total, flight),
                        arrival,
                    )
                )
            elif index_vol in no_bag_pax_list:
                env.process(
                    Pax_no_bag(
                        env,
                        "pax_{}_{}_no_bag".format(index_total, flight),
                        arrival,
                    )
                )
            else:
                env.process(
                    Pax_traditional(
                        env,
                        "pax_{}_{}_traditional".format(index_total, flight),
                        arrival,
                    )
                )

    # Create dataframe of results
    dummy_list = [np.nan for i in df_Pax.index]
    list_checkpoints = [
        "Pax_ID",
        "terminal_show_up",
        "quarantine_queue_length",
        "start_quarantine_queue",
        "end_quarantine_queue",
        "end_quarantine_process",
        "immigration_counter_queue_length",
        "start_immigration_counter_queue",
        "end_immigration_counter_queue",
        "end_immigration_counter_process",
        "immigration_self_queue_length",
        "start_immigration_self_queue",
        "end_immigration_self_queue",
        "end_immigration_self_process",
        "bag_claim_queue_length",
        "start_bag_claim_queue",
        "end_bag_claim_queue",
        "end_bag_claim_process",
        "customs_counter_queue_length",
        "start_customs_counter_queue",
        "end_customs_counter_queue",
        "end_customs_counter_process",
        "customs_self_queue_length",
        "start_customs_self_queue",
        "end_customs_self_queue",
        "end_customs_self_process",
    ]

    dct_result = {checkpoint: dummy_list for checkpoint in list_checkpoints}
    df_result = pd.DataFrame(dct_result)

    # Create an environment and start the setup process
    env = simpy.Environment(initial_time=0)
    arrival = arrival_creator(
        env,
    )

    # Generate the Pax

    index_total = 0

    for flight in list_flight:
        # global df_Pax_flight
        df_Pax_flight = (
            df_Pax[df_Pax["Flight Number"] == flight]
            .sort_values(["minutes"])
            .reset_index(drop=True)
        )
        env.process(Pax_generator(env, arrival, flight, df_Pax_flight, index_total))
        index_total += len(df_Pax_flight["minutes"])

    # Execute!
    end_time = 1441

    if show_loading == True:
        if call_n_iter is not None and totalpbar is not None:
            with tqdm(total=end_time - 1, desc="Simulation running...") as runpbar:
                for i in range(1, end_time):
                    env.run(until=i)
                    runpbar.update(1)
                    totalpbar.update(1)
        else:
            with tqdm(total=end_time - 1, desc="Simulation running...") as runpbar:
                for i in range(1, end_time):
                    env.run(until=i)
                    runpbar.update(1)

    else:
        env.run(until=1500)

    # ======================================= Results formatting =======================================

    # Manipulate results dat
    # Change to datetinme
    list_minutes_columns = [
        "terminal_show_up",
        "start_quarantine_queue",
        "end_quarantine_queue",
        "end_quarantine_process",
        "start_immigration_counter_queue",
        "end_immigration_counter_queue",
        "end_immigration_counter_process",
        "start_immigration_self_queue",
        "end_immigration_self_queue",
        "end_immigration_self_process",
        "start_bag_claim_queue",
        "end_bag_claim_queue",
        "end_bag_claim_process",
        "start_customs_counter_queue",
        "end_customs_counter_queue",
        "end_customs_counter_process",
        "start_customs_self_queue",
        "end_customs_self_queue",
        "end_customs_self_process",
    ]

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

    for column in list_minutes_columns:
        df_result[column] = pd.to_datetime(
            df_result[column].apply(lambda x: minutes_to_hms(x))
        )

    # add "Pax_N"
    df_result["Pax_N"] = 1
    # add 'flight_number'
    df_result["flight_number"] = df_result["Pax_ID"].map(lambda x: x.split("_")[2])
    # add "STD" eventually, this may be done inside the simulation as we will use STD
    # to determine who has missed his flight and flag them as such
    df_result = (
        pd.merge(
            df_result,
            df_Pax.drop_duplicates("Flight Number")[
                ["Flight Number", "Scheduled Time"]
            ],
            left_on="flight_number",
            right_on="Flight Number",
            how="left",
        )
        .drop(columns="Flight Number")
        .rename(columns={"Scheduled Time": "STD"})
    )

    # 2 fake guys to prevent bug
    if all(pd.isnull(df_result["start_customs_self_queue"])):
        df_result.loc[0:1, "start_customs_self_queue"] = pd.to_datetime(
            "2020-10-13 00:00:01"
        )
        df_result.loc[0:1, "end_customs_self_queue"] = pd.to_datetime(
            "2020-10-13 00:00:01"
        )
        df_result.loc[0:1, "end_customs_self_process"] = pd.to_datetime(
            "2020-10-13 00:00:01"
        )
        df_result.loc[0:1, "customs_self_queue_length"] = 0

    # Create waiting times
    df_result["wait_time_quarantine"] = (
        df_result["end_quarantine_queue"] - df_result["start_quarantine_queue"]
    ).fillna(datetime.timedelta(0))

    df_result["wait_time_bag_claim"] = (
        df_result["end_bag_claim_queue"] - df_result["start_bag_claim_queue"]
    ).fillna(datetime.timedelta(0))

    df_result["wait_time_customs_counter"] = (
        df_result["end_customs_counter_queue"]
        - df_result["start_customs_counter_queue"]
    ).fillna(datetime.timedelta(0))

    df_result["wait_time_customs_self"] = (
        df_result["end_customs_self_queue"] - df_result["start_customs_self_queue"]
    ).fillna(datetime.timedelta(0))

    df_result["wait_time_immigration_counter"] = (
        df_result["end_immigration_counter_queue"]
        - df_result["start_immigration_counter_queue"]
    ).fillna(datetime.timedelta(0))

    df_result["wait_time_immigration_self"] = (
        df_result["end_immigration_self_queue"]
        - df_result["start_immigration_self_queue"]
    ).fillna(datetime.timedelta(0))

    # dct plot for graphs by list comprehension
    # they correspond to in/out/queue length/wait time
    dct_plot = {
        "quarantine": [
            "start_quarantine_queue",
            "end_quarantine_process",
            "quarantine_queue_length",
            "wait_time_quarantine",
        ],
        "immigration_counter": [
            "start_immigration_counter_queue",
            "end_immigration_counter_process",
            "immigration_counter_queue_length",
            "wait_time_immigration_counter",
        ],
        "immigration_self": [
            "start_immigration_self_queue",
            "end_immigration_self_process",
            "immigration_self_queue_length",
            "wait_time_immigration_self",
        ],
        "bag_claim": [
            "start_bag_claim_queue",
            "end_bag_claim_process",
            "bag_claim_queue_length",
            "wait_time_bag_claim",
        ],
        "customs_counter": [
            "start_customs_counter_queue",
            "end_customs_counter_process",
            "customs_counter_queue_length",
            "wait_time_customs_counter",
        ],
        "customs_self": [
            "start_customs_self_queue",
            "end_customs_self_process",
            "customs_self_queue_length",
            "wait_time_customs_self",
        ],
    }
    # if nobody at customs_self
    if all(pd.isnull(df_result["start_customs_self_queue"])):
        df_result["start_quarantine_queue"] = 0
        df_result["end_quarantine_process"] = 0
        df_result["quarantine_queue_length"] = 0
        df_result["wait_time_quarantine"] = 0

    # correction ratio for resampling with sums
    ratio_sampling = pd.to_timedelta("1H") / pd.to_timedelta(FREQ)

    # in
    plt_in = [
        (
            df_result.set_index(dct_plot[key][0], drop=False)["Pax_N"]
            .resample(FREQ)
            .agg(["sum"])
            .rolling(window=WINDOW, center=True)
            .mean()
            .dropna()
            .apply(lambda x: x * ratio_sampling)
        )
        if not all(
            pd.isnull(df_result.set_index(dct_plot[key][0], drop=False)["Pax_N"].index)
        )
        else (
            df_result.set_index(dct_plot[key][0], drop=False)["Pax_N"]
            .resample(FREQ)
            .agg(["sum"])
            .rolling(window=WINDOW, center=True)
            .mean()
            .dropna()
            .apply(lambda x: x * ratio_sampling)
        )
        for key in [*dct_plot]
    ]

    # out
    plt_out = [
        (
            df_result.set_index(dct_plot[key][1], drop=False)["Pax_N"]
            .resample(FREQ)
            .agg(["sum"])
            .rolling(window=WINDOW, center=True)
            .mean()
            .dropna()
            .apply(lambda x: x * ratio_sampling)
        )
        for key in [*dct_plot]
    ]

    # queue length
    plt_queue_length = [
        (
            df_result.set_index(dct_plot[key][0], drop=False)[dct_plot[key][2]]
            .resample(FREQ)
            .agg(["max"])
            .rolling(window=WINDOW, center=True)
            .mean()
        )
        for key in [*dct_plot]
    ]

    # queue duration
    plt_queue_duration = [
        (
            df_result.set_index(dct_plot[key][0], drop=False)[dct_plot[key][3]]
            .apply(lambda x: x.total_seconds() / 60)
            .resample(FREQ)
            .agg(["max"])
            .rolling(window=WINDOW, center=True)
            .mean()
        )
        for key in [*dct_plot]
    ]
    # histograms of queue duration and queue length

    plt_hist_wait_time = [
        (
            df_result[df_result[dct_plot[key][0]].notnull()][dct_plot[key][3]].apply(
                lambda x: x.total_seconds() / 60
            )
        )
        for key in [*dct_plot]
    ]

    dct_hist_wait_time = {
        key: (
            df_result[df_result[dct_plot[key][0]].notnull()][dct_plot[key][3]].apply(
                lambda x: x.total_seconds() / 60
            )
        )
        for key in [*dct_plot]
    }

    dct_hist_queue_length = {
        key: (df_result[df_result[dct_plot[key][0]].notnull()][dct_plot[key][2]])
        for key in [*dct_plot]
    }

    # ======================================= Plotting =======================================
    n_graph = len([*dct_plot])
    if show_graph == True:
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
            axs[i, 0].plot(plt_in[i], label="in", lw=2)
            axs[i, 0].plot(plt_out[i], label="out", lw=2)
            axs[i, 0].plot(plt_queue_length[i], label="queue length", ls="--", lw=1)
            ax2[i].plot(
                plt_queue_duration[i],
                label="queue duration",
                color="r",
                ls="--",
                lw=1,
            )

            sns.histplot(plt_hist_wait_time[i], ax=axs[i, 1], bins=30)

            axs[i, 0].set(
                ylabel="Pax/hr or Pax",
                title=[*dct_plot][i],
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

        if save_graph == True:
            plt.savefig(path + "/KIX_T1_arr.jpg")

    # set parameters in a DataFrame
    # maybe there is a more elegant way to do that
    dct_param_run = {
        "path": path,
        "N_quarantine": N_quarantine,
        "Pt_quarantine": Pt_quarantine,
        "N_immigration_counter": N_immigration_counter,
        "Pt_immigration_counter": Pt_immigration_counter,
        "N_immigration_self": N_immigration_self,
        "Pt_immigration_self": Pt_immigration_self,
        "Wt_bag_claim": Wt_bag_claim,
        "N_customs_counter": N_customs_counter,
        "Pt_customs_counter": Pt_customs_counter,
        "N_customs_self": N_customs_self,
        "Pt_customs_self": Pt_customs_self,
        "traditional_pax_ratio": traditional_pax_ratio,
        "modern_pax_ratio": modern_pax_ratio,
        "digital_pax_ratio": digital_pax_ratio,
        "no_bag_pax_ratio": no_bag_pax_ratio,
        "start_special_pax_ratio": start_special_pax_ratio,
        "end_special_pax_ratio": end_special_pax_ratio,
        "freq": freq,
        "win": win,
        "show_loading": show_loading,
        "show_graph": show_graph,
        "save_graph": save_graph,
        "save_xls": save_xls,
        "call_n_iter": call_n_iter,
    }
    df_param_run = pd.DataFrame(dct_param_run, index=[0])
    if show_graph == True:
        print(df_param_run)

    if save_xls == True:
        # write set results to Excel
        writer = pd.ExcelWriter(
            path + r"\run_results.xlsx",
            engine="xlsxwriter",
        )
        df_result.to_excel(writer, sheet_name="results")
        df_param_run.transpose().to_excel(writer, sheet_name="parameters")
        df_Pax.to_excel(writer, sheet_name="Pax_input")

        writer.save()

    list_kpi_queue_length = [
        list(plt_queue_length[i]["max"].replace(np.nan, 0)) for i in range(n_graph)
    ]
    list_kpi_wait_time = [list(plt_hist_wait_time[i]) for i in range(n_graph)]

    kpi_queue_length = [
        min(
            heapq.nlargest(
                max(int(len(list_kpi_queue_length[i]) / 99), 1),
                list_kpi_queue_length[i],
            )
        )
        for i in range(n_graph)
    ]
    kpi_wait_time = [
        min(
            heapq.nlargest(
                max(int(len(list_kpi_wait_time[i]) / 99), 1),
                list_kpi_wait_time[i],
            )
        )
        for i in range(n_graph)
    ]

    list_KPI_run = [[kpi_queue_length[i], kpi_wait_time[i]] for i in range(n_graph)]

    return (
        df_result,
        list_KPI_run,
        dct_hist_wait_time,
        dct_hist_queue_length,
    )
