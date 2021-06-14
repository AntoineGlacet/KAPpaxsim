# KIX_T1d.py
# includes: 
# - KIX_T1_departure_sim_function
# - univariate_cost_function_generator_t1d

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

def KIX_T1_departure_sim_function(
    path,
    df_Pax,
    df_Counters,
    Pt_checkin_1step_counter=150 / 60,
    Pt_checkin_2step_counter=90 / 60,
    N_kiosk=92,
    Pt_kiosk=90 / 60,
    N_security_lanes=16,
    Pt_security_lanes=14 / 60,
    N_emigration_counter=20,
    Pt_emigration_counter=23 / 60,
    N_emigration_self=23,
    Pt_emigration_self=28 / 60,
    modern_pax_ratio=0.1,
    digital_pax_ratio=0.05,
    premium_pax_ratio=0.05,
    start_special_pax_ratio=0.3,
    end_special_pax_ratio=0.95,
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
    Function corresponding to one run of the simulation for KIX T1 dep int.
    returns  df_result,
            list_KPI_run,
            dct_hist_wait_time,
            dct_hist_queue_length,
    """
    random.seed(12) # for reproductibility and smooth optimization

    # change units of Pt
    Pt_checkin_1step_counter = Pt_checkin_1step_counter / 60
    Pt_checkin_2step_counter = Pt_checkin_2step_counter / 60
    Pt_kiosk = Pt_kiosk / 60
    Pt_security_lanes = Pt_security_lanes / 60
    Pt_emigration_counter = Pt_emigration_counter / 60
    Pt_emigration_self = Pt_emigration_self / 60

    # Creating some useful data
    traditionnal_pax_ratio = (
        1 - modern_pax_ratio - digital_pax_ratio - premium_pax_ratio
    )

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

    data_orig = df_Counters.copy()
    data = data_orig.drop(columns=["total"])

    data = data.fillna(1e-12)
    data = data.replace(0, 1e-12)

    FREQ = freq
    WINDOW = win

    """
    KIX T1 Int'l departure
    """

    class departure_creator(object):
        """
        description
        """

        def __init__(
            self,
            env,
            list_airlines,
            Pt_checkin_1step_counter,
            N_kiosk,
            Pt_kiosk,
            N_security_lanes,
            Pt_security_lanes,
            N_emigration_counter,
            Pt_emigration_counter,
            N_emigration_self,
            Pt_emigration_self,
        ):
            self.env = env
            # dummy_machine with infinite capacity
            self.dummy_machine = simpy.Resource(env, 9999999999999999999999999)
            self.dummy_machine2 = simpy.Resource(env, 9999999999999999999999999)

            # define one check-in entity per airline we will adapt processing time to represent the number of counters opened
            self.checkin = [
                simpy.PriorityResource(env, 1) for i in range(len(list_airlines))
            ]
            self.Pt_checkin_1step_counter = Pt_checkin_1step_counter

            self.kiosk = simpy.PriorityResource(env, N_kiosk)
            self.Pt_kiosk = Pt_kiosk

            self.security_lanes = simpy.PriorityResource(env, N_security_lanes)
            self.Pt_security_lanes = Pt_security_lanes

            self.emigration_counter = simpy.Resource(env, N_emigration_counter)
            self.Pt_emigration_counter = Pt_emigration_counter

            self.emigration_self = simpy.Resource(env, N_emigration_self)
            self.Pt_emigration_self = Pt_emigration_self

        def wait_opening(self, Pax):
            """wait for an openned counter"""
            opened_counters = data.loc[
                int(env.now / 5) % 288, Pax.split("_")[2].split()[0]
            ]
            while opened_counters < 1:
                yield self.env.timeout(5)
                opened_counters = data.loc[
                    int(env.now / 5) % 288, Pax.split("_")[2].split()[0]
                ]

        def checkin_1step_counter(self, Pax):
            """The check-in process
            if the counter are closed by the time the Pax ends queueing,
            we should flag him as 'missed flight at check-in'
            right now we just let them wait until reopening."""
            opened_counters = data.loc[
                int(env.now / 5) % 288, Pax.split("_")[2].split()[0]
            ]
            while opened_counters < 1:
                yield self.env.timeout(5)
                opened_counters = data.loc[
                    int(env.now / 5) % 288, Pax.split("_")[2].split()[0]
                ]
            test_time = Pt_checkin_1step_counter / opened_counters
            yield self.env.timeout(test_time)

        def checkin_1step_dummy(self, Pax):
            """dummy process to have the good processing time for each checkin operation"""
            opened_counters = data.loc[
                int(env.now / 5) % 288, Pax.split("_")[2].split()[0]
            ]
            test_time2 = Pt_checkin_1step_counter - (
                Pt_checkin_1step_counter / opened_counters
            )
            if test_time2 < 0:
                test_time2 = 0.00001
            yield self.env.timeout(test_time2)

        def checkin_2step_counter(self, Pax):
            """same as 1-step process but shorter"""
            opened_counters = data.loc[
                int(env.now / 5) % 288, Pax.split("_")[2].split()[0]
            ]
            while opened_counters < 1:
                yield self.env.timeout(5)
                opened_counters = data.loc[
                    int(env.now / 5) % 288, Pax.split("_")[2].split()[0]
                ]
            test_time = Pt_checkin_2step_counter / opened_counters
            yield self.env.timeout(test_time)

        def checkin_2step_dummy(self, Pax):
            """dummy process to have the good processing time for each checkin operation"""
            opened_counters = data.loc[
                int(env.now / 5) % 288, Pax.split("_")[2].split()[0]
            ]
            test_time3 = Pt_checkin_1step_counter - (
                Pt_checkin_2step_counter / opened_counters
            )
            if test_time3 < 0:
                test_time3 = 0.00001
            yield self.env.timeout(test_time3)

        def checkin_2step_kiosk(self, Pax):
            """ check-in at Kiosk """
            yield self.env.timeout(Pt_kiosk)

        def security_screening(self, Pax):
            """ security screening """
            yield self.env.timeout(Pt_security_lanes)

        def emigration_counter_check(self, Pax):
            """ emigration counter check """
            yield self.env.timeout(Pt_emigration_counter)

        def emigration_self_check(self, Pax):
            """ emigration self check """
            yield self.env.timeout(Pt_emigration_self)

    # ======================================= Passenger journey for each type of Pax=======================================

    def Pax_traditional(env, name, dep):
        """description"""
        df_result.loc[int(name.split("_")[1]), "Pax_ID"] = name
        df_result.loc[int(name.split("_")[1]), "terminal_show_up"] = env.now

        airline_code = name.split("_")[2].split()[0]
        index_airline = list_airlines.index(airline_code)
        index_Pax = int(name.split("_")[1])

        # wait if counter is closed
        with dep.dummy_machine2.request() as request:
            df_result.loc[
                index_Pax, "n_people_waiting_for_counter_opening"
            ] = dep.dummy_machine2.count
            df_result.loc[index_Pax, "start_wait_for_counter_opening"] = env.now
            yield env.process(dep.wait_opening(name))
            df_result.loc[index_Pax, "start_checkin_counter_queue"] = env.now

        with dep.checkin[index_airline].request(priority=2) as request:
            df_result.loc[index_Pax, "checkin_counter_queue_length"] = sum(
                [len(dep.checkin[i].queue) for i in range(len(list_airlines))]
            )
            yield request
            df_result.loc[index_Pax, "end_checkin_counter_queue"] = env.now
            yield env.process(dep.checkin_1step_counter(name))

        with dep.dummy_machine.request() as request:
            yield request
            yield env.process(dep.checkin_1step_dummy(name))
            df_result.loc[index_Pax, "end_checkin_counter_process"] = env.now

        with dep.security_lanes.request(priority=2) as request:
            df_result.loc[index_Pax, "security_queue_length"] = len(
                dep.security_lanes.queue
            )
            df_result.loc[index_Pax, "start_security_queue"] = env.now
            yield request
            df_result.loc[index_Pax, "end_security_queue"] = env.now
            yield env.process(dep.security_screening(name))
            df_result.loc[index_Pax, "end_security_process"] = env.now

        with dep.emigration_counter.request() as request:
            df_result.loc[index_Pax, "emigration_counter_queue_length"] = len(
                dep.emigration_counter.queue
            )
            df_result.loc[index_Pax, "start_emigration_counter_queue"] = env.now
            yield request
            df_result.loc[index_Pax, "end_emigration_counter_queue"] = env.now
            yield env.process(dep.emigration_counter_check(name))
            df_result.loc[index_Pax, "end_emigration_counter_process"] = env.now

    def Pax_modern(env, name, dep):
        """description"""
        df_result.loc[int(name.split("_")[1]), "Pax_ID"] = name
        df_result.loc[int(name.split("_")[1]), "terminal_show_up"] = env.now

        airline_code = name.split("_")[2].split()[0]
        index_airline = list_airlines.index(airline_code)
        index_Pax = int(name.split("_")[1])

        with dep.kiosk.request() as request:
            df_result.loc[index_Pax, "checkin_kiosk_queue_length"] = len(
                dep.kiosk.queue
            )
            df_result.loc[index_Pax, "start_checkin_kiosk_queue"] = env.now
            yield request
            df_result.loc[index_Pax, "end_checkin_kiosk_queue"] = env.now
            yield env.process(dep.checkin_2step_kiosk(name))
            df_result.loc[index_Pax, "end_checkin_kiosk_process"] = env.now

        # wait if counter is closed
        with dep.dummy_machine2.request() as request:
            df_result.loc[
                index_Pax, "n_people_waiting_for_counter_opening"
            ] = dep.dummy_machine2.count
            df_result.loc[index_Pax, "start_wait_for_counter_opening"] = env.now
            yield env.process(dep.wait_opening(name))
            df_result.loc[index_Pax, "start_checkin_counter_queue"] = env.now

        with dep.checkin[index_airline].request(priority=2) as request:
            df_result.loc[index_Pax, "checkin_counter_queue_length"] = sum(
                [len(dep.checkin[i].queue) for i in range(len(list_airlines))]
            )
            yield request
            df_result.loc[index_Pax, "end_checkin_counter_queue"] = env.now
            yield env.process(dep.checkin_2step_counter(name))

        with dep.dummy_machine.request() as request:
            yield request
            yield env.process(dep.checkin_1step_dummy(name))
            df_result.loc[index_Pax, "end_checkin_counter_process"] = env.now

        with dep.security_lanes.request(priority=2) as request:
            df_result.loc[index_Pax, "security_queue_length"] = len(
                dep.security_lanes.queue
            )
            df_result.loc[index_Pax, "start_security_queue"] = env.now
            yield request
            df_result.loc[index_Pax, "end_security_queue"] = env.now
            yield env.process(dep.security_screening(name))
            df_result.loc[index_Pax, "end_security_process"] = env.now

        with dep.emigration_self.request() as request:
            df_result.loc[index_Pax, "emigration_self_queue_length"] = len(
                dep.emigration_self.queue
            )
            df_result.loc[index_Pax, "start_emigration_self_queue"] = env.now
            yield request
            df_result.loc[index_Pax, "end_emigration_self_queue"] = env.now
            yield env.process(dep.emigration_self_check(name))
            df_result.loc[index_Pax, "end_emigration_self_process"] = env.now

    def Pax_digital(env, name, dep):
        """description"""
        df_result.loc[int(name.split("_")[1]), "Pax_ID"] = name
        df_result.loc[int(name.split("_")[1]), "terminal_show_up"] = env.now

        airline_code = name.split("_")[2].split()[0]
        index_airline = list_airlines.index(airline_code)
        index_Pax = int(name.split("_")[1])

        with dep.kiosk.request() as request:
            df_result.loc[index_Pax, "checkin_kiosk_queue_length"] = len(
                dep.kiosk.queue
            )
            df_result.loc[index_Pax, "start_checkin_kiosk_queue"] = env.now
            yield request
            df_result.loc[index_Pax, "end_checkin_kiosk_queue"] = env.now
            yield env.process(dep.checkin_2step_kiosk(name))
            df_result.loc[index_Pax, "end_checkin_kiosk_process"] = env.now

        # wait if counter is closed
        with dep.dummy_machine2.request() as request:
            df_result.loc[
                index_Pax, "n_people_waiting_for_counter_opening"
            ] = dep.dummy_machine2.count
            df_result.loc[index_Pax, "start_wait_for_counter_opening"] = env.now
            yield env.process(dep.wait_opening(name))
            df_result.loc[index_Pax, "start_checkin_counter_queue"] = env.now

        with dep.checkin[index_airline].request(priority=2) as request:
            df_result.loc[index_Pax, "checkin_counter_queue_length"] = sum(
                [len(dep.checkin[i].queue) for i in range(len(list_airlines))]
            )
            yield request
            df_result.loc[index_Pax, "end_checkin_counter_queue"] = env.now
            yield env.process(dep.checkin_2step_counter(name))

        with dep.dummy_machine.request() as request:
            yield request
            yield env.process(dep.checkin_1step_dummy(name))
            df_result.loc[index_Pax, "end_checkin_counter_process"] = env.now

        with dep.security_lanes.request(priority=2) as request:
            df_result.loc[index_Pax, "security_queue_length"] = len(
                dep.security_lanes.queue
            )
            df_result.loc[index_Pax, "start_security_queue"] = env.now
            yield request
            df_result.loc[index_Pax, "end_security_queue"] = env.now
            yield env.process(dep.security_screening(name))
            df_result.loc[index_Pax, "end_security_process"] = env.now

        with dep.emigration_self.request() as request:
            df_result.loc[index_Pax, "emigration_self_queue_length"] = len(
                dep.emigration_self.queue
            )
            df_result.loc[index_Pax, "start_emigration_self_queue"] = env.now
            yield request
            df_result.loc[index_Pax, "end_emigration_self_queue"] = env.now
            yield env.process(dep.emigration_self_check(name))
            df_result.loc[index_Pax, "end_emigration_self_process"] = env.now

    def Pax_premium(env, name, dep):
        """description"""
        df_result.loc[int(name.split("_")[1]), "Pax_ID"] = name
        df_result.loc[int(name.split("_")[1]), "terminal_show_up"] = env.now

        airline_code = name.split("_")[2].split()[0]
        index_airline = list_airlines.index(airline_code)
        index_Pax = int(name.split("_")[1])

        # wait if counter is closed
        with dep.dummy_machine2.request() as request:
            df_result.loc[
                index_Pax, "n_people_waiting_for_counter_opening"
            ] = dep.dummy_machine2.count
            df_result.loc[index_Pax, "start_wait_for_counter_opening"] = env.now
            yield env.process(dep.wait_opening(name))
            df_result.loc[index_Pax, "start_checkin_counter_queue"] = env.now

        with dep.checkin[index_airline].request(priority=1) as request:
            df_result.loc[index_Pax, "checkin_counter_queue_length"] = sum(
                [len(dep.checkin[i].queue) for i in range(len(list_airlines))]
            )
            yield request
            df_result.loc[index_Pax, "end_checkin_counter_queue"] = env.now
            yield env.process(dep.checkin_1step_counter(name))

        with dep.dummy_machine.request() as request:
            yield request
            yield env.process(dep.checkin_1step_dummy(name))
            df_result.loc[index_Pax, "end_checkin_counter_process"] = env.now

        with dep.security_lanes.request(priority=1) as request:
            df_result.loc[index_Pax, "security_queue_length"] = len(
                dep.security_lanes.queue
            )
            df_result.loc[index_Pax, "start_security_queue"] = env.now
            yield request
            df_result.loc[index_Pax, "end_security_queue"] = env.now
            yield env.process(dep.security_screening(name))
            df_result.loc[index_Pax, "end_security_process"] = env.now

        with dep.emigration_counter.request() as request:
            df_result.loc[index_Pax, "emigration_counter_queue_length"] = len(
                dep.emigration_counter.queue
            )
            df_result.loc[index_Pax, "start_emigration_counter_queue"] = env.now
            yield request
            df_result.loc[index_Pax, "end_emigration_counter_queue"] = env.now
            yield env.process(dep.emigration_counter_check(name))
            df_result.loc[index_Pax, "end_emigration_counter_process"] = env.now

    def setup(env, Pt_checkin_1step_counter):
        """Create a departure, a number of initial Paxs and keep creating Paxs
        approx. every ``t_inter`` minutes."""
        # Create the departure
        departure = departure_creator(env, Pt_checkin_1step_counter)

    # ======================================= Passenger generator by flight =======================================

    def Pax_generator(env, departure, flight, df_Pax_flight, index_total):

        # Create initial Pax of the flight
        # global index_vol
        index_vol = 0
        index_total = index_total + index_vol
        N_pax_flight = len(df_Pax_flight["minutes"])
        yield env.timeout(df_Pax_flight["minutes"][index_vol])
        env.process(
            Pax_traditional(
                env, "pax_{}_{}_traditional".format(index_total, flight), departure
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
            premium_pax_list = flight_index_list[
                0 : int(N_pax_flight * premium_pax_ratio)
            ]
            list_substract(flight_index_list, premium_pax_list)

            # then, generate Pax accordingly
            if index_vol in modern_pax_list:
                env.process(
                    Pax_modern(
                        env,
                        "pax_{}_{}_modern".format(index_total, flight),
                        departure,
                    )
                )
            elif index_vol in digital_pax_list:
                env.process(
                    Pax_digital(
                        env,
                        "pax_{}_{}_digital".format(index_total, flight),
                        departure,
                    )
                )
            elif index_vol in premium_pax_list:
                env.process(
                    Pax_premium(
                        env,
                        "pax_{}_{}_premium".format(index_total, flight),
                        departure,
                    )
                )
            else:
                env.process(
                    Pax_traditional(
                        env,
                        "pax_{}_{}_traditional".format(index_total, flight),
                        departure,
                    )
                )
                
    # Create dataframe of results
    dummy_list = [np.nan for i in df_Pax.index]
    list_checkpoints = [
        "Pax_ID",
        "terminal_show_up",
        "start_wait_for_counter_opening",
        "n_people_waiting_for_counter_opening",
        "checkin_kiosk_queue_length",
        "start_checkin_kiosk_queue",
        "end_checkin_kiosk_queue",
        "end_checkin_kiosk_process",
        "checkin_counter_queue_length",
        "start_checkin_counter_queue",
        "end_checkin_counter_queue",
        "end_checkin_counter_process",
        "security_queue_length",
        "start_security_queue",
        "end_security_queue",
        "end_security_process",
        "emigration_counter_queue_length",
        "start_emigration_counter_queue",
        "end_emigration_counter_queue",
        "end_emigration_counter_process",
        "emigration_self_queue_length",
        "start_emigration_self_queue",
        "end_emigration_self_queue",
        "end_emigration_self_process",
    ]

    dct_result = {checkpoint: dummy_list for checkpoint in list_checkpoints}
    df_result = pd.DataFrame(dct_result)

    # Create an environment and start the setup process
    env = simpy.Environment(initial_time=0)
    departure = departure_creator(
        env,
        list_airlines,
        Pt_checkin_1step_counter,
        N_kiosk,
        Pt_kiosk,
        N_security_lanes,
        Pt_security_lanes,
        N_emigration_counter,
        Pt_emigration_counter,
        N_emigration_self,
        Pt_emigration_self,
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
        env.process(Pax_generator(env, departure, flight, df_Pax_flight, index_total))
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
        "start_checkin_kiosk_queue",
        "end_checkin_kiosk_queue",
        "end_checkin_kiosk_process",
        "start_wait_for_counter_opening",
        "start_checkin_counter_queue",
        "end_checkin_counter_queue",
        "end_checkin_counter_process",
        "start_security_queue",
        "end_security_queue",
        "end_security_process",
        "start_emigration_counter_queue",
        "end_emigration_counter_queue",
        "end_emigration_counter_process",
        "start_emigration_self_queue",
        "end_emigration_self_queue",
        "end_emigration_self_process",
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

    # Create waiting times
    df_result["wait_time_checkin_kiosk"] = (
        df_result["end_checkin_kiosk_queue"] - df_result["start_checkin_kiosk_queue"]
    ).fillna(datetime.timedelta(0))

    df_result["wait_time_checkin_counter_opening"] = (
        df_result["start_checkin_counter_queue"]
        - df_result["start_wait_for_counter_opening"]
    ).fillna(datetime.timedelta(0))

    df_result["wait_time_checkin_counter"] = (
        df_result["end_checkin_counter_queue"]
        - df_result["start_checkin_counter_queue"]
    ).fillna(datetime.timedelta(0))

    df_result["wait_time_security"] = (
        df_result["end_security_queue"] - df_result["start_security_queue"]
    ).fillna(datetime.timedelta(0))

    df_result["wait_time_emigration_counter"] = (
        df_result["end_emigration_counter_queue"]
        - df_result["start_emigration_counter_queue"]
    ).fillna(datetime.timedelta(0))

    df_result["wait_time_emigration_self"] = (
        df_result["end_emigration_self_queue"]
        - df_result["start_emigration_self_queue"]
    ).fillna(datetime.timedelta(0))
    
    # for process with start queue but no end queue, set waiting time at 8hrs
    # actually, the queue did not end during sim time so we set the result as high
    for process in ['checkin_kiosk','security','emigration_counter','emigration_self']:
        mask = (pd.isna(df_result["end_{}_queue".format(process)])) & (
            pd.notna(df_result["start_{}_queue".format(process)])
        )
        
        df_result.loc[mask, "wait_time_{}".format(process)] = datetime.timedelta(hours=14)  

    # dct plot for graphs by list comprehension
    # they correspond to in/out/queue length/wait time
    dct_plot = {
        "kiosk": [
            "start_checkin_kiosk_queue",
            "end_checkin_kiosk_process",
            "checkin_kiosk_queue_length",
            "wait_time_checkin_kiosk",
        ],
        "wait_counter_opening": [
            "start_wait_for_counter_opening",
            "start_checkin_counter_queue",
            "n_people_waiting_for_counter_opening",
            "wait_time_checkin_counter_opening",
        ],
        "checkin_counter": [
            "start_checkin_counter_queue",
            "end_checkin_counter_process",
            "checkin_counter_queue_length",
            "wait_time_checkin_counter",
        ],
        "security_lanes": [
            "start_security_queue",
            "end_security_process",
            "security_queue_length",
            "wait_time_security",
        ],
        "emigration_counter": [
            "start_emigration_counter_queue",
            "end_emigration_counter_process",
            "emigration_counter_queue_length",
            "wait_time_emigration_counter",
        ],
        "emigration_self": [
            "start_emigration_self_queue",
            "end_emigration_self_process",
            "emigration_self_queue_length",
            "wait_time_emigration_self",
        ],
    }

    # Plot des counters
    data_orig["time"] = data_orig.index
    data_orig["time"] = data_orig["time"].apply(lambda x: minutes_to_hms(5 * x))
    data_orig["time"] = pd.to_datetime(data_orig["time"])
    plot_counter = data_orig.set_index("time").resample("60S").ffill()[["total"]]

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
        for i in [0, 2, 3, 4, 5]:
            axs[i, 0].plot(plt_in[i], label="in", lw=2)
            axs[i, 0].plot(plt_out[i], label="out", lw=2)
            axs[i, 0].plot(plt_queue_length[i], label="queue length", ls="--", lw=1)
            ax2[i].plot(
                plt_queue_duration[i], label="queue duration", color="r", ls="--", lw=1
            )

            sns.histplot(plt_hist_wait_time[i], ax=axs[i, 1], bins=30)

            axs[i, 0].set(
                ylabel="Pax/hr or Pax", title=[*dct_plot][i], xlim=[xmin, xmax]
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

        # plot for 'wait for counter opening'
        i = 1
        axs[i, 0].plot(plt_in[i], label="in", lw=2)
        axs[i, 0].plot(plt_out[i], label="out", ls="--", lw=1)
        axs[i, 0].plot(plt_queue_length[i], label="queue length", ls="--", lw=1)
        # ax2[i].plot(plt_queue_duration[i], label="queue duration", color="r",ls='--',lw=1)
        ax2[1].plot(plot_counter, label="Counters", lw=2, color="r")

        sns.histplot(plt_hist_wait_time[i], ax=axs[i, 1], bins=30)

        axs[i, 0].set(ylabel="Pax/hr or Pax", title=[*dct_plot][i], xlim=[xmin, xmax])
        axs[i, 0].set_ylim(bottom=0)
        axs[i, 0].xaxis.set_major_locator(hours)
        axs[i, 0].xaxis.set_major_formatter(h_fmt)
        axs[i, 0].xaxis.set_minor_locator(half_hours)
        axs[i, 0].legend(loc="upper left")

        axs[i, 1].set_xlim(left=0)

        ax2[i].legend(loc="upper right")
        ax2[i].set(ylabel="counters opened [unit]")
        ax2[i].set_ylim(bottom=0)
        ax2[i].spines["right"].set_color("r")
        ax2[i].tick_params(axis="y", colors="r")
        ax2[i].yaxis.label.set_color("r")

        # remove ticks labels for all grows except last
        for i in range(n_graph - 1):
            ax2[i].tick_params(
                axis="x", which="both", bottom=False, top=False, labelbottom=False
            )
            axs[i, 0].tick_params(
                axis="x", which="both", bottom=False, top=False, labelbottom=False
            )

        # format last row graphs xticks labels
        labels = axs[n_graph - 1, 0].get_xticklabels()
        plt.setp(labels, rotation=45, horizontalalignment="right")
        axs[n_graph - 1, 0].set(xlabel="time")

        if save_graph == True:
            plt.savefig(path + "/KIX_T1_dep.jpg")

    # set parameters in a DataFrame
    # maybe there is a more elegant way to do that
    dct_param_run = {
        "path": path,
        "Pt_checkin_1step_counter": Pt_checkin_1step_counter,
        "Pt_checkin_2step_counter": Pt_checkin_2step_counter,
        "N_kiosk": N_kiosk,
        "Pt_kiosk": Pt_kiosk,
        "N_security_lanes": N_security_lanes,
        "Pt_security_lanes": Pt_security_lanes,
        "N_emigration_counter": N_emigration_counter,
        "Pt_emigration_counter": Pt_emigration_counter,
        "N_emigration_self": N_emigration_self,
        "Pt_emigration_self": Pt_emigration_self,
        "modern_pax_ratio": modern_pax_ratio,
        "digital_pax_ratio": digital_pax_ratio,
        "premium_pax_ratio": premium_pax_ratio,
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
        df_Counters.to_excel(writer, sheet_name="Counters_input")

        writer.save()

    list_kpi_queue_length = [
        list(plt_queue_length[i]["max"].replace(np.nan, 0)) for i in range(n_graph)
    ]
    list_kpi_wait_time = [list(plt_hist_wait_time[i]) for i in range(n_graph)]

    kpi_queue_length = [
        min(
            heapq.nlargest(
                int(len(list_kpi_queue_length[i]) / 99), list_kpi_queue_length[i]
            )
        )
        for i in range(n_graph)
    ]
    kpi_wait_time = [
        min(heapq.nlargest(int(len(list_kpi_wait_time[i]) / 99), list_kpi_wait_time[i]))
        for i in range(n_graph)
    ]

    list_KPI_run = [[kpi_queue_length[i], kpi_wait_time[i]] for i in range(n_graph)]

    return (
        df_result,
        list_KPI_run,
        dct_hist_wait_time,
        dct_hist_queue_length,
    )

def univariate_cost_function_generator_t1d_N(
    variable_string,  # eg. N_kiosk
    target_wait_time,  # target waiting time in minutes for each system, same for all system for now
    dct_param_T1d,  # includes df_Pax and df_Counters
    call_n_iter=None,
    totalpbar=None,
):
    """
    this function generates a univariate cost function for T1 departure

    the variable must be selected: which parameter should the function optimize
    => for now we will do the N only
    => after we should also make the following possible:
        - traffic (easy)
        - areas
    """

    def cost_function_T1d_covid_N(
        x,  # value of variable for which cost function should be evaluated
        variable_string=variable_string,  # variable that should be optimized, eg. N_Z
        target_wait_time=target_wait_time,  # target waiting time in minutes
        dct_param_T1d=dct_param_T1d,
    ):
        """
        calculate the cost function for KIX T1 departure
        """
        # regenerate the system name corresponding to dct_hist_wait_time
        # dct_hist_wait_time keys are:
        # 'kiosk', 'wait_counter_opening', 'checkin_counter',
        # 'security_lanes', 'emigration_counter', 'emigration_self'
        # note: counters are determined by allocation rule so it's not optimizable
        system_string = variable_string.split("_")[1]

        # pass the variable for the parameter to be optimized
        dct_param_T1d[variable_string] = x

        # run the model and get the wait_time and queue_length dicts
        (
            _,
            _,
            dct_hist_wait_time,
            dct_hist_queue_length,
        ) = KIX_T1_departure_sim_function(**dct_param_T1d)

        # caculate cost for specific variable
        cost_wait_time_run = (
            dct_hist_wait_time[system_string].quantile(q=0.90) - target_wait_time
        ) ** 2

        # correct if:

        # if top90% pax do not wait, then we need to penalize high number of N
        if dct_hist_wait_time[system_string].quantile(q=0.90) == 0:
            cost_wait_time_run += x / 10000

        # if the top90% Pax waits 8hrs or more, we need to consider the mean waiting time
        if dct_hist_wait_time[system_string].quantile(q=0.90) >= 13.9 * 60:
            cost_wait_time_run += (
                (dct_hist_wait_time[system_string].mean() - target_wait_time) ** 2
            ) / 10000

        return cost_wait_time_run

    return cost_function_T1d_covid_N