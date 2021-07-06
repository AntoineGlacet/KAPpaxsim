import pandas as pd
import numpy as np


def calculate_EBS_LBC(
    df_result,
    MUP_open_time=pd.Timedelta(hours=2, minutes=10),
    MUP_close_time=pd.Timedelta(hours=1),
):
    # change the times after midnight to the next day to calculate properly
    end = pd.Timestamp("2020-10-13 02:00:00")
    mask_late_flight = df_result["STD"] < end
    df_result.loc[mask_late_flight, "STD"] += pd.Timedelta(days=1)

    # mask for bags who will use EBS
    # we take start security queue because some come from checkin and some from CUSBD
    # we should also remove no bag pax (TBD)

    mask_EBS = (
        df_result["start_security_queue"] < df_result["STD"] - MUP_open_time
    )
    df_result.loc[mask_EBS, "EBS_in"] = df_result.loc[
        mask_EBS, "start_security_queue"
    ]
    df_result.loc[mask_EBS, "EBS_out"] = (
        df_result.loc[mask_EBS, "STD"] - MUP_open_time
    )

    plt_in_EBS = (
        df_result.loc[mask_EBS, ["EBS_in", "Pax_N"]]
        .set_index("EBS_in", drop=False)["Pax_N"]
        .resample("15min")
        .agg(["sum"])
        .cumsum()
    )

    plt_out_EBS = (
        df_result.loc[mask_EBS, ["EBS_out", "Pax_N"]]
        .set_index("EBS_out", drop=False)["Pax_N"]
        .resample("15min")
        .agg(["sum"])
        .cumsum()
    )

    EBS_req = (plt_in_EBS - plt_out_EBS).max()[0]

    # mask for bags who will use LBC

    mask_LBC = (
        df_result["start_security_queue"] > df_result["STD"] - MUP_close_time
    )
    df_result.loc[mask_LBC, "LBC_in"] = df_result.loc[
        mask_LBC, "start_security_queue"
    ]
    df_result.loc[mask_LBC, "LBC_out"] = df_result.loc[mask_LBC, "STD"]

    plt_in_LBC = (
        df_result.loc[mask_LBC, ["LBC_in", "Pax_N"]]
        .set_index("LBC_in", drop=False)["Pax_N"]
        .resample("15min")
        .agg(["sum"])
        .cumsum()
    )

    plt_out_LBC = (
        df_result.loc[mask_LBC, ["LBC_out", "Pax_N"]]
        .set_index("LBC_out", drop=False)["Pax_N"]
        .resample("15min")
        .agg(["sum"])
        .cumsum()
    )

    LBC_req = (plt_in_LBC - plt_out_LBC).max()[0]

    # change NaN results to zero
    if np.isnan(EBS_req):
        EBS_req = 0

    if np.isnan(LBC_req):
        LBC_req = 0

    return EBS_req, LBC_req


def calculate_avg_dwell_time(df_result, offset=pd.Timedelta(minutes=15)):
    """
    we could use: df_result[["end_emigration_self_process", "end_emigration_counter_process"]]
    but as we do not consider immigration here (check-in study)
    let's consider Pax take about 15 minutes to clear immigration
    """
    # change the times after midnight to the next day to calculate properly
    end = pd.Timestamp("2020-10-13 02:00:00")
    mask_late_flight = df_result["STD"] < end
    df_result.loc[mask_late_flight, "STD"] += pd.Timedelta(days=1)

    df_dwell = df_result["STD"] - (df_result["end_security_process"] + offset)

    q_high = df_dwell.quantile(q=0.90)
    q_low = df_dwell.quantile(q=0.10)

    mask_q = (q_low < df_dwell) & (df_dwell < q_high)

    mean = df_dwell[mask_q].mean() / pd.Timedelta(minutes=1)
    top90 = df_dwell.quantile(q=0.9) / pd.Timedelta(minutes=1)

    return df_dwell, mean, top90