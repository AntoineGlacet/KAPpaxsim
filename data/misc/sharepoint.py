# sharepoint.py
import os
from pathlib import Path

from decouple import AutoConfig
from office365.runtime.auth.client_credential import ClientCredential
from office365.sharepoint.client_context import ClientContext


def login_sharepoint():
    # get env variables and secrets from .env file
    DOTENV_FILE_PATH = Path(__file__).parent / "../../data/secret/.env"
    config = AutoConfig(search_path=DOTENV_FILE_PATH)
    # login to Sharepoint
    client_credentials = ClientCredential(config("client_id"), config("client_secret"))
    ctx = ClientContext(config("site_url")).with_credentials(client_credentials)
    return ctx, config


def get_ADRM_param_full():
    # login to Sharepoint
    ctx, config = login_sharepoint()
    download_path = Path(__file__).parent / ".." / ".." / config("ADRM_param_full_path")

    with open(download_path, "wb") as local_file:
        file = (
            ctx.web.get_file_by_server_relative_path(config("ADRM_param_full_url"))
            .download(local_file)
            .execute_query()
        )
    print(f"[Ok] file has been downloaded: {download_path}")


def get_schedule_forecast_FY19_25():
    # login to Sharepoint
    ctx, config = login_sharepoint()
    download_path = (
        Path(__file__).parent / ".." / ".." / config("schedule_forecast_FY19_25_path")
    )

    with open(download_path, "wb") as local_file:
        file = (
            ctx.web.get_file_by_server_relative_path(
                config("schedule_forecast_FY19_25_url"),
            )
            .download(local_file)
            .execute_query()
        )
    print(f"[Ok] file has been downloaded: {download_path}")


def get_T1_ren_6kPax_schedule():
    # login to Sharepoint
    ctx, config = login_sharepoint()
    download_path = (
        Path(__file__).parent / ".." / ".." / config("T1_ren_6kPax_schedule_path")
    )

    with open(download_path, "wb") as local_file:
        file = (
            ctx.web.get_file_by_server_relative_path(
                config("T1_ren_6kPax_schedule_url"),
            )
            .download(local_file)
            .execute_query()
        )
    print(f"[Ok] file has been downloaded: {download_path}")
