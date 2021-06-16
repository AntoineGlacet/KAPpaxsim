import os
from pathlib import Path

from decouple import AutoConfig
from office365.runtime.auth.client_credential import ClientCredential
from office365.sharepoint.client_context import ClientContext

# get env variables and secrets from .env file
DOTENV_FILE_PATH = Path(__file__).parent / "../../data/secret/.env"
config = AutoConfig(search_path=DOTENV_FILE_PATH)

def get_ADRM_param_full():
    # login to Sharepoint
    client_credentials = ClientCredential(config('client_id'),config('client_secret'))
    ctx = ClientContext(config('site_url')).with_credentials(client_credentials)
    
    # to be set in .env file?    
    download_path = Path(__file__).parent / "../../data/raw/ADRM_param_full.xlsx"

    with open(download_path, "wb") as local_file:
        file = ctx.web.get_file_by_server_relative_path(config('ADRM_param_full_xlsx_url')).download(local_file).execute_query()
    print("[Ok] file has been downloaded: {0}".format(download_path))
    
def get_schedule_forecast():
        # login to Sharepoint
    client_credentials = ClientCredential(config('client_id'),config('client_secret'))
    ctx = ClientContext(config('site_url')).with_credentials(client_credentials)
    
    # to be set in .env file?    
    download_path = Path(__file__).parent / "../../data/raw/FY2019_FY2025 merge.xlsx"

    with open(download_path, "wb") as local_file:
        file = ctx.web.get_file_by_server_relative_path(config('FY2019_FY2025_xlsx_url')).download(local_file).execute_query()
    print("[Ok] file has been downloaded: {0}".format(download_path))
    
    