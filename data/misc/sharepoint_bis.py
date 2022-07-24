# sharepoint.py
import os
from pathlib import Path

from decouple import AutoConfig
from office365.runtime.auth.client_credential import ClientCredential
from office365.sharepoint.client_context import ClientContext
from tqdm import tqdm


def login_sharepoint(site: str = "FS"):
    """
    convenience function to login sharepoint site
    need to specify site amongst: (more can be added depending on needs)
        - MP: Technicaldepartment-Masterplan (eg. for ADRM files)
        - FS: KansaiAirportsFileServer (eg. for videos)
    """
    # get env variables and secrets from .env file
    DOTENV_FILE_PATH = Path(__file__).parent / "../.env"
    config = AutoConfig(search_path=DOTENV_FILE_PATH)
    # login to Sharepoint
    client_credentials = ClientCredential(config("client_id"), config("client_secret"))
    ctx = ClientContext(config(site)).with_credentials(client_credentials)
    return ctx


def download_file(
    site: str = "FS",
    source_file_path: Path = Path(
        r"/sites/KansaiAirportsFileServer/Shared Documents/Other/Throughput videos/test"
        r" folder/test.txt"
    ),
    download_file_path: Path = Path(__file__).parent / "../data/dump/test.txt",
):
    """
    download source path file of target site to download path file
    includes progress bar
    """
    # login to sharepoint
    ctx = login_sharepoint(site)

    # check if local path exist, create it otherwise
    parent_folder = download_file_path.parent
    if not parent_folder.exists():
        parent_folder.mkdir(mode=0o777, parents=True, exist_ok=False)

    # open local file for writing
    with open(download_file_path, "wb") as local_file:
        # get remote file metadata
        file = (
            ctx.web.get_file_by_server_relative_path(str(source_file_path))
            .get()
            .execute_query()
        )

        # write remote file to local file

        # dirty trick for pbar update
        global pbar

        def progress(offset):
            pbar.update(1024 * 1024)

        # end of the dirty trick

        with tqdm(
            total=int(file.properties["Length"]),
            desc="downloading {}".format(str(file.properties["Name"])),
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            (
                ctx.web.get_file_by_server_relative_path(str(source_file_path))
                .download_session(local_file, progress)
                .execute_query()
            )
    print("[Ok] file has been downloaded: {0}".format(download_file_path))


def get_items_in_directory(
    site: str = "FS",
    source_folder_path: Path = Path(
        r"/sites/KansaiAirportsFileServer/Shared Documents/Other/Throughput videos/test"
        r" folder"
    ),
    recursive: bool = True,
):
    """
    This function provides a way to get all items in a directory in SharePoint, with
    the option to traverse nested directories to extract all child objects.

    :param ctx_client: office365.sharepoint.client_context.ClientContext object
        SharePoint ClientContext object.
    :param directory_relative_uri: str
        Path to directory in SharePoint.
    :param recursive: bool
        default = False
        Tells function whether or not to perform a recursive call.
    :return: list (we divided into 2 lists)
        Returns a flattened array of all child file and/or folder objects
        given some parent directory. All items will be of the following types:
            - office365.sharepoint.file.File
            - office365.sharepoint.folder.Folder

    Examples
    ---------
    All examples assume you've already authenticated with SharePoint per
    documentation found here:
        - https://github.com/vgrem/Office365-REST-Python-Client#examples

    Assumed directory structure:
        some_directory/
            my_file.csv
            your_file.xlsx
            sub_directory_one/
                123.docx
                abc.csv
            sub_directory_two/
                xyz.xlsx

    directory = 'some_directory'
    # Non-recursive call
    extracted_child_objects = get_items_in_directory(directory)
    # extracted_child_objects would contain (my_file.csv, your_file.xlsx, sub_directory_one/, sub_directory_two/)


    # Recursive call
    extracted_child_objects = get_items_in_directory(directory, recursive=True)
    # extracted_child_objects would contain (my_file.csv, your_file.xlsx, sub_directory_one/, sub_directory_two/, sub_directory_one/123.docx, sub_directory_one/abc.csv, sub_directory_two/xyz.xlsx)

    """
    directory_relative_uri = str(source_folder_path)
    ctx_client = login_sharepoint(site)
    contents = list()
    folders = ctx_client.web.get_folder_by_server_relative_url(
        directory_relative_uri
    ).folders
    ctx_client.load(folders)
    ctx_client.execute_query()

    if recursive:
        for folder in folders:
            contents.extend(
                get_items_in_directory(
                    site=site,
                    source_folder_path=Path(folder.properties["ServerRelativeUrl"]),
                    recursive=recursive,
                )
            )

    # contents.extend(folders)

    files = ctx_client.web.get_folder_by_server_relative_url(
        directory_relative_uri
    ).files
    ctx_client.load(files)
    ctx_client.execute_query()

    # contents.extend(files)

    return folders, files


def download_folder(
    site: str = "FS",
    source_folder_path: Path = Path(
        r"/sites/KansaiAirportsFileServer/Shared Documents/Other/Throughput videos/test"
        r" folder"
    ),
    download_folder_path: Path = Path(__file__).parent / "../test",
    recursive: bool = True,
):
    """
    description
    """
    ctx = login_sharepoint(site)
    folders, files = get_items_in_directory(
        site=site,
        source_folder_path=source_folder_path,
        recursive=True,
    )

    for file in files:
        source_file_path = Path(file.serverRelativeUrl)
        download_file_path = download_folder_path / source_file_path.relative_to(
            source_folder_path
        )

        download_file(
            site=site,
            source_file_path=source_file_path,
            download_file_path=download_file_path,
        )
