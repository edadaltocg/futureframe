import logging
import os
from abc import ABC
from dataclasses import dataclass
from enum import Enum
from typing import Callable

import requests
from sklearn.datasets import fetch_openml

from futureframe.utils import extract_zip, recursive_unzip

log = logging.getLogger(__name__)


@dataclass
class Source(ABC):
    """
    Represents a data source for downloading datasets.

    Attributes:
        link (str): The URL of the dataset.
        base_name (str): The base name of the downloaded file.
        download_fn (Callable[[str, str], None]): The function used to download the dataset.
        get_dataset_id_fn (Callable[[str], str]): The function used to get the dataset ID.
    """

    link: str
    base_name: str
    download_fn: Callable[[str, str], None]
    get_dataset_id_fn: Callable[[str], str]


def download_dataset_from_kaggle(link: str, dest: str):
    dataset_id = get_dataset_id_from_kaggle(link)
    os.makedirs(dest, exist_ok=True)
    try:
        os.system(f"kaggle datasets download -d {dataset_id} -p {dest}")
    except Exception as e:
        log.error(f"Failed to download dataset from Kaggle: {e}")
        log.warning("This function requires the Kaggle API and authentication.")
        raise e

    recursive_unzip(dest)


def download_metadata_from_kaggle(link: str, dest: str):
    dataset_id = get_dataset_id_from_kaggle(link)
    os.makedirs(dest, exist_ok=True)

    try:
        os.system(f"kaggle datasets metadata -d {dataset_id} -p {dest}")
    except Exception as e:
        log.error(f"Failed to download metadata from Kaggle: {e}")
        log.warning("This function requires the Kaggle API and authentication.")
        raise e


def download_competition_from_kaggle(link: str, dest: str):
    competition_id = get_competition_id_from_kaggle(link)
    os.makedirs(dest, exist_ok=True)
    try:
        os.system(f"kaggle competitions download -c {competition_id} -p {dest}")
    except Exception as e:
        log.error(f"Failed to download dataset from Kaggle: {e}")
        log.warning("This function requires the Kaggle API and authentication.")
        raise e

    for file in os.listdir(dest):
        if file.endswith(".zip"):
            zip_path = os.path.join(dest, file)
            extract_zip(zip_path, dest, True)


def download_dataset_from_openml(link: str, dest: str):
    # Extract dataset id from the link
    dataset_id = int(get_dataset_id_from_openml(link))
    X, y = fetch_openml(data_id=dataset_id, as_frame=True, n_retries=5, return_X_y=True)
    X.to_csv(os.path.join(dest, "X.csv"), index=False)
    y.to_csv(os.path.join(dest, "y.csv"), index=False)


def download_dataset_from_uci(*args, **kwargs):
    raise NotImplementedError


def download_dataset_from_catalog(*args, **kwargs):
    raise NotImplementedError


def download_dataset_from_link(link: str, dest: str):
    response = requests.get(link, stream=True)
    if response.status_code == 200:
        with open(os.path.join(dest, os.path.basename(link)), "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    else:
        print(f"Failed to download from {link}")


def download_with_request(url, download_path):
    response = requests.get(url, stream=True)
    with open(download_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)


def get_dataset_id_from_openml(link: str):
    link = link.rstrip("/")
    return link.split("/")[-1]


def get_dataset_id_from_kaggle(link: str):
    link = link.rstrip("/")
    return "/".join(link.split("/")[-2:])


def get_competition_id_from_kaggle(link: str):
    link = link.rstrip("/")
    return link.split("/")[-1]


def get_dataset_id_from_uci(link: str):
    raise NotImplementedError


def get_dataset_id_from_catalog(link: str):
    raise NotImplementedError


class Sources(Enum):
    KAGGLE = Source(
        link="kaggle.com/datasets",
        base_name="kaggle",
        download_fn=download_dataset_from_kaggle,
        get_dataset_id_fn=get_dataset_id_from_kaggle,
    )
    KAGGLE_COMPETITIONS = Source(
        link="kaggle.com/competitions",
        base_name="kaggle/competitions",
        download_fn=download_competition_from_kaggle,
        get_dataset_id_fn=get_competition_id_from_kaggle,
    )
    OPENML = Source(
        link="openml.org",
        base_name="openml",
        download_fn=download_dataset_from_openml,
        get_dataset_id_fn=get_dataset_id_from_openml,
    )
    UCI = Source(
        link="archive.ics.uci.edu",
        base_name="uci",
        download_fn=download_dataset_from_uci,
        get_dataset_id_fn=get_dataset_id_from_uci,
    )
    CATALOG = Source(
        link="catalog",
        base_name="catalog",
        download_fn=download_dataset_from_catalog,
        get_dataset_id_fn=get_dataset_id_from_catalog,
    )


def get_dataset_source_from_link(link: str):
    for source in Sources:
        if source.value.link in link:
            return source
    raise ValueError(f"Unknown source for link {link}")


def get_dataset_base_name_from_link(link: str):
    source = get_dataset_source_from_link(link)
    return source.value.base_name


def get_dataset_id_from_link(link: str):
    source = get_dataset_source_from_link(link)
    return source.value.get_dataset_id_fn(link)


def get_dataset_dest_from_link(link: str, root_dir: str):
    base_name = get_dataset_base_name_from_link(link)
    id = get_dataset_id_from_link(link)

    return os.path.join(root_dir, base_name, id)


def download_dataset(link: str, root_dir: str):
    source = get_dataset_source_from_link(link)
    dest = get_dataset_dest_from_link(link, root_dir)
    source.value.download_fn(link, dest)


if __name__ == "__main__":
    from fire import Fire

    Fire(download_dataset)