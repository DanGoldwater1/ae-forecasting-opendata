"""
Functions to download admissions data from the NHS England website.
"""

from pytz import timezone
from datetime import datetime, date
from itertools import starmap

import pandas as pd

from .data_sources import DataSource, get_ae_monthly_data_sources
from .data_readers import read_data_source

UTC_TZ = timezone("UTC")


def download_admissions_data() -> pd.DataFrame:
    """Download the A&E admissions data as a pandas Dataframe. Performs
    minor preprocessing steps to select the required rows/columns from the
    excel sheet.

    Returns:
        pd.DataFrame: A&E admissions data downloaded from the NHSE website.
    """

    def read_ae_monthly_data(
        source_date: date, data_source: DataSource
    ) -> pd.DataFrame:
        df_source = read_data_source(data_source)
        df_source["ingested_timestamp"] = UTC_TZ.localize(datetime.now())
        df_source["source_date"] = source_date
        df_source["origin"] = data_source.origin
        return df_source

    activity_sources = get_ae_monthly_data_sources()
    df_raw = pd.concat(starmap(read_ae_monthly_data, activity_sources.items()))
    return df_raw
