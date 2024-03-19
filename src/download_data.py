"""
Functions to download admissions data from the NHS England website.
"""

from pytz import timezone
from datetime import datetime, date
from itertools import starmap

import pandas as pd
import plotly.express as px

from .data_sources import DataSource, get_ae_monthly_data_sources
from .data_readers import read_data_source

UTC_TZ = timezone("UTC")


def get_admissions_data() -> pd.DataFrame:
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


if __name__ == "__main__":
    df_admissions = get_admissions_data()
    print(df_admissions.columns)
    print(df_admissions.head())

    # TODO: rename columns
    # TODO: parse period into date

    # Plot type 1 admissions for January 2024
    time_col = "Period"
    location_col = "Org Code"
    metric_col = "Emergency admissions via A&E - Type 1"
    row_filter = (df_admissions[location_col] != "TOTAL") & (
        df_admissions[time_col] == "MSitAE-JANUARY-2023"
    )

    df_plot = df_admissions.loc[row_filter].sort_values(
        by=[metric_col], ignore_index=True, ascending=False
    )
    fig = px.bar(
        df_plot,
        x=location_col,
        y=metric_col,
    )
    fig.show()
