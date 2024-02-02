"""
Functions to download admissions data from the NHS England website.
"""

import pandas as pd
import plotly.express as px

from .data_sources import get_data_source
from .data_readers import read_data_source


def get_admissions_data() -> pd.DataFrame:
    """Download the A&E admissions data as a pandas Dataframe. Performs
    minor preprocessing steps to select the required rows/columns from the
    excel sheet.

    Returns:
        pd.DataFrame: A&E admissions data downloaded from the NHSE website.
    """
    activity_source = get_data_source("ae_activity")
    df_activity = read_data_source(activity_source).pipe(_prep_raw_activity_data)

    df_admissions = df_activity["Emergency Admissions"]

    return df_admissions


def _prep_raw_activity_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = [
        col_name
        for col_name in df_raw.columns.get_level_values(0)
        if "Unnamed" in col_name
    ]
    df_pre = df_raw.drop(cols_to_drop, axis=1, level=0)
    return df_pre


if __name__ == "__main__":
    df_admissions = get_admissions_data()
    print(df_admissions.head())

    fig = px.line(df_admissions, y="Total Emergency Admissions")
    fig.show()
