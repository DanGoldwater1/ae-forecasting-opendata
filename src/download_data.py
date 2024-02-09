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
    activity_source = get_data_source("ae_monthly_jan24")
    df_raw = read_data_source(activity_source)

    return df_raw


if __name__ == "__main__":
    df_admissions = get_admissions_data()
    print(df_admissions.head())

    # Plot type 1 admissions for January 2024
    time_col = "Period"
    location_col = "Org Code"
    metric_col = "Emergency admissions via A&E - Type 1"
    row_filter = (df_admissions[location_col] != "TOTAL") & (
        df_admissions[time_col] == "MSitAE-JANUARY-2024"
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
