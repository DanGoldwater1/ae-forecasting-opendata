"""
Functions to download admissions data from the NHS England website.
"""
from typing import Any, Callable, Literal

import pandas as pd
import plotly.express as px

DATA_SOURCES = {
    # Available from: https://www.england.nhs.uk/statistics/statistical-work-areas/ae-waiting-times-and-activity/
    "ae_activity": {
        "io": "https://www.england.nhs.uk/statistics/wp-content/uploads/sites/2/2023/11/Monthly-AE-Time-Series-October-2023.xls",
        "io_type": "url",
        "file_type": "excel",
        "loader_kwargs": {
            "sheet_name": "Activity",
            "skiprows": 12,
            "index_col": 1,
            "header": [0, 1],
        },
    }
}

DATA_READERS = {"excel": pd.read_excel}


def get_admissions_data() -> pd.DataFrame:
    """Download the A&E admissions data as a pandas Dataframe. Performs
    minor preprocessing steps to select the required rows/columns from the
    excel sheet.

    Returns:
        pd.DataFrame: A&E admissions data downloaded from the NHSE website.
    """
    activity_source = DATA_SOURCES["ae_activity"]
    df_activity = _get_data_from_source(activity_source).pipe(_prep_raw_activity_data)

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


def _get_data_from_source(source: dict[str, Any]) -> pd.DataFrame:
    io_type = source.get("io_type", "url")
    read_data = _get_data_reader(source["file_type"])

    if io_type == "url":
        df = read_data(source["io"], **source.get("loader_kwargs"))
    else:
        raise NotImplementedError(f"IO type '{io_type}' not supported")

    return df


def _get_data_reader(file_type: Literal["excel"]) -> Callable[[Any], pd.DataFrame]:
    data_reader = DATA_READERS.get(file_type)

    if data_reader:
        return data_reader
    else:
        raise NotImplementedError(f"File type '{file_type}' not supported")


if __name__ == "__main__":
    df_admissions = get_admissions_data()
    print(df_admissions.head())

    fig = px.line(df_admissions, y="Total Emergency Admissions")
    fig.show()
