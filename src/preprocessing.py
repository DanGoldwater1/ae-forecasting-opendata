"""
Functions to preprocess A&E data.
"""

import pandas as pd


def prepare_admissions_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare admissions data by renaming columns, adding derived fields and
    selecting a subset of columns.

    Args:
        df_raw (pd.DataFrame): Raw data downloaded from source.

    Returns:
        pd.DataFrame: Prepared data.
    """
    col_name_map = {
        "Org Code": "org_code",
        "Org name": "org_name",
        "Emergency admissions via A&E - Type 1": "ae_admissions_type1",
        "Emergency admissions via A&E - Type 2": "ae_admissions_type2",
        "Emergency admissions via A&E - Other A&E department": "ae_admissions_other",
        "Other emergency admissions": "admissions_other",
    }
    ae_admissions_cols = [
        "ae_admissions_type1",
        "ae_admissions_type2",
        "ae_admissions_other",
    ]
    derivations = {"ae_admissions_total": lambda df: df[ae_admissions_cols].sum(axis=1)}
    selected_cols = [
        "org_code",
        "org_name",
        "source_date",
        *ae_admissions_cols,
        "ae_admissions_total",
        "admissions_other",
    ]
    df_prepped = df_raw.rename(columns=col_name_map).assign(**derivations)[
        selected_cols
    ]
    return df_prepped
