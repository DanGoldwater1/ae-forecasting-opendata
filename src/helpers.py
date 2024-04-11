"""Higher-order helper functions.
"""

import pandas as pd

from preprocessing import prepare_admissions_data
from download_data import download_admissions_data


def get_admissions_data() -> pd.DataFrame:
    """
    Helper function to download admissions data and apply preprocessing.

    Returns:
        pd.DataFrame: Preprocessed admissions data.
    """
    return download_admissions_data().pipe(prepare_admissions_data)
