"""
Defines data sources and functions for getting them.
"""

from typing import Any, Literal
from dataclasses import dataclass


FileFormat = Literal["excel"]

_DATA_SOURCES = {
    # Available from: https://www.england.nhs.uk/statistics/statistical-work-areas/ae-waiting-times-and-activity/
    "ae_activity": {
        "origin": "https://www.england.nhs.uk/statistics/wp-content/uploads/sites/2/2023/11/Monthly-AE-Time-Series-October-2023.xls",
        "origin_type": "url",
        "file_format": "excel",
        "reader_params": {
            "sheet_name": "Activity",
            "skiprows": 12,
            "index_col": 1,
            "header": [0, 1],
        },
    }
}


@dataclass
class DataSource:
    origin: str
    origin_type: Literal["url"]
    file_format: FileFormat
    reader_params: dict[str, Any]


def get_data_source(source_name: str) -> DataSource:
    source_dict = _DATA_SOURCES[source_name]
    return DataSource(**source_dict)
