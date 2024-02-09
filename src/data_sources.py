"""
Defines data sources and functions for getting them.
"""

from typing import Any, Literal
from dataclasses import dataclass, field


FileFormat = Literal["excel"]
NHSE_BASE_URL = "https://www.england.nhs.uk/statistics/wp-content/uploads/sites/2"


# Available from: https://www.england.nhs.uk/statistics/statistical-work-areas/ae-waiting-times-and-activity/
_AE_MONTHLY_FILES = [
    ("012024", "2024/02/Monthly-AE-January-2024.csv"),
    ("122023", "2024/03/Monthly-AE-December-2023.csv"),
]
_AE_MONTHLY_DATA_SOURCES = {
    f"ae_monthly_{month_year}": {
        "origin": f"{NHSE_BASE_URL}/{filename}",
        "origin_type": "url",
        "file_format": "csv",
        "reader_params": {},
    }
    for month_year, filename in _AE_MONTHLY_FILES
}
_DATA_SOURCES = {**_AE_MONTHLY_DATA_SOURCES}


@dataclass
class DataSource:
    origin: str
    origin_type: Literal["url"]
    file_format: FileFormat
    reader_params: dict[str, Any] = field(default_factory=lambda: {})


def get_data_source(source_name: str) -> DataSource:
    source_dict = _DATA_SOURCES[source_name]
    return DataSource(**source_dict)
