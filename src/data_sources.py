"""
Defines data sources and functions for getting them.
"""

import re
from datetime import datetime

from typing import Any, Literal
from dataclasses import dataclass, field
from string import Template

from .helpers.scrape_links import get_admissions_csv_links


FileFormat = Literal["excel"]
OriginType = Literal["url"]


# TODO: move extract functions to helpers


def extract_year(x: str):
    return next(re.finditer(r"\d{4}", x)).group(0)


def extract_month_name(x: str):
    return next(re.finditer(r"[A-Z][a-z]+", x)).group(0)


# TODO: add years
_SOURCE_YEARS = [22]

_AE_MONTHLY_NAME_TEMPLATE = Template("ae_monthly_${month_year}")
_AE_MONTHLY_DATA_SOURCES = {}
for start_yy in _SOURCE_YEARS:
    for link in get_admissions_csv_links(start_yy):
        filename = link.split("/")[-1]
        year = extract_year(filename)
        month = extract_month_name(filename)
        # TODO: get MM from month name
        # month = datetime.strptime(month_name, "%b")
        month_year = f"{month}_{year}"
        source_name = _AE_MONTHLY_NAME_TEMPLATE.substitute(month_year=month_year)
        data_source = {
            "origin": link,
        }
        _AE_MONTHLY_DATA_SOURCES[source_name] = data_source

# TODO: sort
# TODO: probably better to group by source and index by date
_DATA_SOURCES = {**_AE_MONTHLY_DATA_SOURCES}


@dataclass
class DataSource:
    origin: str
    origin_type: OriginType = "url"
    file_format: FileFormat = "csv"
    reader_params: dict[str, Any] = field(default_factory=lambda: {})


def get_data_source_by_name(source_name: str) -> DataSource:
    source_dict = _DATA_SOURCES[source_name]
    return DataSource(**source_dict)
