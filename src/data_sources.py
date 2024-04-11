"""
Defines data sources and functions for getting them.
"""

from tenacity import retry
from typing import Any, Literal
from dataclasses import dataclass, field
from string import Template
from functools import cache
from datetime import datetime, date
import tenacity

from utils.web_scraping import find_all_tags
from utils.string_parsers import extract_month_name, extract_year


FileFormat = Literal["csv", "excel"]
OriginType = Literal["url"]
SourceGroup = Literal["ae_monthly"]

AE_BASE_URL = "https://www.england.nhs.uk/statistics/statistical-work-areas/ae-waiting-times-and-activity"
ADMISSIONS_SUBPAGE_TEMPLATE = Template(
    "ae-attendances-and-emergency-admissions-20${start_yy}-${end_yy}"
)
# TODO: add years
_SOURCE_YEARS = list(range(20, 22))


@dataclass
class DataSource:
    origin: str
    origin_type: OriginType = "url"
    file_format: FileFormat = "csv"
    reader_params: dict[str, Any] = field(default_factory=lambda: {})


def get_ae_monthly_data_sources() -> dict[date, DataSource]:
    data_sources = _get_data_sources()
    return data_sources["ae_monthly"]


@cache
def _get_data_sources() -> dict[SourceGroup, dict[date, DataSource]]:
    _AE_MONTHLY_DATA_SOURCES = {}
    for start_yy in _SOURCE_YEARS:
        for link in _get_admissions_csv_links(start_yy):
            filename = link.split("/")[-1]
            year_yyyy = extract_year(filename)
            month_name = extract_month_name(filename)
            # TODO: get MM from month name
            month_mm = datetime.strptime(month_name, "%B").strftime("%m")
            source_date = datetime(
                year=int(year_yyyy), month=int(month_mm), day=1
            ).date()
            data_source = {
                "origin": link,
            }
            _AE_MONTHLY_DATA_SOURCES[source_date] = DataSource(**data_source)

    return {"ae_monthly": _AE_MONTHLY_DATA_SOURCES}


@retry(
        stop=tenacity.stop_after_attempt(4), 
        wait=tenacity.wait_exponential(),
        )
def _get_admissions_csv_links(start_yy: str | int) -> list[str]:
    end_yy = f"{int(start_yy) + 1}"
    subpage_ext = ADMISSIONS_SUBPAGE_TEMPLATE.substitute(
        start_yy=start_yy, end_yy=end_yy
    )
    url = f"{AE_BASE_URL}/{subpage_ext}"

    def is_csv_link(result):
        return "CSV" in result.text

    def is_ae_link(result):
        return "Monthly A&E" in result.text

    results = find_all_tags(url, "a", [is_csv_link, is_ae_link])
    links = [result.get("href") for result in results]

    return links
