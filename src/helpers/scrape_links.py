"""
Functions to scrape the NHSE publications webpages to find the download links
to admissions files.
"""

from string import Template
from typing import Callable, Sequence, Any

import requests
from bs4 import BeautifulSoup, ResultSet

AE_BASE_URL = "https://www.england.nhs.uk/statistics/statistical-work-areas/ae-waiting-times-and-activity"
ADMISSIONS_SUBPAGE_TEMPLATE = Template(
    "ae-attendances-and-emergency-admissions-20${start_yy}-${end_yy}"
)
LINK_TAG = "a"


def get_admissions_csv_links(start_yy: str | int) -> list[str]:
    end_yy = f"{int(start_yy) + 1}"
    subpage_ext = ADMISSIONS_SUBPAGE_TEMPLATE.substitute(
        start_yy=start_yy, end_yy=end_yy
    )
    url = f"{AE_BASE_URL}/{subpage_ext}"

    def is_csv_link(result):
        return "CSV" in result.text

    def is_ae_link(result):
        return "Monthly A&E" in result.text

    results = _find_all_tags(url, LINK_TAG, [is_csv_link, is_ae_link])
    links = [result.get("href") for result in results]

    return links


def _find_all_tags(
    url: str,
    tag: str,
    filters: Callable[[ResultSet], bool],
    features: str | Sequence[str] | None = "html.parser",
    **kwargs: Any,
):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, features, **kwargs)
    results = soup.find_all(tag)

    for f in filters:
        results = filter(f, results)

    return list(results)
