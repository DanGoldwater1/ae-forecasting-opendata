"""
Functions to scrape webpages.
"""

from typing import Callable, Sequence, Any

import requests
from bs4 import BeautifulSoup, ResultSet


def find_all_tags(
    url: str,
    tag: str,
    filters: Callable[[ResultSet], bool],
    features: str | Sequence[str] | None = "html.parser",
    **kwargs: Any,
) -> list[ResultSet]:
    response = requests.get(url)
    soup = BeautifulSoup(response.content, features, **kwargs)
    results = soup.find_all(tag)

    for f in filters:
        results = filter(f, results)

    return list(results)
