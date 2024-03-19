import re


def extract_year(x: str) -> str:
    return next(re.finditer(r"\d{4}", x)).group(0)


def extract_month_name(x: str) -> str:
    # BUG: this doesn't work for years after 22 due to 'Monthly'
    # within file name
    return next(re.finditer(r"[A-Z][a-z]+", x)).group(0)
