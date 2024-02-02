"""
"""

from typing import Callable

import pandas as pd

from .data_sources import DataSource, FileFormat


DataReaderType = Callable[[DataSource], pd.DataFrame]
_DATA_READERS: dict[FileFormat, DataReaderType] = {}


def read_data_source(source: DataSource) -> pd.DataFrame:
    reader = _DATA_READERS.get(source.file_format)

    if not reader:
        raise NotImplementedError(f"File format '{source.file_format}' not supported")

    return reader(source)


def _register_data_reader(
    file_format: FileFormat,
) -> Callable[[DataReaderType], DataReaderType]:
    def _decorator(reader: DataReaderType):
        _DATA_READERS[file_format] = reader
        return reader

    return _decorator


@_register_data_reader("excel")
def _excel_reader(source: DataSource) -> pd.DataFrame:
    return pd.read_excel(source.origin, **source.reader_params)
