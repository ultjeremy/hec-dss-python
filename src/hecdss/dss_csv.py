import csv
import datetime
from .regular_timeseries import RegularTimeSeries
from .irregular_timeseries import IrregularTimeSeries
from .dsspath import DssPath

def timeseries_to_csv(series: RegularTimeSeries | IrregularTimeSeries, path: str, with_metadata: bool) -> None:
    """
    Exports a timeseries (either regular or irregular) to a .csv file.

    Args:
        series: The timeseries object to export. Must be either a RegularTimeSeries or IrregularTimeSeries
        path (str): The file path where the .csv file will be exported.
        with_metadata (bool): Whether to include metadata in the .csv file.
    """
    if not isinstance(series, (RegularTimeSeries, IrregularTimeSeries)):
        raise TypeError(f"series must be a RegularTimeSeries or IrregularTimeSeries")

    metadata_rows: list[str] = ['A', 'B', 'C', 'D', 'E', 'F']
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer: csv.writer = csv.writer(f)
        if with_metadata:
            if series.id:
                id_components: list[str] = DssPath(series.id).path_to_list()
            else:
                id_components: list[str] = [""] * 6
            for i in range(len(metadata_rows)):
                row: str = metadata_rows[i]
                metadata_value: str = id_components[i]
                if row == 'D': # Skip D by convention
                    continue
                writer.writerow([row, '', '', metadata_value]) # Writing metadata rows
            writer.writerow(['Units', '', '', series.units])
            if len(series.quality) > 0:
                writer.writerow(['Type', 'Date/Time', series.data_type, 'Quality']) # Write column names with quality
            else:
                writer.writerow(['Type', 'Date/Time', series.data_type]) # Write column names without quality
            
        time_format: str = "%d%b%Y %H%M%S" if _needs_second_precision(series) else "%d%b%Y %H%M"

        ordinate: int = 1
        if len(series.quality) > 0:
            for time, value, quality in zip(series.times, series.values, series.quality):
                formatted_time: datetime = time.strftime(time_format)
                writer.writerow([ordinate, formatted_time, value, quality])
                ordinate += 1
        else:
            for time, value in zip(series.times, series.values):
                formatted_time: datetime = time.strftime(time_format)
                writer.writerow([ordinate, formatted_time, value])
                ordinate += 1


def _needs_second_precision(series: RegularTimeSeries | IrregularTimeSeries) -> bool:
    """
    Returns True if any datetime in the series has a non-zero seconds component

    Args:
        series: must be either a RegularTimeSeries or IrregularTimeSeries

    Returns:
        bool: whether we need seconds precision or not
    """
    if not isinstance(series, (RegularTimeSeries, IrregularTimeSeries)):
        raise TypeError(f"series must be a RegularTimeSeries or IrregularTimeSeries")

    return any(getattr(t, "second", 0) != 0 for t in series.times)