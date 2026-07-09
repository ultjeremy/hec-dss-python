import csv
from datetime import datetime, timedelta
from .regular_timeseries import RegularTimeSeries
from .irregular_timeseries import IrregularTimeSeries
from .dsspath import DssPath


def timeseries_to_csv(series: RegularTimeSeries | IrregularTimeSeries, path: str, with_metadata: bool) -> None:
    """
    Exports a timeseries (either regular or irregular) to a .csv file.

    Parameters:
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
                if row == 'D': # Skip D by convention # ts patter
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


def timeseries_read_csv(cls: type[RegularTimeSeries] | type[IrregularTimeSeries], path: str) -> RegularTimeSeries | IrregularTimeSeries:
    """
    Reads a .csv file and builds a new timeseries of the given type.

    Parameters:
        cls: the class to build - either RegularTimeSeries or IrregularTimeSeries
        path (str): File path to .csv that we are reading from

    Returns:
        An instance of cls populated from the .csv file.
    """
    if cls not in (RegularTimeSeries, IrregularTimeSeries):
        raise TypeError("cls must be RegularTimeSeries or IrregularTimeSeries")
    
    times, values, quality, units, data_type = [], [], [], "", ""
    path_parts: dict[str, str] = {'A': '', 'B': '', 'C': '', 'D': '', 'E': '', 'F': ''}
    has_quality: bool = False # flags

    _DSS_TIME_FORMATS_BY_LENGTH: dict[int, str] = {
        14: "%d%b%Y %H%M",
        16: "%d%b%Y %H%M%S" # Numbers subject to change if we add ISO support
    }

    with open(path, 'r', newline='', encoding='utf-8') as f:
        reader: csv.reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            first_column_item: str = row[0].strip() # first item in the row we grabbed, the first column's item
            if first_column_item in path_parts: # If the first column item is a path component (['A', 'B', 'C', 'D', 'E', 'F'])
                path_parts[first_column_item] = row[-1].strip() # last cell in csv row (convention)
            elif first_column_item == 'Units':
                units = row[-1].strip()
            elif first_column_item == 'Type': # reached the header
                if len(row) >= 3: # ['Type', 'Date/Time', data_type, ...potentially more]
                    data_type = row[2].strip()
                if len(row) >= 4 and row[3].strip() == 'Quality': # ['Type', 'Date/Time', data_type, 'Quality', ...potentially more] 
                    has_quality = True
            else: # Data row
                if len(row) < 3:
                    continue # csv is malformed, something is missing

                raw_time: str = row[1].strip()
                roll_day: bool = "2400" in raw_time # Do we need to roll over the day date? Yes if time is 2400
                if roll_day: # 2400 isn't a valid hour, so roll it to 0000 before parsing and add a day after
                    raw_time = raw_time.replace(" 2400", " 0000")

                time_format: str = _DSS_TIME_FORMATS_BY_LENGTH.get(len(raw_time)) # Time format is determined by length of raw_time
                if time_format is None:
                    # Time format is unrecognized
                    continue

                try:
                    time: datetime = datetime.strptime(raw_time, time_format)
                except ValueError:
                    continue # Skip a malformed date
                if roll_day: # Convert a 24:00 time to 00:00 of the next day
                    time += timedelta(days=1)

                value_str: str = row[2].strip()
                try:
                    value: float = float(value_str) if value_str else 0.0
                except ValueError:
                    continue # Skip a malformed value

                times.append(time)
                values.append(value)
                if has_quality: # Always keep quality index-aligned with values, defaulting a missing cell to 0
                    quality_str: str = row[3].strip() if len(row) >= 4 else ""
                    quality.append(int(quality_str) if quality_str else 0)

    id: str = f"/{path_parts['A']}/{path_parts['B']}/{path_parts['C']}/{path_parts['D']}/{path_parts['E']}/{path_parts['F']}/"
    interval: str | int = path_parts['E']

    return cls.create(values=values, times=times, quality=quality, units=units, data_type=data_type, interval=interval, path=id)


def _needs_second_precision(series: RegularTimeSeries | IrregularTimeSeries) -> bool:
    """
    Returns True if any datetime in the series has a non-zero seconds component

    Parameters:
        series: must be either a RegularTimeSeries or IrregularTimeSeries

    Returns:
        bool: whether we need seconds precision or not
    """
    if not isinstance(series, (RegularTimeSeries, IrregularTimeSeries)):
        raise TypeError(f"series must be a RegularTimeSeries or IrregularTimeSeries")

    return any(getattr(t, "second", 0) != 0 for t in series.times)