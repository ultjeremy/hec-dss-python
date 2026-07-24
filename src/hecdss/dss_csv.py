# THESE ARE ALL HELPER FUNCTIONS, MEANT TO BE TREATED AS PRIVATE. These functions are internally called in their respective classes.

import csv
import re
from datetime import datetime, timedelta

from .dsspath import DssPath
from .irregular_timeseries import IrregularTimeSeries
from .paired_data import PairedData
from .regular_timeseries import RegularTimeSeries


ROUND_PRECISION: int = 4
DEFAULT_MISSING_VALUE = None


def timeseries_to_csv(series: RegularTimeSeries | IrregularTimeSeries, path: str) -> None:
    """
    Exports a timeseries (either regular or irregular) to a .csv file.

    Parameters:
        series: The timeseries object to export. Must be either a RegularTimeSeries or IrregularTimeSeries
        path (str): The file path where the .csv file will be exported.
    """
    if not isinstance(series, (RegularTimeSeries, IrregularTimeSeries)):
        raise TypeError("series must be a RegularTimeSeries or IrregularTimeSeries")

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        header: list[str] = ["Date/Time", "Value"]

        has_quality: bool = len(series.quality) > 0
        has_notes: bool = len(series.notes) > 0

        if has_quality:
            header.append("Quality")
        if has_notes:
            header.append("Notes")

        header.append("Units")
        header.append("Type")
        header.append("Path")

        writer.writerow(header)

        time_format: str = ("%d%b%Y %H%M%S" if _needs_second_precision(series) else "%d%b%Y %H%M")

        for i, (time, value) in enumerate(zip(series.times, series.values)):
            formatted_time: str = time.strftime(time_format)
            row: list[str] = [formatted_time, value]

            if has_quality:
                try:
                    row.append(series.quality[i])
                except IndexError:
                    row.append(DEFAULT_MISSING_VALUE)
            if has_notes:
                try:
                    row.append(series.notes[i])
                except IndexError:
                    row.append(DEFAULT_MISSING_VALUE)

            if i == 0:  # Add metadata to only first row
                row.append(series.units)
                row.append(series.data_type)
                row.append(series.id)

            writer.writerow(row)


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

    times, values, quality, notes, units, data_type, id = [], [], [], [], "", "", ""
    column_index: dict[str, int] = {}

    with open(path, "r", newline="", encoding="utf-8") as f:
        is_first_data_row: bool = True
        header_detected: bool = False

        reader = csv.reader(f)
        row: list[str]
        for row in reader:
            if not row:
                continue
            # first item in the row we grabbed, the first column's item
            first_column_item: str = row[0].strip()

            if first_column_item == "Date/Time":
                header_detected = True
                column_index = {name.strip(): i for i, name in enumerate(row)}
                continue

            if len(row) < 2:
                continue  # need at least Date/Time and Value

            if not header_detected:
                raise ValueError("No header found!")

            if is_first_data_row:
                is_first_data_row = False

                units_idx: int = column_index.get("Units")
                if units_idx is not None and len(row) > units_idx:
                    units = row[units_idx].strip()
                type_idx: int = column_index.get("Type")
                if type_idx is not None and len(row) > type_idx:
                    data_type = row[type_idx].strip()
                path_idx: int = column_index.get("Path")
                if path_idx is not None and len(row) > path_idx:
                    id = row[path_idx].strip()

            time_idx: int = column_index.get("Date/Time")
            raw_time: str = row[time_idx].strip()
            time_format: str = _get_time_format(raw_time)
            if time_format is None:
                # Time format is unrecognized
                continue

            roll_day: bool = _need_roll_day(time_format, raw_time)
            if roll_day:  # 2400 isn't a valid hour, so roll it to 0000 before parsing and add a day after
                raw_time = raw_time.replace(" 2400", " 0000")

            try:
                time: datetime = datetime.strptime(raw_time, time_format)
            except ValueError:
                continue  # Skip a malformed date

            if roll_day:  # Convert a 24:00 time to 00:00 of the next day
                time += timedelta(days=1)

            value_idx: int = column_index.get("Value")
            value_str: str = row[value_idx].strip()
            try:
                value: float = float(value_str) if value_str else DEFAULT_MISSING_VALUE
            except ValueError:
                value: float = DEFAULT_MISSING_VALUE  # If datetime is valid but not value, use DEFAULT_MISSING_VALUE

            times.append(time)
            values.append(value)

            quality_idx: int = column_index.get("Quality")
            if quality_idx is not None:
                quality_str = row[quality_idx].strip() if len(row) > quality_idx else ""
                try:
                    quality.append(int(quality_str) if quality_str else 0)
                except ValueError:
                    quality.append(0)

            notes_idx: int = column_index.get("Notes")
            if notes_idx is not None:
                notes.append(row[notes_idx].strip() if len(row) > notes_idx else "")

    path_parts: dict = _id_to_path_parts(id)
    interval: str | int = path_parts["E"]

    return cls.create(
        values=values,
        times=times,
        quality=quality,
        notes=notes,
        units=units,
        data_type=data_type,
        interval=interval,
        path=id or None,
    )


def paired_data_to_csv(paired_data: PairedData, path: str) -> None:
    """
    Exports a PairedData object to a .csv file.

    Parameters:
        paired_data (PairedData): Paired Data object to convert to csv.
        path: (str): file path to export csv to.

    Returns:
        None
    """
    if not isinstance(paired_data, PairedData):
        raise TypeError("paired_data must be a PairedData!")

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        path_parts: dict = _id_to_path_parts(paired_data.id)
        c_part: str = path_parts["C"]

        curve_count: int = paired_data.curve_count()
        header_labels: list[str] = list(paired_data.labels) + [""] * \
            (curve_count - len(paired_data.labels))

        writer.writerow([_prefix_before_dash(c_part)] + header_labels + ["X Units",
                        "Y Units", "X Type", "Y Type", "Path"])

        x: float  # x is the same as "ordinate"
        y_row: list[float]  # Each y_row is one row of y_values, as paired_data.values is Row-Major
        is_first_data_row: bool = True
        for x, y_row in zip(paired_data.ordinates, paired_data.values):
            full_row: list[float] = [_round_or_none(x)] + \
                [_round_or_none(y) for y in y_row]
            if is_first_data_row:
                is_first_data_row = False
                full_row.extend([paired_data.units_independent, paired_data.units_dependent,
                                paired_data.type_independent, paired_data.type_dependent, paired_data.id])
            writer.writerow(full_row)

    return


def paired_data_read_csv(cls: type[PairedData], path: str) -> PairedData:
    """
    Reads a .csv file and builds a new paired data of the given type.

    Parameters:
        cls: the class to build - PairedData
        path (str): File path to .csv that we are reading from

    Returns:
        An instance of cls populated from the .csv file.
    """
    if cls is not PairedData:
        raise TypeError("cls must be PairedData")

    x_values: list[float] = []
    y_values: list[list[float]] = []
    labels: list[str] = []
    x_units: str = ""
    x_type: str = ""
    y_units: str = ""
    y_type: str = ""
    id: str = ""
    column_index: dict[str, int] = {}

    with open(path, "r", newline="", encoding="utf-8") as f:
        is_first_data_row: bool = True
        header_detected: bool = False

        reader = csv.reader(f)
        row: list[str]
        for row in reader:
            if not row:
                continue

            if row[-1] == "Path":
                header_detected = True
                column_index = {name.strip(): i for i, name in enumerate(row)}

                # The index of the x units column is the number of curves + 1 (1 is the x column)
                x_units_col: int = column_index.get("X Units", len(row) - 5)
                curve_count: int = x_units_col - 1

                for label in row[1:x_units_col]:
                    labels.append(label)
                continue

            if len(row) < 2:
                continue

            if not header_detected:
                raise ValueError("No header found!")

            if not is_first_data_row and curve_count != len(row) - 1:
                raise ValueError("Curve count mismatch!")

            if is_first_data_row:
                is_first_data_row = False

                if len(row) - 1 < curve_count:
                    raise ValueError("Curve count mismatch!")

                x_units_idx: int = column_index.get("X Units")
                if x_units_idx is not None and len(row) > x_units_idx:
                    x_units = row[x_units_idx].strip()
                y_units_idx: int = column_index.get("Y Units")
                if y_units_idx is not None and len(row) > y_units_idx:
                    y_units = row[y_units_idx].strip()
                x_type_idx: int = column_index.get("X Type")
                if x_type_idx is not None and len(row) > x_type_idx:
                    x_type = row[x_type_idx].strip()
                y_type_idx: int = column_index.get("Y Type")
                if y_type_idx is not None and len(row) > y_type_idx:
                    y_type = row[y_type_idx].strip()
                path_idx: int = column_index.get("Path")
                if path_idx is not None and len(row) > path_idx:
                    id = row[path_idx].strip()

            # If we make it here, we are a data row
            x_column_idx: int = 0
            x_str: str = row[x_column_idx].strip()
            try:
                x: float = float(x_str) if x_str else DEFAULT_MISSING_VALUE
            except ValueError:
                x: float = DEFAULT_MISSING_VALUE
            x_values.append(x)

            y_row: list[float] = []
            y_row_str: list[str] = row[1:curve_count + 1]
            y_str: str
            for y_str in y_row_str:
                try:
                    y: float = float(y_str) if y_str else DEFAULT_MISSING_VALUE
                except ValueError:
                    y: float = DEFAULT_MISSING_VALUE
                y_row.append(y)
            y_values.append(y_row)

    return cls.create(
        x_values=x_values,
        y_values=y_values,
        labels=labels,
        x_units=x_units,
        x_type=x_type,
        y_units=y_units,
        y_type=y_type,
        path=id,
    )


def _round_or_none(value: float | None) -> float | None:
    """
    Rounds a numeric value to ROUND_PRECISION, passing None through unchanged.

    Parameters:
        value (float | None): the value to round, or None if missing

    Returns:
        float | None: the rounded value, or None if value was None
    """
    return round(value, ROUND_PRECISION) if value is not None else None


def _empty_path_parts() -> dict[str, str]:
    """
    Returns a fresh dict of empty A-F DSS path components.

    Returns:
        dict[str, str]: mapping of each path part letter to an empty string
    """
    return {"A": "", "B": "", "C": "", "D": "", "E": "", "F": ""}


def _id_to_path_parts(dss_id: str | None) -> dict[str, str]:
    """
    Splits a DSS id string into its A-F components, or returns empties if id is missing.

    Parameters:
        dss_id (str | None): the DSS pathname to split, or None

    Returns:
        dict[str, str]: mapping of each path part letter to its value
    """
    if dss_id:
        return DssPath(dss_id).path_to_dict()
    return _empty_path_parts()


def _needs_second_precision(series: RegularTimeSeries | IrregularTimeSeries) -> bool:
    """
    Returns True if any datetime in the series has a non-zero seconds component

    Parameters:
        series: must be either a RegularTimeSeries or IrregularTimeSeries

    Returns:
        bool: whether we need seconds precision or not
    """
    return any(getattr(t, "second", 0) != 0 for t in series.times)


def _get_time_format(raw_time: str) -> str | None:
    """
    Given a raw DSS time string, detect and return the correct time format, whether it be minutes or seconds precision.

    Parameters:
        raw_time (str): time in DSS string format

    Returns:
        str | None: time format to use to convert to datetime 
    """
    dss_minutes_pattern: str = r"^\d{2}[A-Z][a-z]{2}\d{4} \d{4}$"
    dss_seconds_pattern: str = r"^\d{2}[A-Z][a-z]{2}\d{4} \d{6}$"

    if re.fullmatch(dss_minutes_pattern, raw_time):
        return "%d%b%Y %H%M"
    elif re.fullmatch(dss_seconds_pattern, raw_time):
        return "%d%b%Y %H%M%S"
    else:
        return None


def _need_roll_day(time_format: str, raw_time: str) -> bool:
    """
    Primarily to deal with the DSS time sometimes being "2400" and needing to roll over.

    Parameters:
        time_format (str): time format being used
        raw_time (str): raw time string extracted from csv

    Returns: 
        bool: whether or not we need to roll over to the next day
    """
    if time_format != r"%d%b%Y %H%M" and time_format != r"%d%b%Y %H%M%S":
        return False  # Only DSS formats can return True

    # This should correctly catch if we need to roll day
    roll_day_pattern: str = r"^\d{2}[A-Z][a-z]{2}\d{4} 2400(00)?$"

    if re.fullmatch(roll_day_pattern, raw_time):
        return True

    return False


def _prefix_before_dash(s: str) -> str:
    """
    Helper to determine column name for independent column of PairedData.

    Parameters:
        s (str): Input string (c-part of path)

    Returns:
        str: Substring of c-path part that contains independent column name
    """
    return s.split("-", 1)[0] if "-" in s else 'X'
