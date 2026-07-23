from datetime import datetime

import numpy as np


class IrregularTimeSeries:
    """ container for time-series data that is not at a consistent interval.
    data is stored internally as a numpy array
    """

    def __init__(self):
        """
        Initialize an IrregularTimeSeries object with default values.
        """

        self.times = []
        self.values = np.empty(0)
        self.quality = []
        self.notes: list[str] = []
        self.units = ""
        self.data_type = ""
        self.interval = 0
        self.start_date = ""
        self.time_granularity_seconds = 1
        self.julian_base_date = 0
        self.time_zone_name = ""
        self.id = ""
        self.location_info = None

    def add_data_point(self, date, value, flag=None, note=None):
        """
        append a date,value to this time-series
        """

        self.times.append(date)
        self.values.append(value)
        if flag is not None:
            self.quality.append(flag)
        if note is not None:
            self.notes.append(note)

    def get_value_at(self, date):
        """
         Retrieve the value at a specific date in the time-series.

         Parameters:
         date (datetime): The date for which to retrieve the value.

         Returns:
         float or None: The value at the specified date if it exists, otherwise None.
         """
        if date in self.times:
            index = self.times.index(date)
            return self.values[index]
        else:
            return None

    def get_values(self):
        """
        Retrieve all values in the time-series.

        Returns:
        numpy.ndarray: An array of all values in the time-series.
        """
        return self.values

    def get_dates(self):
        """
        Retrieve all dates in the time-series.

        Returns:
        list of datetime: A list of all dates in the time-series.
        """
        return self.times

    def get_length(self):
        """
        Retrieve the number of data points in the time-series.

        Returns:
        int: The number of data points in the time-series.
        """
        return len(self.times)

    def print_to_console(self):
        """
        Print the time-series data to the console in a readable format.
        """
        print("dsspath='" + self.id + "'")
        print("units='" + self.units + "'")
        print("dataType='" + self.data_type + "'")
        has_quality = len(self.quality) > 0
        has_notes = len(self.notes) > 0
        for i, (time, value) in enumerate(zip(self.times, self.values)):
            line = f"Time: {time}, Value: {value}"
            if has_quality:
                line += f", Flag: {self.quality[i]}"
            if has_notes:
                line += f", Note: {self.notes[i]}"
            print(line)

    def to_csv(self, file_path: str, with_metadata: bool = True) -> None:
        """
        Exports the IrregularTimeSeries object to a .csv file.

        Parameters:
            file_path (str): The path to the .csv file where the data will be exported.
            with_metadata (bool): Whether to include metadata in the exported file.
        """
        from .dss_csv import timeseries_to_csv
        timeseries_to_csv(self, file_path, with_metadata)
        print(f"Wrote IrregularTimeSeries to .csv file at {file_path}.")

    @staticmethod
    def read_csv(file_path: str) -> "IrregularTimeSeries":
        """
        Reads a .csv file and creates an IrregularTimeSeries instance from the data.

        Parameters:
            file_path (str): The path to the .csv file to read

        Returns:
            IrregularTimeSeries: A new instance of IrregularTimeSeries populated with the data from the .csv file.
        """
        from .dss_csv import timeseries_read_csv
        return timeseries_read_csv(IrregularTimeSeries, file_path)

    @staticmethod
    def create(values, times, quality=[], notes=[], units="", data_type="", interval=0, start_date="", time_granularity_seconds=1, julian_base_date=None, time_zone_name="", path=None, location_info=None):
        """
         Retrieve the value at a specific date in the time-series.

         Parameters:
         date (datetime): The date for which to retrieve the value.

         Returns:
         float or None: The value at the specified date if it exists, otherwise None.
         """
        irts = IrregularTimeSeries()
        irts.times = times
        irts.values = np.array(values)
        irts.quality = quality
        irts.notes = notes
        irts.units = units
        irts.data_type = data_type
        irts.interval = interval
        irts.start_date = start_date
        irts.time_granularity_seconds = time_granularity_seconds
        irts.julian_base_date = 0
        if julian_base_date is None and len(times):
            irts.julian_base_date = (times[0] - datetime(1900, 1, 1)).days
        irts.time_zone_name = time_zone_name
        irts.id = path
        irts.location_info = location_info
        return irts
