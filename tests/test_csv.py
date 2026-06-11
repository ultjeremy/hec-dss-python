import unittest
from unittest.mock import patch, mock_open
from file_manager import FileManager
from hecdss import HecDss
from hecdss.regular_timeseries import RegularTimeSeries
from datetime import datetime

class TestCSV(unittest.TestCase):

    def setUp(self) -> None:
        self.test_files = FileManager()

    def tearDown(self) -> None:
        self.test_files.cleanup()

    """def test_to_csv(self):
        path = "/regular-time-series/GAPT/FLOW/01Sep2021 - 31Oct2021/6Hour/forecast1/"
        with HecDss(self.test_files.get_copy("examples-all-data-types.dss")) as dss:
            rts: RegularTimeSeries = dss.get(path)
            rts.to_csv("test.csv", with_metadata=True)
            rts.to_csv("test_no_metadata.csv", with_metadata=False)

    def test_read_csv(self):
        regular_time_series = RegularTimeSeries.read_csv("test.csv")
        assert regular_time_series is not None
        self.assertEqual(regular_time_series.units, "cfs")
        self.assertEqual(regular_time_series.data_type, "INST-VAL")
        self.assertEqual(regular_time_series.times[0], datetime(2021, 9, 15, 7, 0))
        regular_time_series_no_metadata = RegularTimeSeries.read_csv("test_no_metadata.csv")
        assert regular_time_series_no_metadata is not None
        self.assertEqual(regular_time_series.times[0], datetime(2021, 9, 15, 7, 0))
        self.assertEqual(regular_time_series_no_metadata.units, "")
        self.assertIsNotNone(regular_time_series_no_metadata.quality)

    def test_read_24_hour_time(self):
        '''
        Test that a time of 24:00 is correctly parsed as the end of the day.
        '''
        regular_time_series = RegularTimeSeries.read_csv("24_hour_test.csv")
        regular_time_series.print_to_console()"""


    def test_to_csv_writes_correct_structure(self):
        # Create a dummy RegularTimeSeries instance
        rts = RegularTimeSeries.create(
            values=[10.5, 20.0],
            times=[datetime(2021, 9, 1, 6, 0), datetime(2021, 9, 1, 12, 0)],
            units="CFS",
            data_type="INST-VAL",
            path="/A/B/C/01Sep2021/6Hour/F/"
        )

        # Mock 'open' and capture written content
        mock_file = mock_open()
        with patch("builtins.open", mock_file):
            rts.to_csv("fake_path.csv", with_metadata=True)

        # Assert that open was called with correct parameters
        mock_file.assert_called_once_with("fake_path.csv", "w", newline="", encoding="utf-8")

        # Extract all written data
        handle = mock_file()
        written_data = "".join(call.args[0] for call in handle.write.call_args_list)

        # Assertions on the CSV content structure
        self.assertIn("A,,,A", written_data)
        self.assertIn("B,,,B", written_data)
        self.assertIn("C,,,C", written_data)
        self.assertIn("Units,,,CFS", written_data)
        self.assertIn("Type,Date/Time,INST-VAL", written_data)
        self.assertIn("1,01Sep2021 0600,10.5", written_data)
        self.assertIn("2,01Sep2021 1200,20.0", written_data)


if __name__ == "__main__":
    unittest.main()
