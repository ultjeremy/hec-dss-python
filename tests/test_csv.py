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

    def test_to_csv_writes_correct_structure(self):
        # Create a dummy RegularTimeSeries instance
        rts = RegularTimeSeries.create(
            values=[10.5, 20.0],
            times=[datetime(2021, 9, 1, 6, 0), datetime(2021, 9, 1, 12, 0)],
            units="CFS",
            data_type="INST-VAL",
            path="/A/B/C/01Sep2021/6Hour/F/",
        )

        # Mock 'open' and capture written content
        mock_file = mock_open()
        with patch("builtins.open", mock_file):
            rts.to_csv("fake_path.csv", with_metadata=True)

        # Assert that open was called with correct parameters
        mock_file.assert_called_once_with(
            "fake_path.csv", "w", newline="", encoding="utf-8"
        )

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

    def test_to_csv_without_metadata(self):
        """No metadata rows should be written; only data rows."""
        rts = RegularTimeSeries.create(
            values=[1.0, 2.0],
            times=[datetime(2021, 9, 1, 6, 0), datetime(2021, 9, 1, 12, 0)],
            units="CFS",
            data_type="INST-VAL",
            path="/A/B/C//6Hour/F/",
        )
        mock_file = mock_open()
        with patch("builtins.open", mock_file):
            rts.to_csv("fake.csv", with_metadata=False)
        handle = mock_file()
        written = "".join(call.args[0] for call in handle.write.call_args_list)
        self.assertNotIn("Units", written)
        self.assertNotIn("Type,Date/Time", written)
        self.assertIn("1,01Sep2021 0600,1.0", written)

    def test_to_csv_empty_times(self):
        rts = RegularTimeSeries.create(
            values=[],
            times=[],
            units="CFS",
            data_type="INST-VAL",
            path="/A/B/C//6Hour/F/",
        )

        mock_file = mock_open()
        with patch("builtins.open", mock_file):
            rts.to_csv("fake_path.csv", with_metadata=True)

        handle = mock_file()
        written_data = "".join(call.args[0] for call in handle.write.call_args_list)

        self.assertIn("Units,,,CFS", written_data)
        self.assertIn("Type,Date/Time,INST-VAL", written_data)
        self.assertNotIn("1,", written_data)  # No data rows should be present

    def test_to_csv_second_precision(self):
        rts = RegularTimeSeries.create(
            values=[i for i in range(10)],
            times=[datetime(2021, 9, 1, 6, 0, i) for i in range(10)],
            units="CFS",
            data_type="INST-VAL",
            path="/A/B/C//1Second/F/",
        )

        mock_file = mock_open()
        with patch("builtins.open", mock_file):
            rts.to_csv("fake_path.csv", with_metadata=True)

        handle = mock_file()
        written_data = "".join(call.args[0] for call in handle.write.call_args_list)
        self.assertIn("Type,Date/Time,INST-VAL", written_data)
        for i in range(10):
            self.assertIn(f"{i + 1},01Sep2021 06000{i},", written_data)

    def test_to_csv_with_quality(self):
        """When quality is present, header gets 'Quality' col and rows get flags."""
        rts = RegularTimeSeries.create(
            values=[1.0, 2.0],
            times=[datetime(2021, 9, 1, 6, 0), datetime(2021, 9, 1, 12, 0)],
            quality=[0, 5],
            units="CFS",
            data_type="INST-VAL",
            path="/A/B/C//6Hour/F/",
        )
        mock_file = mock_open()
        with patch("builtins.open", mock_file):
            rts.to_csv("fake.csv", with_metadata=True)
        handle = mock_file()
        written = "".join(call.args[0] for call in handle.write.call_args_list)
        self.assertIn("Type,Date/Time,INST-VAL,Quality", written)
        self.assertIn("1,01Sep2021 0600,1.0,0", written)
        self.assertIn("2,01Sep2021 1200,2.0,5", written)

    def read_from_string(self, content):
        """Helper to run read_csv against an in-memory CSV string."""
        m = mock_open(read_data=content)
        with patch("builtins.open", m):
            return RegularTimeSeries.read_csv("fake.csv")

    def test_read_csv_basic(self):
        content = (
            "A,,,A\n"
            "B,,,B\n"
            "C,,,FLOW\n"
            "E,,,6Hour\n"
            "F,,,F\n"
            "Units,,,CFS\n"
            "Type,Date/Time,INST-VAL\n"
            "1,01Sep2021 0600,10.5\n"
            "2,01Sep2021 1200,20.0\n"
        )
        rts = self.read_from_string(content)
        self.assertEqual(rts.units, "CFS")
        self.assertEqual(rts.data_type, "INST-VAL")
        self.assertEqual(rts.values.tolist(), [10.5, 20.0])
        self.assertEqual(rts.times[0], datetime(2021, 9, 1, 6, 0))

    def test_read_csv_midnight_2400_rolls_to_next_day(self):
        content = (
            "E,,,1Day\n"
            "Type,Date/Time,INST-VAL\n"
            "1,31Aug2021 2400,10.5\n"
            "2,01Sep2021 2400,20.0\n"
        )
        rts = self.read_from_string(content)
        self.assertEqual(rts.times, [datetime(2021, 9, 1, 0, 0), datetime(2021, 9, 2, 0, 0)])
        self.assertEqual(rts.values.tolist(), [10.5, 20.0])

    def test_read_csv_with_quality(self):
        content = (
            "Type,Date/Time,INST-VAL,Quality\n"
            "1,01Sep2021 0600,10.5,0\n"
            "2,01Sep2021 1200,20.0,5\n"
        )
        rts = self.read_from_string(content)
        self.assertEqual(rts.values.tolist(), [10.5, 20.0])
        self.assertEqual(rts.quality, [0, 5])

    def test_read_csv_skips_malformed_rows(self):
        content = (
            "Type,Date/Time,INST-VAL\n"
            "1,01Sep2021 0600,10.5\n"
            "2,not-a-date,20.0\n"
            "3,01Sep2021 1200,not-a-number\n"
            "4,01Sep2021 1800,30.0\n"
        )
        rts = self.read_from_string(content)
        self.assertEqual(rts.values.tolist(), [10.5, 30.0])
        self.assertEqual(rts.times, [datetime(2021, 9, 1, 6, 0), datetime(2021, 9, 1, 18, 0)])

    def test_read_csv_empty_file(self):
        rts = self.read_from_string("")
        self.assertEqual(rts.values.tolist(), [])
        self.assertEqual(rts.times, [])

    def test_read_csv_single_row_uses_path_interval(self):
        content = (
            "E,,,6Hour\n"
            "Type,Date/Time,INST-VAL\n"
            "1,01Sep2021 0600,10.5\n"
        )
        rts = self.read_from_string(content)
        self.assertEqual(rts.values.tolist(), [10.5])
        self.assertEqual(rts.times, [datetime(2021, 9, 1, 6, 0)])

    def test_read_csv_irregular_interval_raises(self):
        """RegularTimeSeries.read_csv expects a genuinely regular interval;
        data implying an irregular gap (with no usable E row) is not valid
        input for this class (that's what IrregularTimeSeries is for)."""
        content = (
            "Type,Date/Time,INST-VAL\n"
            "1,01Sep2021 0600,10.5\n"
            "2,01Sep2021 0637,20.0\n"
        )
        with self.assertRaises(ValueError):
            self.read_from_string(content)

    def test_round_trip_basic(self):
        path = self.test_files.create_test_file(".csv")
        rts = RegularTimeSeries.create(
            values=[10.5, 20.0],
            times=[datetime(2021, 9, 1, 6, 0), datetime(2021, 9, 1, 12, 0)],
            units="CFS",
            data_type="INST-VAL",
            path="/A/B/C/01Sep2021/6Hour/F/",
        )
        rts.to_csv(path, with_metadata=True)
        result = RegularTimeSeries.read_csv(path)

        self.assertEqual(result.units, "CFS")
        self.assertEqual(result.data_type, "INST-VAL")
        self.assertEqual(result.interval, "6Hour")
        self.assertEqual(result.id, "/A/B/C//6Hour/F/")
        self.assertEqual(result.values.tolist(), [10.5, 20.0])
        self.assertEqual(
            result.times,
            [datetime(2021, 9, 1, 6, 0), datetime(2021, 9, 1, 12, 0)],
        )

    def test_round_trip_with_quality(self):
        path = self.test_files.create_test_file(".csv")
        rts = RegularTimeSeries.create(
            values=[1.0, 2.0, 3.0],
            times=[
                datetime(2021, 9, 1, 6, 0),
                datetime(2021, 9, 1, 12, 0),
                datetime(2021, 9, 1, 18, 0),
            ],
            quality=[0, 5, 10],
            units="CFS",
            data_type="INST-VAL",
            path="/A/B/C//6Hour/F/",
        )
        rts.to_csv(path, with_metadata=True)
        result = RegularTimeSeries.read_csv(path)

        self.assertEqual(result.values.tolist(), [1.0, 2.0, 3.0])
        self.assertEqual(result.quality, [0, 5, 10])

    def test_round_trip_without_metadata_infers_interval_from_times(self):
        """With no metadata rows, units/data_type come back empty but values,
        times, and the interval/id (inferred from the time deltas) are still
        recovered correctly."""
        path = self.test_files.create_test_file(".csv")
        rts = RegularTimeSeries.create(
            values=[1.0, 2.0],
            times=[datetime(2021, 9, 1, 6, 0), datetime(2021, 9, 1, 12, 0)],
            units="CFS",
            data_type="INST-VAL",
            path="/A/B/C//6Hour/F/",
        )
        rts.to_csv(path, with_metadata=False)
        result = RegularTimeSeries.read_csv(path)

        self.assertEqual(result.units, "")
        self.assertEqual(result.data_type, "")
        self.assertEqual(result.interval, 21600)
        self.assertEqual(result.id, "/////6Hour//")
        self.assertEqual(result.values.tolist(), [1.0, 2.0])
        self.assertEqual(
            result.times,
            [datetime(2021, 9, 1, 6, 0), datetime(2021, 9, 1, 12, 0)],
        )


if __name__ == "__main__":
    unittest.main()
