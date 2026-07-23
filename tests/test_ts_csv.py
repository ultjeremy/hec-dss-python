import unittest
from datetime import datetime
from unittest.mock import mock_open, patch

import numpy as np
from file_manager import FileManager

from hecdss import HecDss
from hecdss.dss_csv import DEFAULT_MISSING_VALUE
from hecdss.irregular_timeseries import IrregularTimeSeries
from hecdss.regular_timeseries import RegularTimeSeries


class TestCSV(unittest.TestCase):
    """
    Tests for the CSV convention:

        Date/Time,Value[,Quality][,Notes],Units,Type,Path
        <time>,<value>[,<qual>][,<note>],<units>,<type>,<path>   <- first data row
        <time>,<value>[,<qual>][,<note>]                          <- later rows

    The Quality/Notes columns appear only when the series carries them.
    Units/Type/Path are always trailing columns, and their values are written
    on the first data row only.
    """

    def setUp(self) -> None:
        self.test_files = FileManager()

    def tearDown(self) -> None:
        self.test_files.cleanup()

    @staticmethod
    def _written(mock_file):
        """Join everything written through a mock_open() handle."""
        handle = mock_file()
        return "".join(call.args[0] for call in handle.write.call_args_list)

    # ================================================================== #
    # REGULAR TIME SERIES -- WRITE
    # ================================================================== #

    def test_to_csv_writes_correct_structure(self):
        rts = RegularTimeSeries.create(
            values=[10.5, 20.0],
            times=[datetime(2021, 9, 1, 6, 0), datetime(2021, 9, 1, 12, 0)],
            units="CFS",
            data_type="INST-VAL",
            path="/A/B/C//6Hour/F/",
        )

        mock_file = mock_open()
        with patch("builtins.open", mock_file):
            rts.to_csv("fake_path.csv")

        mock_file.assert_called_once_with(
            "fake_path.csv", "w", newline="", encoding="utf-8"
        )

        written = self._written(mock_file)
        self.assertIn("Date/Time,Value,Units,Type,Path", written)
        # metadata rides on the first data row
        self.assertIn("01Sep2021 0600,10.5,CFS,INST-VAL,/A/B/C//6Hour/F/", written)
        # later rows carry no metadata
        self.assertIn("01Sep2021 1200,20.0\r\n", written)

    def test_to_csv_metadata_only_on_first_row(self):
        """Units/Type/Path values appear exactly once -- on the first data row."""
        rts = RegularTimeSeries.create(
            values=[1.0, 2.0, 3.0],
            times=[
                datetime(2021, 9, 1, 6, 0),
                datetime(2021, 9, 1, 12, 0),
                datetime(2021, 9, 1, 18, 0),
            ],
            units="CFS",
            data_type="INST-VAL",
            path="/A/B/C//6Hour/F/",
        )
        mock_file = mock_open()
        with patch("builtins.open", mock_file):
            rts.to_csv("fake.csv")
        written = self._written(mock_file)
        # "CFS" appears in the header? No -- header is Units/Type/Path labels.
        # The metadata *values* should appear once total.
        self.assertEqual(written.count("CFS"), 1)
        self.assertEqual(written.count("INST-VAL"), 1)
        self.assertEqual(written.count("/A/B/C//6Hour/F/"), 1)

    def test_to_csv_empty_times(self):
        """An empty series writes just the header (no data rows, no metadata)."""
        rts = RegularTimeSeries.create(
            values=[],
            times=[],
            units="CFS",
            data_type="INST-VAL",
            path="/A/B/C//6Hour/F/",
        )
        mock_file = mock_open()
        with patch("builtins.open", mock_file):
            rts.to_csv("fake_path.csv")
        written = self._written(mock_file)
        self.assertIn("Date/Time,Value,Units,Type,Path", written)
        # No data rows -> the metadata values are never written
        self.assertNotIn("CFS", written)

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
            rts.to_csv("fake_path.csv")
        written = self._written(mock_file)
        self.assertIn("Date/Time,Value,Units,Type,Path", written)
        for i in range(10):
            self.assertIn(f"01Sep2021 06000{i},{i}", written)

    def test_to_csv_with_quality(self):
        """Quality present -> header gets a 'Quality' col and rows get flags."""
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
            rts.to_csv("fake.csv")
        written = self._written(mock_file)
        self.assertIn("Date/Time,Value,Quality,Units,Type,Path", written)
        self.assertIn("01Sep2021 0600,1.0,0,CFS,INST-VAL,/A/B/C//6Hour/F/", written)
        self.assertIn("01Sep2021 1200,2.0,5\r\n", written)

    def test_to_csv_with_notes(self):
        """Notes present (no quality) -> header gets a 'Notes' col."""
        rts = RegularTimeSeries.create(
            values=[1.0, 2.0],
            times=[datetime(2021, 9, 1, 6, 0), datetime(2021, 9, 1, 12, 0)],
            notes=["", "manual override"],
            units="CFS",
            data_type="INST-VAL",
            path="/A/B/C//6Hour/F/",
        )
        mock_file = mock_open()
        with patch("builtins.open", mock_file):
            rts.to_csv("fake.csv")
        written = self._written(mock_file)
        self.assertIn("Date/Time,Value,Notes,Units,Type,Path", written)
        self.assertIn("01Sep2021 0600,1.0,,CFS,INST-VAL,/A/B/C//6Hour/F/", written)
        self.assertIn("01Sep2021 1200,2.0,manual override\r\n", written)

    def test_to_csv_with_quality_and_notes(self):
        """Both present -> Quality column comes before Notes."""
        rts = RegularTimeSeries.create(
            values=[1.0, 2.0],
            times=[datetime(2021, 9, 1, 6, 0), datetime(2021, 9, 1, 12, 0)],
            quality=[0, 5],
            notes=["", "manual override"],
            units="CFS",
            data_type="INST-VAL",
            path="/A/B/C//6Hour/F/",
        )
        mock_file = mock_open()
        with patch("builtins.open", mock_file):
            rts.to_csv("fake.csv")
        written = self._written(mock_file)
        self.assertIn("Date/Time,Value,Quality,Notes,Units,Type,Path", written)
        self.assertIn("01Sep2021 0600,1.0,0,,CFS,INST-VAL,/A/B/C//6Hour/F/", written)
        self.assertIn("01Sep2021 1200,2.0,5,manual override\r\n", written)

    def test_to_csv_with_notes_but_no_quality_omits_quality_column(self):
        """Notes present but quality empty -> only a Notes column, no Quality."""
        rts = RegularTimeSeries.create(
            values=[1.0],
            times=[datetime(2021, 9, 1, 6, 0)],
            notes=["manual override"],
            units="CFS",
            data_type="INST-VAL",
            path="/A/B/C//6Hour/F/",
        )
        mock_file = mock_open()
        with patch("builtins.open", mock_file):
            rts.to_csv("fake.csv")
        written = self._written(mock_file)
        self.assertNotIn("Quality", written)
        self.assertIn("Date/Time,Value,Notes,Units,Type,Path", written)
        self.assertIn("01Sep2021 0600,1.0,manual override,CFS,INST-VAL,/A/B/C//6Hour/F/", written)

    def test_to_csv_writes_empty_cell_for_missing_value(self):
        """A missing (None) value writes as an empty cell, never 'None'."""
        rts = RegularTimeSeries.create(
            values=[1.0, 2.0],
            times=[datetime(2021, 9, 1, 6, 0), datetime(2021, 9, 1, 12, 0)],
            units="CFS",
            data_type="INST-VAL",
            path="/A/B/C//6Hour/F/",
        )
        rts.values = np.array([1.0, None], dtype=object)  # simulate a missing value
        mock_file = mock_open()
        with patch("builtins.open", mock_file):
            rts.to_csv("fake.csv")
        written = self._written(mock_file)
        self.assertIn("01Sep2021 1200,\r\n", written)
        self.assertNotIn("None", written)

    def test_to_csv_writes_empty_cell_for_empty_note(self):
        """An empty-string note writes as an empty cell -- never 'None'."""
        rts = RegularTimeSeries.create(
            values=[1.0, 2.0],
            times=[datetime(2021, 9, 1, 6, 0), datetime(2021, 9, 1, 12, 0)],
            notes=["storm event", ""],
            units="CFS",
            data_type="INST-VAL",
            path="/A/B/C//6Hour/F/",
        )
        mock_file = mock_open()
        with patch("builtins.open", mock_file):
            rts.to_csv("fake.csv")
        written = self._written(mock_file)
        self.assertIn("01Sep2021 1200,2.0,\r\n", written)
        self.assertNotIn("None", written)

    # ================================================================== #
    # REGULAR TIME SERIES -- READ
    # ================================================================== #

    def read_rts_from_string(self, content):
        """Helper to run read_csv against an in-memory CSV string."""
        m = mock_open(read_data=content)
        with patch("builtins.open", m):
            return RegularTimeSeries.read_csv("fake.csv")

    def test_read_csv_basic(self):
        content = (
            "Date/Time,Value,Units,Type,Path\n"
            "01Sep2021 0600,10.5,CFS,INST-VAL,/A/B/C//6Hour/F/\n"
            "01Sep2021 1200,20.0\n"
        )
        rts = self.read_rts_from_string(content)
        self.assertEqual(rts.units, "CFS")
        self.assertEqual(rts.data_type, "INST-VAL")
        self.assertEqual(rts.id, "/A/B/C//6Hour/F/")
        self.assertEqual(rts.values.tolist(), [10.5, 20.0])
        self.assertEqual(rts.times[0], datetime(2021, 9, 1, 6, 0))

    def test_read_csv_midnight_2400_rolls_to_next_day(self):
        content = (
            "Date/Time,Value,Units,Type,Path\n"
            "31Aug2021 2400,10.5,CFS,INST-VAL,/A/B/C//1Day/F/\n"
            "01Sep2021 2400,20.0\n"
        )
        rts = self.read_rts_from_string(content)
        self.assertEqual(
            rts.times, [datetime(2021, 9, 1, 0, 0), datetime(2021, 9, 2, 0, 0)]
        )
        self.assertEqual(rts.values.tolist(), [10.5, 20.0])

    def test_read_csv_with_quality(self):
        content = (
            "Date/Time,Value,Quality,Units,Type,Path\n"
            "01Sep2021 0600,10.5,0,CFS,INST-VAL,/A/B/C//6Hour/F/\n"
            "01Sep2021 1200,20.0,5\n"
        )
        rts = self.read_rts_from_string(content)
        self.assertEqual(rts.values.tolist(), [10.5, 20.0])
        self.assertEqual(rts.quality, [0, 5])

    def test_read_csv_with_partial_quality(self):
        """A row missing its trailing Quality cell defaults that entry to 0."""
        content = (
            "Date/Time,Value,Quality,Units,Type,Path\n"
            "05Nov2004 0200,8,0,CFS,INST-VAL,/A/B/C//1Hour/F/\n"
            "05Nov2004 0300,9\n"  # missing quality!
            "05Nov2004 0400,10,1\n"
        )
        rts = self.read_rts_from_string(content)
        self.assertEqual(rts.values.tolist(), [8, 9, 10])
        self.assertEqual(rts.quality, [0, 0, 1])

    def test_read_csv_with_notes(self):
        content = (
            "Date/Time,Value,Notes,Units,Type,Path\n"
            "01Sep2021 0600,10.5,,CFS,INST-VAL,/A/B/C//6Hour/F/\n"
            "01Sep2021 1200,20.0,manual override\n"
        )
        rts = self.read_rts_from_string(content)
        self.assertEqual(rts.values.tolist(), [10.5, 20.0])
        self.assertEqual(rts.notes, ["", "manual override"])

    def test_read_csv_with_partial_notes(self):
        """A row missing its trailing Notes cell defaults that entry to ''."""
        content = (
            "Date/Time,Value,Notes,Units,Type,Path\n"
            "05Nov2004 0200,8,manual override,CFS,INST-VAL,/A/B/C//1Hour/F/\n"
            "05Nov2004 0300,9\n"  # missing notes cell!
            "05Nov2004 0400,10,estimated\n"
        )
        rts = self.read_rts_from_string(content)
        self.assertEqual(rts.values.tolist(), [8, 9, 10])
        self.assertEqual(rts.notes, ["manual override", "", "estimated"])

    def test_read_csv_notes_and_quality_together(self):
        """Both Quality and Notes columns parse independently (Quality first)."""
        content = (
            "Date/Time,Value,Quality,Notes,Units,Type,Path\n"
            "05Nov2004 0200,8,0,,CFS,INST-VAL,/A/B/C//1Hour/F/\n"
            "05Nov2004 0300,9,1,estimated\n"
        )
        rts = self.read_rts_from_string(content)
        self.assertEqual(rts.values.tolist(), [8, 9])
        self.assertEqual(rts.quality, [0, 1])
        self.assertEqual(rts.notes, ["", "estimated"])

    def test_read_csv_skips_malformed_date_rows(self):
        content = (
            "Date/Time,Value,Units,Type,Path\n"
            "01Sep2021 0600,10.5,CFS,INST-VAL,/A/B/C//6Hour/F/\n"
            "not-a-date,20.0\n"
            "01Sep2021 1200,not-a-number\n"
            "01Sep2021 1800,30.0\n"
        )
        rts = self.read_rts_from_string(content)
        self.assertEqual(rts.values.tolist(), [10.5, DEFAULT_MISSING_VALUE, 30.0])
        self.assertEqual(
            rts.times,
            [
                datetime(2021, 9, 1, 6, 0),
                datetime(2021, 9, 1, 12, 0),
                datetime(2021, 9, 1, 18, 0),
            ],
        )

    def test_read_csv_seconds_precision_basic(self):
        """HHMMSS (seconds-precision) timestamps parse correctly."""
        content = (
            "Date/Time,Value,Units,Type,Path\n"
            "01Sep2021 060000,10.5,CFS,INST-VAL,/A/B/C//15Second/F/\n"
            "01Sep2021 060015,20.0\n"
        )
        rts = self.read_rts_from_string(content)
        self.assertEqual(rts.values.tolist(), [10.5, 20.0])
        self.assertEqual(
            rts.times,
            [datetime(2021, 9, 1, 6, 0, 0), datetime(2021, 9, 1, 6, 0, 15)],
        )

    def test_read_csv_skips_wrong_length_time(self):
        """A clock field that isn't 4 or 6 digits matches neither DSS format."""
        content = (
            "Date/Time,Value,Units,Type,Path\n"
            "01Sep2021 0600,10.5,CFS,INST-VAL,/A/B/C//6Hour/F/\n"
            "01Sep2021 12345,20.0\n"  # 5-digit clock -- not a valid DSS format
            "01Sep2021 1200,30.0\n"
        )
        rts = self.read_rts_from_string(content)
        self.assertEqual(rts.values.tolist(), [10.5, 30.0])
        self.assertEqual(
            rts.times, [datetime(2021, 9, 1, 6, 0), datetime(2021, 9, 1, 12, 0)]
        )

    def test_read_csv_single_digit_day_skipped(self):
        """The format regex requires a zero-padded 2-digit day."""
        content = (
            "Date/Time,Value,Units,Type,Path\n"
            "1Sep2021 0600,10.5,CFS,INST-VAL,/A/B/C//6Hour/F/\n"  # single-digit day
            "01Sep2021 1200,20.0\n"
        )
        rts = self.read_rts_from_string(content)
        self.assertEqual(rts.values.tolist(), [20.0])
        self.assertEqual(rts.times, [datetime(2021, 9, 1, 12, 0)])

    def test_read_csv_month_case_mismatch_skipped(self):
        """The format regex requires a title-case month abbreviation."""
        content = (
            "Date/Time,Value,Units,Type,Path\n"
            "01SEP2021 0600,10.5,CFS,INST-VAL,/A/B/C//6Hour/F/\n"  # all-caps month
            "01sep2021 1200,20.0\n"  # all-lowercase month
            "01Sep2021 1800,30.0\n"
        )
        rts = self.read_rts_from_string(content)
        self.assertEqual(rts.values.tolist(), [30.0])
        self.assertEqual(rts.times, [datetime(2021, 9, 1, 18, 0)])

    def test_read_csv_2400_with_nonzero_seconds_not_treated_as_rollover(self):
        """24:00:15 is not a valid DSS midnight-rollover (only 24:00:00 is)."""
        content = (
            "Date/Time,Value,Units,Type,Path\n"
            "15Sep2021 240015,10.5,CFS,INST-VAL,/A/B/C//1Day/F/\n"
        )
        rts = self.read_rts_from_string(content)
        self.assertEqual(rts.values.tolist(), [])
        self.assertEqual(rts.times, [])

    def test_read_csv_empty_file(self):
        rts = self.read_rts_from_string("")
        self.assertEqual(rts.values.tolist(), [])
        self.assertEqual(rts.times, [])

    def test_read_csv_header_only_no_data_rows(self):
        """A header with no data rows yields an empty series (no crash)."""
        content = "Date/Time,Value,Units,Type,Path\n"
        rts = self.read_rts_from_string(content)
        self.assertEqual(rts.values.tolist(), [])
        self.assertEqual(rts.times, [])
        self.assertEqual(rts.units, "")  # metadata rides on a data row; none here

    def test_read_csv_single_row_uses_path_interval(self):
        content = (
            "Date/Time,Value,Units,Type,Path\n"
            "01Sep2021 0600,10.5,CFS,INST-VAL,/A/B/C//6Hour/F/\n"
        )
        rts = self.read_rts_from_string(content)
        self.assertEqual(rts.values.tolist(), [10.5])
        self.assertEqual(rts.times, [datetime(2021, 9, 1, 6, 0)])

    def test_read_csv_skips_short_data_row(self):
        """A data row with fewer than 2 columns is malformed and skipped."""
        content = (
            "Date/Time,Value,Units,Type,Path\n"
            "01Sep2021 0600\n"  # only 1 column -> skipped
            "01Sep2021 1200,20.0,CFS,INST-VAL,/A/B/C//6Hour/F/\n"
        )
        rts = self.read_rts_from_string(content)
        self.assertEqual(rts.values.tolist(), [20.0])
        self.assertEqual(rts.times, [datetime(2021, 9, 1, 12, 0)])

    def test_read_csv_inconsistent_interval_raises(self):
        """A Path interval that disagrees with the actual time deltas is not
        valid input for RegularTimeSeries (that's IrregularTimeSeries' job)."""
        content = (
            "Date/Time,Value,Units,Type,Path\n"
            "01Sep2021 0600,10.5,CFS,INST-VAL,/A/B/C//6Hour/F/\n"
            "01Sep2021 0637,20.0\n"  # 37-min gap, but Path says 6Hour
        )
        with self.assertRaises(ValueError):
            self.read_rts_from_string(content)

    # ================================================================== #
    # REGULAR TIME SERIES -- ROUND TRIP
    # ================================================================== #

    def test_round_trip_basic(self):
        path = self.test_files.create_test_file(".csv")
        rts = RegularTimeSeries.create(
            values=[10.5, 20.0],
            times=[datetime(2021, 9, 1, 6, 0), datetime(2021, 9, 1, 12, 0)],
            units="CFS",
            data_type="INST-VAL",
            path="/A/B/C//6Hour/F/",
        )
        rts.to_csv(path)
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
        rts.to_csv(path)
        result = RegularTimeSeries.read_csv(path)

        self.assertEqual(result.values.tolist(), [1.0, 2.0, 3.0])
        self.assertEqual(result.quality, [0, 5, 10])

    def test_round_trip_with_notes(self):
        path = self.test_files.create_test_file(".csv")
        rts = RegularTimeSeries.create(
            values=[1.0, 2.0, 3.0],
            times=[
                datetime(2021, 9, 1, 6, 0),
                datetime(2021, 9, 1, 12, 0),
                datetime(2021, 9, 1, 18, 0),
            ],
            notes=["", "storm event", ""],
            units="CFS",
            data_type="INST-VAL",
            path="/A/B/C//6Hour/F/",
        )
        rts.to_csv(path)
        result = RegularTimeSeries.read_csv(path)

        self.assertEqual(result.values.tolist(), [1.0, 2.0, 3.0])
        self.assertEqual(result.notes, ["", "storm event", ""])

    def test_round_trip_with_quality_and_notes(self):
        path = self.test_files.create_test_file(".csv")
        rts = RegularTimeSeries.create(
            values=[1.0, 2.0, 3.0],
            times=[
                datetime(2021, 9, 1, 6, 0),
                datetime(2021, 9, 1, 12, 0),
                datetime(2021, 9, 1, 18, 0),
            ],
            quality=[0, 5, 10],
            notes=["", "storm event", ""],
            units="CFS",
            data_type="INST-VAL",
            path="/A/B/C//6Hour/F/",
        )
        rts.to_csv(path)
        result = RegularTimeSeries.read_csv(path)

        self.assertEqual(result.values.tolist(), [1.0, 2.0, 3.0])
        self.assertEqual(result.quality, [0, 5, 10])
        self.assertEqual(result.notes, ["", "storm event", ""])

    def test_round_trip_notes_containing_comma(self):
        """A note containing a comma is CSV-quoted on write and returns intact."""
        path = self.test_files.create_test_file(".csv")
        rts = RegularTimeSeries.create(
            values=[1.0],
            times=[datetime(2021, 9, 1, 6, 0)],
            notes=["flow, cfs, estimated"],
            units="CFS",
            data_type="INST-VAL",
            path="/A/B/C//6Hour/F/",
        )
        rts.to_csv(path)
        result = RegularTimeSeries.read_csv(path)
        self.assertEqual(result.notes, ["flow, cfs, estimated"])

    def test_round_trip_with_missing_value(self):
        """A missing value survives as DEFAULT_MISSING_VALUE across a round trip."""
        path = self.test_files.create_test_file(".csv")
        rts = RegularTimeSeries.create(
            values=[10.5, 20.0, 30.0],
            times=[
                datetime(2021, 9, 1, 6, 0),
                datetime(2021, 9, 1, 12, 0),
                datetime(2021, 9, 1, 18, 0),
            ],
            units="CFS",
            data_type="INST-VAL",
            path="/A/B/C//6Hour/F/",
        )
        rts.values = np.array([10.5, None, 30.0], dtype=object)
        rts.to_csv(path)
        result = RegularTimeSeries.read_csv(path)
        self.assertEqual(
            result.values.tolist(), [10.5, DEFAULT_MISSING_VALUE, 30.0]
        )

    # ================================================================== #
    # IRREGULAR TIME SERIES -- WRITE
    # ================================================================== #

    def test_basic_to_csv_irregular(self):
        its = IrregularTimeSeries.create(
            values=[10.5, 20.0, 42.0],
            times=[
                datetime(2021, 9, 1, 0, 0),
                datetime(2021, 9, 2, 0, 0),
                datetime(2021, 9, 4, 0, 0),
            ],  # inconsistent time interval
            units="CFS",
            data_type="INST-VAL",
            path="/A/B/C//IR-Year/F/",
        )
        mock_file = mock_open()
        with patch("builtins.open", mock_file):
            its.to_csv("fake_path.csv")

        mock_file.assert_called_once_with(
            "fake_path.csv", "w", newline="", encoding="utf-8"
        )
        written = self._written(mock_file)
        self.assertIn("Date/Time,Value,Units,Type,Path", written)
        self.assertIn("01Sep2021 0000,10.5,CFS,INST-VAL,/A/B/C//IR-Year/F/", written)
        self.assertIn("02Sep2021 0000,20.0\r\n", written)
        self.assertIn("04Sep2021 0000,42.0\r\n", written)

    def test_irregular_to_csv_empty_times(self):
        its = IrregularTimeSeries.create(
            values=[],
            times=[],
            units="CFS",
            data_type="INST-VAL",
            path="/A/B/C//IR-Year/F/",
        )
        mock_file = mock_open()
        with patch("builtins.open", mock_file):
            its.to_csv("fake_path.csv")
        written = self._written(mock_file)
        self.assertIn("Date/Time,Value,Units,Type,Path", written)
        self.assertNotIn("CFS", written)

    def test_irregular_to_csv_second_precision(self):
        its = IrregularTimeSeries.create(
            values=[i for i in range(3)],
            times=[
                datetime(2021, 9, 1, 6, 0, 0),
                datetime(2021, 9, 1, 6, 0, 1),
                datetime(2021, 9, 1, 6, 0, 4),
            ],
            units="CFS",
            data_type="INST-VAL",
            path="/A/B/C//IR-Day/F/",
        )
        mock_file = mock_open()
        with patch("builtins.open", mock_file):
            its.to_csv("fake_path.csv")
        written = self._written(mock_file)
        self.assertIn("Date/Time,Value,Units,Type,Path", written)
        self.assertIn("01Sep2021 060000,0,CFS,INST-VAL,/A/B/C//IR-Day/F/", written)
        self.assertIn("01Sep2021 060001,1\r\n", written)
        self.assertIn("01Sep2021 060004,2\r\n", written)

    def test_irregular_to_csv_with_notes(self):
        its = IrregularTimeSeries.create(
            values=[10.5, 20.0, 42.0],
            times=[
                datetime(2021, 9, 1, 0, 0),
                datetime(2021, 9, 2, 0, 0),
                datetime(2021, 9, 4, 0, 0),
            ],
            notes=["", "storm event", ""],
            units="CFS",
            data_type="INST-VAL",
            path="/A/B/C//IR-Year/F/",
        )
        mock_file = mock_open()
        with patch("builtins.open", mock_file):
            its.to_csv("fake_path.csv")
        written = self._written(mock_file)
        self.assertIn("Date/Time,Value,Notes,Units,Type,Path", written)
        self.assertIn("01Sep2021 0000,10.5,,CFS,INST-VAL,/A/B/C//IR-Year/F/", written)
        self.assertIn("02Sep2021 0000,20.0,storm event\r\n", written)
        self.assertIn("04Sep2021 0000,42.0,\r\n", written)

    # ================================================================== #
    # IRREGULAR TIME SERIES -- READ
    # ================================================================== #

    def read_its_from_string(self, content):
        """Helper to run read_csv against an in-memory CSV string."""
        m = mock_open(read_data=content)
        with patch("builtins.open", m):
            return IrregularTimeSeries.read_csv("fake.csv")

    def test_irregular_read_csv_basic(self):
        content = (
            "Date/Time,Value,Units,Type,Path\n"
            "01Sep2021 0000,10.5,CFS,INST-VAL,/A/B/C//IR-Year/F/\n"
            "02Sep2021 0000,20.0\n"
            "04Sep2021 0000,20.0\n"
        )
        its = self.read_its_from_string(content)
        self.assertEqual(its.units, "CFS")
        self.assertEqual(its.data_type, "INST-VAL")
        self.assertEqual(its.values.tolist(), [10.5, 20.0, 20.0])
        self.assertEqual(
            its.times,
            [datetime(2021, 9, 1), datetime(2021, 9, 2), datetime(2021, 9, 4)],
        )

    def test_irregular_read_csv_with_notes(self):
        content = (
            "Date/Time,Value,Notes,Units,Type,Path\n"
            "01Sep2021 0000,10.5,,CFS,INST-VAL,/A/B/C//IR-Year/F/\n"
            "02Sep2021 0000,20.0,storm event\n"
            "04Sep2021 0000,20.0,\n"
        )
        its = self.read_its_from_string(content)
        self.assertEqual(its.values.tolist(), [10.5, 20.0, 20.0])
        self.assertEqual(its.notes, ["", "storm event", ""])

    def test_roll_day_edge_case(self):
        content = (
            "Date/Time,Value,Units,Type,Path\n"
            "01Sep2021 002400,1,CFS,INST-VAL,/A/B/C//IR-Day/F/\n"  # 12:24 AM
            "02Sep2021 024000,1\n"  # 2:40 AM
            "03Sep2021 240000,1\n"  # 12:00 AM next day
        )
        its = self.read_its_from_string(content)
        self.assertEqual(its.units, "CFS")
        self.assertEqual(its.data_type, "INST-VAL")
        self.assertEqual(its.values.tolist(), [1, 1, 1])
        self.assertEqual(
            its.times,
            [
                datetime(2021, 9, 1, 0, 24, 0),
                datetime(2021, 9, 2, 2, 40, 0),
                datetime(2021, 9, 4, 0, 0, 0),
            ],
        )

    def test_year_is_2400(self):
        content = (
            "Date/Time,Value,Units,Type,Path\n"
            "01Sep2400 000000,1,CFS,INST-VAL,/A/B/C//IR-Day/F/\n"
            "01Sep2400 240000,1\n"
            "01Sep2400 240015,1\n"  # should not be accepted
            "01Sep2400 24000,1\n"  # should not be accepted
        )
        its = self.read_its_from_string(content)
        self.assertEqual(its.units, "CFS")
        self.assertEqual(its.data_type, "INST-VAL")
        self.assertEqual(its.values.tolist(), [1, 1])
        self.assertEqual(
            its.times,
            [datetime(2400, 9, 1, 0, 0, 0), datetime(2400, 9, 2, 0, 0, 0)],
        )

    # ================================================================== #
    # IRREGULAR TIME SERIES -- ROUND TRIP
    # ================================================================== #

    def test_round_trip_irregular(self):
        path = self.test_files.create_test_file(".csv")
        its = IrregularTimeSeries.create(
            values=[10.5, 20.0, 42.0],
            times=[datetime(2021, 9, 1), datetime(2021, 9, 5), datetime(2021, 9, 20)],
            units="CFS",
            data_type="INST-VAL",
            path="/A/B/C//IR-Year/F/",
        )
        its.to_csv(path)
        result = IrregularTimeSeries.read_csv(path)
        self.assertEqual(result.values.tolist(), [10.5, 20.0, 42.0])
        self.assertEqual(
            result.times,
            [datetime(2021, 9, 1), datetime(2021, 9, 5), datetime(2021, 9, 20)],
        )
        self.assertEqual(result.units, "CFS")
        self.assertEqual(result.data_type, "INST-VAL")
        self.assertEqual(result.id, "/A/B/C//IR-Year/F/")

    def test_round_trip_irregular_with_notes(self):
        path = self.test_files.create_test_file(".csv")
        its = IrregularTimeSeries.create(
            values=[10.5, 20.0, 42.0],
            times=[datetime(2021, 9, 1), datetime(2021, 9, 5), datetime(2021, 9, 20)],
            notes=["", "storm event", ""],
            units="CFS",
            data_type="INST-VAL",
            path="/A/B/C//IR-Year/F/",
        )
        its.to_csv(path)
        result = IrregularTimeSeries.read_csv(path)
        self.assertEqual(result.values.tolist(), [10.5, 20.0, 42.0])
        self.assertEqual(result.notes, ["", "storm event", ""])

    def test_round_trip_irregular_with_quality_and_notes(self):
        path = self.test_files.create_test_file(".csv")
        its = IrregularTimeSeries.create(
            values=[10.5, 20.0, 42.0],
            times=[datetime(2021, 9, 1), datetime(2021, 9, 5), datetime(2021, 9, 20)],
            quality=[0, 5, 10],
            notes=["", "storm event", ""],
            units="CFS",
            data_type="INST-VAL",
            path="/A/B/C//IR-Year/F/",
        )
        its.to_csv(path)
        result = IrregularTimeSeries.read_csv(path)
        self.assertEqual(result.values.tolist(), [10.5, 20.0, 42.0])
        self.assertEqual(result.quality, [0, 5, 10])
        self.assertEqual(result.notes, ["", "storm event", ""])


if __name__ == "__main__":
    unittest.main()
