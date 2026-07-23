# import pandas as pd
import math
import unittest
from unittest.mock import mock_open, patch

from file_manager import FileManager

from hecdss import HecDss
from hecdss.dss_csv import (
    DEFAULT_MISSING_VALUE,
    _prefix_before_dash,
    paired_data_read_csv,
    paired_data_to_csv,
)
from hecdss.paired_data import PairedData
from hecdss.regular_timeseries import RegularTimeSeries


class TestCSV(unittest.TestCase):

    def setUp(self) -> None:
        self.test_files = FileManager()

    def tearDown(self) -> None:
        self.test_files.cleanup()

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def read_pd_from_string(self, content: str) -> PairedData:
        """
        Helper to run read_csv against an in-memory CSV string.

        Parameters:
            content (str): PairedData CSV string

        Returns:
            PairedData: PairedData object read from string CSV
        """
        m = mock_open(read_data=content)
        with patch("builtins.open", m):
            return PairedData.read_csv("fake.csv")

    def write_pd_to_string(self, pd: PairedData) -> str:
        """
        Helper to run to_csv against a mocked file and return everything written.

        Parameters:
            pd (PairedData): the object to export

        Returns:
            str: the full CSV text that to_csv wrote
        """
        mock_file = mock_open()
        with patch("builtins.open", mock_file):
            pd.to_csv("fake_path.csv")
        handle = mock_file()
        return "".join(call.args[0] for call in handle.write.call_args_list)

    @staticmethod
    def make_pd(**overrides) -> PairedData:
        """A small, fully-populated PairedData with fields overridable per test."""
        params = dict(
            x_values=[1.0, 2.0],
            y_values=[[10.0, 100.0], [20.0, 200.0]],
            labels=["L1", "L2"],
            x_units="FEET",
            x_type="UNT",
            y_units="CFS",
            y_type="LINEAR",
            path="/A/B/C/DATE/E/F/",
        )
        params.update(overrides)
        return PairedData.create(**params)

    # ================================================================== #
    # to_csv  (write)
    # ================================================================== #

    def test_multiple_curves_to_csv(self):
        """
        This test ensures that the data values for a PairedData object with multiple curves are correct.
        This has no checks related to units or labels.
        """
        num_curves: int = 3
        x_values: list[float] = [1.0, 1.5, 4.0]
        y_values: list[list[float]] = [[x * i for i in range(num_curves)] for x in x_values]
        labels: list[str] = [f"x times {i}" for i in range(num_curves)]
        path: str = "/A/B/C///Source: I made it up/"
        pd: PairedData = PairedData.create(
            x_values=x_values,
            y_values=y_values,
            labels=labels,
            path=path
        )

        mock_file = mock_open()
        with patch("builtins.open", mock_file):
            pd.to_csv("fake_path.csv")

        mock_file.assert_called_once_with(
            "fake_path.csv", "w", newline="", encoding="utf-8"
        )

        handle = mock_file()
        written_data = "".join(call.args[0] for call in handle.write.call_args_list)

        # C part "C" has no dash, so the x column name falls back to "X".
        self.assertIn(
            "X,x times 0,x times 1,x times 2,X Units,Y Units,X Type,Y Type,Path", written_data)
        # The full path is preserved verbatim in the first data row.
        self.assertIn("/A/B/C///Source: I made it up/", written_data)
        # Data rows: x then the row-major y values (no leading counter).
        self.assertIn("1.0,0.0,1.0,2.0", written_data)
        self.assertIn("1.5,0.0,1.5,3.0", written_data)
        self.assertIn("4.0,0.0,4.0,8.0", written_data)

    def test_to_csv_metadata_present(self):
        """The header carries labels + column names; the first data row carries the
        units, types, and full path."""
        written = self.write_pd_to_string(self.make_pd())
        self.assertIn("X,L1,L2,X Units,Y Units,X Type,Y Type,Path", written)
        self.assertIn("FEET,CFS,UNT,LINEAR,/A/B/C/DATE/E/F/", written)

    def test_to_csv_preserves_full_path(self):
        """The whole DSS path (including the D part) is written to the Path cell."""
        pd = self.make_pd(path="/A/B/C/DPART/EPART/FPART/")
        written = self.write_pd_to_string(pd)
        self.assertIn("/A/B/C/DPART/EPART/FPART/", written)

    def test_to_csv_data_rows(self):
        """Data rows are x then the row-major y values."""
        written = self.write_pd_to_string(self.make_pd())
        self.assertIn("1.0,10.0,100.0", written)
        self.assertIn("2.0,20.0,200.0", written)

    def test_to_csv_has_no_counter_column(self):
        """There is no leading index column; a data row starts with the x value."""
        pd = self.make_pd(
            x_values=[5.0, 6.0, 7.0],
            y_values=[[1.0], [2.0], [3.0]],
        )
        written = self.write_pd_to_string(pd)
        self.assertIn("5.0,1.0", written)
        self.assertIn("6.0,2.0", written)
        self.assertIn("7.0,3.0", written)
        # A leading "1,5.0" counter would look like this - make sure it is absent.
        self.assertNotIn("1,5.0", written)

    def test_to_csv_single_curve(self):
        """A single dependent column round-trips through the writer fine."""
        pd = self.make_pd(y_values=[[10.0], [20.0]], labels=["only"])
        written = self.write_pd_to_string(pd)
        self.assertIn("X,only,X Units,Y Units,X Type,Y Type,Path", written)
        self.assertIn("1.0,10.0", written)
        self.assertIn("2.0,20.0", written)

    def test_to_csv_rounds_to_four_decimals(self):
        """x and y are rounded to ROUND_PRECISION (4) decimal places."""
        pd = self.make_pd(x_values=[1.23456], y_values=[[9.87654]], labels=["a"])
        written = self.write_pd_to_string(pd)
        self.assertIn("1.2346,9.8765", written)

    def test_to_csv_negative_values(self):
        """Negative x/y values are written verbatim."""
        pd = self.make_pd(x_values=[-1.0], y_values=[[-5.5, -0.25]], labels=["a", "b"])
        written = self.write_pd_to_string(pd)
        self.assertIn("-1.0,-5.5,-0.25", written)

    def test_to_csv_empty_ordinates_writes_no_data_rows(self):
        """No ordinates -> only the header is written. Because units/types/path ride
        on the first data row, they are absent when there are no data rows."""
        # Labels avoid digits so the "no data row" check can't collide with a label.
        pd = self.make_pd(x_values=[], y_values=[], labels=["aa", "bb"])
        written = self.write_pd_to_string(pd)
        self.assertIn("X,aa,bb,X Units,Y Units,X Type,Y Type,Path", written)
        self.assertNotIn("FEET", written)   # no data row -> no metadata written
        self.assertNotIn("1.0", written)    # no data rows

    def test_to_csv_none_id_writes_empty_path(self):
        """A None id writes an empty Path cell rather than raising or writing 'None'."""
        pd = self.make_pd(path=None)
        written = self.write_pd_to_string(pd)
        # metadata tail is present, with an empty trailing Path cell
        self.assertIn("FEET,CFS,UNT,LINEAR,", written)
        self.assertNotIn("None", written)

    def test_to_csv_x_column_name_from_c_part(self):
        """The x column name in the header is the C path part up to its first dash."""
        pd = self.make_pd(path="/A/B/STAGE-FLOW/D/E/F/")
        written = self.write_pd_to_string(pd)
        self.assertIn("STAGE,L1,L2,X Units", written)

    def test_to_csv_x_column_name_defaults_to_X(self):
        """A C part with no dash (or an empty C part) falls back to 'X'."""
        no_dash = self.write_pd_to_string(self.make_pd(path="/A/B/NODASH/D/E/F/"))
        self.assertIn("X,L1,L2,X Units", no_dash)

        empty_c = self.write_pd_to_string(self.make_pd(path="/A/B//D/E/F/"))
        self.assertIn("X,L1,L2,X Units", empty_c)

    def test_prefix_before_dash_helper(self):
        """The x-column-name helper takes the substring before the first dash."""
        self.assertEqual(_prefix_before_dash("STAGE-FLOW"), "STAGE")
        self.assertEqual(_prefix_before_dash("FREQ-FLOW"), "FREQ")
        self.assertEqual(_prefix_before_dash("A-B-C"), "A")
        self.assertEqual(_prefix_before_dash("NODASH"), "X")
        self.assertEqual(_prefix_before_dash(""), "X")

    def test_to_csv_raises_on_non_paireddata(self):
        """paired_data_to_csv rejects anything that isn't a PairedData."""
        with self.assertRaises(TypeError):
            paired_data_to_csv("not a paired data", "fake.csv")

    # ================================================================== #
    # read_csv  (read)
    # ================================================================== #

    def test_basic_read_csv(self):
        content: str = (
            "FREQ,IMAGINARY,X,Y,Z,X Units,Y Units,X Type,Y Type,Path\n"
            "0,1,5,6,8,DAYS,PPG,TIME,POINTS,/paired-data-test/berkeley/FREQ-FLOW///Source: I made it up/\n"
            "1,3,5,6,9\n"
            "5,1,2,3,4\n"
        )
        pd: PairedData = self.read_pd_from_string(content)
        self.assertEqual(pd.labels, ['IMAGINARY', 'X', 'Y', 'Z'])
        self.assertEqual(pd.units_independent, "DAYS")
        self.assertEqual(pd.units_dependent, "PPG")
        self.assertEqual(pd.type_independent, 'TIME')
        self.assertEqual(pd.type_dependent, 'POINTS')
        self.assertEqual(pd.ordinates.tolist(), [0.0, 1.0, 5.0])
        self.assertEqual(pd.values.tolist(), [
            [1.0, 5.0, 6.0, 8.0],
            [3.0, 5.0, 6.0, 9.0],
            [1.0, 2.0, 3.0, 4.0]
        ])

    def test_read_csv_single_curve(self):
        content = (
            "FLOW,only,X Units,Y Units,X Type,Y Type,Path\n"
            "0,10\n"
            "1,20\n"
        )
        pd = self.read_pd_from_string(content)
        self.assertEqual(pd.ordinates.tolist(), [0.0, 1.0])
        self.assertEqual(pd.values.tolist(), [[10.0], [20.0]])
        self.assertEqual(pd.curve_count(), 1)

    def test_read_csv_no_header_raises(self):
        """Data rows with no header row (nothing ending in 'Path') are rejected."""
        content = (
            "1,0,10,100\n"
            "2,1,20,200\n"
        )
        with self.assertRaises(ValueError):
            self.read_pd_from_string(content)

    def test_read_csv_reads_full_path(self):
        """The id is read verbatim from the Path cell of the first data row,
        including its D and E parts."""
        content = (
            "STAGE,only,X Units,Y Units,X Type,Y Type,Path\n"
            "0,10,STAGE,FLOW,,,/SITE/LOC/STAGE-FLOW/DPART/EPART/MADEUP/\n"
        )
        pd = self.read_pd_from_string(content)
        self.assertEqual(pd.id, "/SITE/LOC/STAGE-FLOW/DPART/EPART/MADEUP/")

    def test_read_csv_curve_count(self):
        content = (
            "TIME,a,b,c,X Units,Y Units,X Type,Y Type,Path\n"
            "0,1,2,3\n"
            "1,4,5,6\n"
        )
        pd = self.read_pd_from_string(content)
        self.assertEqual(pd.curve_count(), 3)

    def test_read_csv_gets_labels_from_header(self):
        """Labels come from the header columns between the x column and 'X Units'."""
        content = (
            "STAGE,alpha,beta,X Units,Y Units,X Type,Y Type,Path\n"
            "0,1,2\n"
        )
        pd = self.read_pd_from_string(content)
        self.assertEqual(pd.labels, ["alpha", "beta"])
        self.assertEqual(pd.curve_count(), 2)

    def test_read_csv_curve_count_independent_of_labels(self):
        """Curve count is derived from the 'X Units' boundary, not the label count,
        so a header with blank label cells still reads every curve."""
        content = (
            "X,,,X Units,Y Units,X Type,Y Type,Path\n"
            "0,10,100\n"
            "1,20,200\n"
        )
        pd = self.read_pd_from_string(content)
        self.assertEqual(pd.curve_count(), 2)
        self.assertEqual(pd.values.tolist(), [[10.0, 100.0], [20.0, 200.0]])
        self.assertEqual(pd.labels, ["", ""])

    def test_read_csv_malformed_x_defaults_missing(self):
        """A non-numeric x cell falls back to the missing-value default."""
        content = (
            "X,a,b,X Units,Y Units,X Type,Y Type,Path\n"
            "not-a-number,10,100\n"
            "2,20,200\n"
        )
        pd = self.read_pd_from_string(content)
        self.assertEqual(pd.ordinates.tolist(), [DEFAULT_MISSING_VALUE, 2.0])
        self.assertEqual(pd.values.tolist(), [[10.0, 100.0], [20.0, 200.0]])

    def test_read_csv_malformed_y_defaults_missing(self):
        """A non-numeric y cell falls back to the default WITHOUT shortening the row,
        so the value matrix stays rectangular."""
        content = (
            "X,a,b,X Units,Y Units,X Type,Y Type,Path\n"
            "1,bad,3\n"
            "2,5,6\n"
        )
        pd = self.read_pd_from_string(content)
        self.assertEqual(pd.values.tolist(), [[DEFAULT_MISSING_VALUE, 3.0], [5.0, 6.0]])

    def test_read_csv_empty_cells_default_missing(self):
        """Blank x and y cells become the missing-value default."""
        content = (
            "X,a,X Units,Y Units,X Type,Y Type,Path\n"
            ",\n"
            "2,\n"
        )
        pd = self.read_pd_from_string(content)
        self.assertEqual(pd.ordinates.tolist(), [DEFAULT_MISSING_VALUE, 2.0])
        self.assertEqual(pd.values.tolist(), [[DEFAULT_MISSING_VALUE], [DEFAULT_MISSING_VALUE]])

    def test_read_csv_skips_short_rows(self):
        """A data row with fewer than 2 columns is malformed and skipped."""
        content = (
            "X,a,X Units,Y Units,X Type,Y Type,Path\n"
            "5\n"          # only 1 column -> skipped
            "2,9\n"
        )
        pd = self.read_pd_from_string(content)
        self.assertEqual(pd.ordinates.tolist(), [2.0])
        self.assertEqual(pd.values.tolist(), [[9.0]])

    def test_read_csv_strips_numeric_whitespace(self):
        """Surrounding whitespace around numeric cells is stripped before parsing."""
        content = (
            "X,a,b,X Units,Y Units,X Type,Y Type,Path\n"
            " 1 , 2 , 3 \n"
        )
        pd = self.read_pd_from_string(content)
        self.assertEqual(pd.ordinates.tolist(), [1.0])
        self.assertEqual(pd.values.tolist(), [[2.0, 3.0]])

    def test_read_csv_labels_preserve_whitespace(self):
        """Labels are captured verbatim from the header (not stripped)."""
        content = (
            "FREQ,  spacey  ,X,X Units,Y Units,X Type,Y Type,Path\n"
            "0,1,2\n"
        )
        pd = self.read_pd_from_string(content)
        self.assertEqual(pd.labels, ["  spacey  ", "X"])

    def test_read_csv_partial_units(self):
        """A first data row missing the dependent-units cell leaves y_units empty."""
        content = (
            "DAY,a,X Units,Y Units,X Type,Y Type,Path\n"
            "0,10,DAYS\n"
        )
        pd = self.read_pd_from_string(content)
        self.assertEqual(pd.units_independent, "DAYS")
        self.assertEqual(pd.units_dependent, "")

    def test_read_csv_partial_type(self):
        """A first data row missing the dependent-type cell leaves y_type empty."""
        content = (
            "TIME,a,X Units,Y Units,X Type,Y Type,Path\n"
            "0,10,,,TIME\n"
        )
        pd = self.read_pd_from_string(content)
        self.assertEqual(pd.type_independent, "TIME")
        self.assertEqual(pd.type_dependent, "")

    def test_read_csv_no_metadata_cells_leaves_fields_empty(self):
        """A first data row carrying only x and y leaves the metadata fields empty."""
        content = (
            "X,a,X Units,Y Units,X Type,Y Type,Path\n"
            "0,10\n"
        )
        pd = self.read_pd_from_string(content)
        self.assertEqual(pd.units_independent, "")
        self.assertEqual(pd.units_dependent, "")
        self.assertEqual(pd.ordinates.tolist(), [0.0])

    def test_read_csv_empty_file(self):
        pd = self.read_pd_from_string("")
        self.assertEqual(pd.ordinates.tolist(), [])
        self.assertEqual(pd.values.tolist(), [])
        self.assertEqual(pd.labels, [])

    def test_read_csv_blank_lines_ignored(self):
        content = (
            "X,a,X Units,Y Units,X Type,Y Type,Path\n"
            "\n"
            "0,10\n"
            "\n"
            "1,20\n"
        )
        pd = self.read_pd_from_string(content)
        self.assertEqual(pd.ordinates.tolist(), [0.0, 1.0])
        self.assertEqual(pd.values.tolist(), [[10.0], [20.0]])

    def test_read_csv_ragged_rows_raise(self):
        """A later data row whose width disagrees with the curve count is rejected."""
        content = (
            "X,a,b,X Units,Y Units,X Type,Y Type,Path\n"
            "0,1,2\n"      # 2 y-values
            "1,3\n"        # 1 y-value
        )
        with self.assertRaises(ValueError):
            self.read_pd_from_string(content)

    def test_read_csv_wrong_cls_raises(self):
        """paired_data_read_csv only builds PairedData; other classes are rejected."""
        with self.assertRaises(TypeError):
            paired_data_read_csv(RegularTimeSeries, "fake.csv")

    # ================================================================== #
    # round trips  (write -> read on a real temp file)
    # ================================================================== #

    def test_round_trip_basic(self):
        path = self.test_files.create_test_file(".csv")
        pd = self.make_pd()
        pd.to_csv(path)
        result = PairedData.read_csv(path)

        self.assertEqual(result.ordinates.tolist(), [1.0, 2.0])
        self.assertEqual(result.values.tolist(), [[10.0, 100.0], [20.0, 200.0]])
        self.assertEqual(result.labels, ["L1", "L2"])
        self.assertEqual(result.units_independent, "FEET")
        self.assertEqual(result.units_dependent, "CFS")
        self.assertEqual(result.type_independent, "UNT")
        self.assertEqual(result.type_dependent, "LINEAR")
        # The full path (including the D part) survives the round trip.
        self.assertEqual(result.id, "/A/B/C/DATE/E/F/")

    def test_round_trip_multiple_curves(self):
        path = self.test_files.create_test_file(".csv")
        pd = self.make_pd(
            x_values=[0.0, 1.0, 2.0],
            y_values=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            labels=["a", "b", "c"],
        )
        pd.to_csv(path)
        result = PairedData.read_csv(path)

        self.assertEqual(result.curve_count(), 3)
        self.assertEqual(
            result.values.tolist(),
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        )
        self.assertEqual(result.labels, ["a", "b", "c"])

    def test_round_trip_no_labels(self):
        """A PairedData written with no labels still round-trips its curves; the
        header pads a blank label column per curve so the count survives."""
        path = self.test_files.create_test_file(".csv")
        pd = self.make_pd(labels=[])
        pd.to_csv(path)
        result = PairedData.read_csv(path)

        self.assertEqual(result.curve_count(), 2)
        self.assertEqual(result.values.tolist(), [[10.0, 100.0], [20.0, 200.0]])
        self.assertEqual(result.ordinates.tolist(), [1.0, 2.0])
        # Padded blank labels come back, one per curve.
        self.assertEqual(result.labels, ["", ""])

    def test_round_trip_rounding(self):
        """Values are rounded to 4 decimals on write, and that rounded value is
        exactly what is read back."""
        path = self.test_files.create_test_file(".csv")
        pd = self.make_pd(x_values=[1.23456], y_values=[[2.34567]], labels=["a"])
        pd.to_csv(path)
        result = PairedData.read_csv(path)

        self.assertEqual(result.ordinates.tolist(), [1.2346])
        self.assertEqual(result.values.tolist(), [[2.3457]])

    def test_round_trip_negative_values(self):
        path = self.test_files.create_test_file(".csv")
        pd = self.make_pd(
            x_values=[-1.0, 0.0, 1.0],
            y_values=[[-5.5], [0.0], [5.5]],
            labels=["a"],
        )
        pd.to_csv(path)
        result = PairedData.read_csv(path)

        self.assertEqual(result.ordinates.tolist(), [-1.0, 0.0, 1.0])
        self.assertEqual(result.values.tolist(), [[-5.5], [0.0], [5.5]])

    def test_round_trip_x_column_name(self):
        """The x column name written from the C part is exactly what appears as the
        first header cell (it is not stored on PairedData, only in the CSV header)."""
        path = self.test_files.create_test_file(".csv")
        pd = self.make_pd(path="/A/B/STAGE-FLOW/D/E/F/")
        pd.to_csv(path)
        with open(path, "r", encoding="utf-8") as f:
            first_line = f.readline()
        self.assertTrue(first_line.startswith("STAGE,"))

    # ================================================================== #
    # edge cases
    # ================================================================== #

    def test_read_csv_quoted_label_with_comma_round_trips(self):
        """A label containing a comma is CSV-quoted on write and comes back intact."""
        path = self.test_files.create_test_file(".csv")
        pd = self.make_pd(
            x_values=[1.0], y_values=[[2.0]], labels=["flow, cfs"],
        )
        pd.to_csv(path)
        result = PairedData.read_csv(path)
        self.assertEqual(result.labels, ["flow, cfs"])

    def test_read_csv_metadata_only_from_first_data_row(self):
        """Metadata is positional: it is read from the first data row (the one right
        after the header), and later rows need not repeat it."""
        content = (
            "X,a,X Units,Y Units,X Type,Y Type,Path\n"
            "0,10,DAYS,PPG,TIME,POINTS,/the/id/\n"
            "1,20\n"
        )
        pd = self.read_pd_from_string(content)
        self.assertEqual(pd.units_independent, "DAYS")
        self.assertEqual(pd.units_dependent, "PPG")
        self.assertEqual(pd.type_independent, "TIME")
        self.assertEqual(pd.type_dependent, "POINTS")
        self.assertEqual(pd.id, "/the/id/")
        self.assertEqual(pd.values.tolist(), [[10.0], [20.0]])

    def test_read_csv_whitespace_only_cells_default_missing(self):
        """Cells that are only whitespace strip to empty and become the default."""
        content = (
            "X,a,X Units,Y Units,X Type,Y Type,Path\n"
            "   ,  \n"
        )
        pd = self.read_pd_from_string(content)
        self.assertEqual(pd.ordinates.tolist(), [DEFAULT_MISSING_VALUE])
        self.assertEqual(pd.values.tolist(), [[DEFAULT_MISSING_VALUE]])

    def test_read_csv_inf_and_nan_parse_through(self):
        """float() accepts 'inf'/'nan', so these land in the array as real inf/NaN
        rather than being treated as missing. Surprising, given the missing-value
        default - documented here so a future change is caught."""
        content = (
            "X,a,X Units,Y Units,X Type,Y Type,Path\n"
            "inf,nan\n"
            "-inf,5\n"
        )
        pd = self.read_pd_from_string(content)
        self.assertTrue(math.isinf(pd.ordinates[0]))
        self.assertTrue(math.isnan(pd.values[0][0]))
        self.assertTrue(math.isinf(pd.ordinates[1]) and pd.ordinates[1] < 0)

    def test_read_csv_miscased_path_header_not_detected(self):
        """FOOTGUN: header detection is case-sensitive on 'Path'. A mis-cased 'path'
        is not recognized as the header, so the reader sees a data row before any
        header and raises."""
        content = (
            "X,a,X Units,Y Units,X Type,Y Type,path\n"   # lowercase -> not a header
            "0,10\n"
        )
        with self.assertRaises(ValueError):
            self.read_pd_from_string(content)

    def test_read_csv_id_from_path_column_not_last_cell(self):
        """The id is read from the 'Path' column position, not the last cell, so a
        stray trailing column on the first data row does not hijack the id."""
        content = (
            "X,a,X Units,Y Units,X Type,Y Type,Path\n"
            "0,10,U,V,XT,YT,/the/id/,EXTRA\n"
        )
        pd = self.read_pd_from_string(content)
        self.assertEqual(pd.id, "/the/id/")
        self.assertNotIn("EXTRA", pd.id or "")

    def test_curve_count_on_dataless_read_is_zero(self):
        """A header with no data rows yields an empty PairedData, and curve_count()
        returns 0 (it guards against an empty value array)."""
        content = (
            "X,a,b,X Units,Y Units,X Type,Y Type,Path\n"
        )
        pd = self.read_pd_from_string(content)
        self.assertEqual(pd.curve_count(), 0)

    def test_to_csv_missing_value_does_not_crash(self):
        """A blank ordinate is read back as DEFAULT_MISSING_VALUE (None), producing
        an object-dtype array. The writer must tolerate a None ordinate (write an
        empty cell) rather than crash trying to round it."""
        pd = self.read_pd_from_string(
            "X,a,X Units,Y Units,X Type,Y Type,Path\n"
            ",10\n"         # blank ordinate -> None
            "2,20\n"
        )
        written = self.write_pd_to_string(pd)
        self.assertIn(",10.0", written)      # first row: empty x, then y
        self.assertIn("2.0,20.0", written)

    def test_round_trip_nan_value_survives(self):
        """A NaN dependent value is written as 'nan' and read back as a real NaN
        (it is NOT collapsed into the missing-value default)."""
        path = self.test_files.create_test_file(".csv")
        pd = self.make_pd(x_values=[1.0], y_values=[[float("nan")]], labels=["a"])
        pd.to_csv(path)
        result = PairedData.read_csv(path)
        self.assertTrue(math.isnan(result.values.tolist()[0][0]))


if __name__ == "__main__":
    unittest.main()
