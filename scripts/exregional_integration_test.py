#!/usr/bin/env python3

"""
Python Script Documentation Block

 Script name:       	exregional_integration_test.py
 Script description:  	Ensures the correct number of netcdf files are generated
 			for each experiment

 Author:  Eddie Snyder 	Org: NOAA EPIC		Date: 2024-02-05

 Instructions:		1. Pass the appropriate info for the required arguments:
                              --fcst_dir=/path/to/forecast/files
                              --fcst_len=<forecast length as Int>
                       2. Run script with arguments

 Notes/future work:    - Currently SRW App only accepts netcdf as the UFS WM
                         output file format. If that changes, then additional
                         logic is needed to address the other file formats.
                       - SRW App doesn't have a variable that updates the
                         forecast increment. The UFS WM does with the
                         output_fh variable, which can be found in the
                         model_configure file. If it becomes available with
                         the SRW App, then logic is needed to account for the
                         forecast increment variable.

"""

# -------------Import modules --------------------------#
import abc
import argparse
import logging
import sys
import unittest
from functools import cached_property
from pathlib import Path

import f90nml

# --------------Define some functions ------------------#

class AbstractIntegrationTest(abc.ABC, unittest.TestCase):
    fcst_dir = ""


class TestExptFiles(AbstractIntegrationTest):
    """
    Set up the test for expected output files.
    """
    filename_list = ""

    def test_fcst_files(self):
        """
        Test that expected files exist.
        """
        for filename in self.filename_list:
            filename_fp = self.fcst_dir / filename
            logging.info(f"Checking existence of: {str(filename_fp)}")
            err_msg = f"Missing file: {str(filename_fp)}"
            self.assertTrue(filename_fp.exists(), err_msg)


class TestUfsFire(AbstractIntegrationTest):
    fcst_len: int | None = None
    namelist_fire: f90nml.Namelist | None = None

    @classmethod
    def setUpClass(cls):
        namelist_path = cls.fcst_dir.parent / "namelist.fire"
        cls.namelist_fire = f90nml.read(namelist_path)
        logging.info(f"{cls.namelist_fire=}")
        return cls.namelist_fire

    def test_fire_output_files_created(self) -> None:
        assert isinstance(self.fcst_dir, Path)
        fire_files = tuple(self.fcst_dir.glob("*fire_output_*nc"))
        n_fire_files = len(fire_files)
        interval_output = self.namelist_fire["time"]["interval_output"]
        logging.info(f"{self.fcst_len=}, {interval_output=}, {n_fire_files=}")
        n_expected_files = ((self.fcst_len * 60 * 60) / interval_output) + 1
        self.assertEqual(n_fire_files, n_expected_files)

    def test_namelist_created(self) -> None:
        expected_keys = {'time': ('dt', 'interval_output'), 'atm': ('interval_atm', 'kde'),
                         'fire': ('fire_num_ignitions', 'fire_ignition_ros1',
                                  'fire_ignition_start_lat1', 'fire_ignition_start_lon1',
                                  'fire_ignition_end_lat1', 'fire_ignition_end_lon1',
                                  'fire_ignition_radius1', 'fire_ignition_start_time1',
                                  'fire_ignition_end_time1', 'fire_wind_height',
                                  'fire_print_msg', 'fire_atm_feedback', 'fire_viscosity',
                                  'fire_upwinding', 'fire_lsm_zcoupling',
                                  'fire_lsm_zcoupling_ref')}
        actual_keys = {}
        for k in self.namelist_fire.keys():
            actual_keys[k] = tuple(self.namelist_fire[k].keys())
        self.assertEqual(set(actual_keys.keys()), set(expected_keys.keys()))
        for key in expected_keys.keys():
            self.assertEqual(set(actual_keys[key]), set(expected_keys[key]))
        self.assertTrue(False)


def setup_logging(debug=False):
    """Calls initialization functions for logging package, and sets the
    user-defined level for logging in the script."""

    level = logging.INFO
    if debug:
        level = logging.DEBUG

    logging.basicConfig(format="%(levelname)s: %(message)s ", level=level)
    if debug:
        logging.info("Logging level set to DEBUG")


# -------------Start of script -------------------------#
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fcst_dir",
        help="Directory to forecast files.",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--fcst_len",
        help="Forecast length.",
        required=True,
        type=int,
    )
    parser.add_argument(
        "--fcst_inc",
        default=1,
        help="Increment of forecast in hours.",
        required=False,
        type=int,
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print debug messages.",
        required=False,
    )
    parser.add_argument(
        "--test_ufs_fire",
        default=False,
        action="store_true",
        help="If true, run UFS-Fire tests.",
        required=False,
    )
    parser.add_argument("unittest_args", nargs="*")
    args = parser.parse_args()
    sys.argv[1:] = args.unittest_args

    # Start logger
    setup_logging()

    # Check if model_configure exists
    MODEL_CONFIGURE_FP = args.fcst_dir / "model_configure"

    if not MODEL_CONFIGURE_FP.is_file():
        logging.error("Experiment's model_configure file is missing! Exiting!")
        sys.exit(1)

    # Loop through model_configure file to find the netcdf base names
    with open(MODEL_CONFIGURE_FP, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("filename_base"):
                filename_base_1 = line.split("'")[1]
                filename_base_2 = line.split("'")[3]
                break

    # Create list of expected filenames from the experiment
    filename_list = []

    for x in range(0, args.fcst_len + 1, args.fcst_inc):
        fhour = str(x).zfill(3)
        filename_1 = f"{filename_base_1}f{fhour}.nc"
        filename_2 = f"{filename_base_2}f{fhour}.nc"
        filename_list.append(filename_1)
        filename_list.append(filename_2)

    # Call unittest class
    TestExptFiles.fcst_dir = args.fcst_dir
    TestExptFiles.filename_list = filename_list
    suite = unittest.TestSuite()
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestExptFiles))

    if args.test_ufs_fire is True:
        logging.info("adding UFS-Fire tests to the runner")
        TestUfsFire.fcst_dir = args.fcst_dir
        TestUfsFire.fcst_len = args.fcst_len
        suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestUfsFire))

    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        sys.exit(1)
