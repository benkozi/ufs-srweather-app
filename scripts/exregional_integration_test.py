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
from dataclasses import dataclass
from pathlib import Path

import f90nml

# --------------Define some functions ------------------#

@dataclass
class Config:
    fcst_dir: Path
    fcst_len: int
    fcst_inc: int


class AbstractIntegrationTest(abc.ABC, unittest.TestCase):
    _cfg: Config | None = None

    @classmethod
    def get_config(cls) -> Config:
        if cls._cfg is None:
            raise ValueError
        return cls._cfg

    @classmethod
    def set_config(cls, cfg: Config) -> None:
        cls._cfg = cfg


class TestExptFiles(AbstractIntegrationTest):
    """
    Set up the test for expected output files.
    """

    def test_fcst_files(self):
        """
        Test that expected files exist.
        """

        cfg = self.get_config()

        # Check if model_configure exists
        model_configure_fp = cfg.fcst_dir / "model_configure"
        self.assertTrue(model_configure_fp.exists())

        # Loop through model_configure file to find the netcdf base names
        with open(model_configure_fp, "r", encoding="utf-8") as f:
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

        # Confirm that filenames exist
        for filename in filename_list:
            filename_fp = cfg.fcst_dir / filename
            logging.info(f"Checking existence of: {str(filename_fp)}")
            err_msg = f"Missing file: {str(filename_fp)}"
            self.assertTrue(filename_fp.exists(), err_msg)


class TestUfsFire(AbstractIntegrationTest):
    _namelist_fire: f90nml.Namelist | None = None

    @classmethod
    def setUpClass(cls) -> None:
        namelist_path = cls.get_config().fcst_dir.parent / "namelist.fire"
        cls._namelist_fire = f90nml.read(namelist_path)
        logging.info(f"{cls._namelist_fire=}")
        return cls._namelist_fire

    def get_namelist_fire(self) -> f90nml.Namelist:
        if self._namelist_fire is None:
            raise ValueError
        return self._namelist_fire

    def test_output_files_created(self) -> None:
        cfg = self.get_config()
        fire_files = tuple(cfg.fcst_dir.glob("*fire_output_*nc"))
        n_fire_files = len(fire_files)
        interval_output = self._namelist_fire["time"]["interval_output"]
        logging.info(f"{interval_output=}, {n_fire_files=}")
        n_expected_files = ((cfg.fcst_len * 60 * 60) / interval_output) + 1
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

        namelist_fire = self.get_namelist_fire()
        self.assertEqual(set(namelist_fire.keys()), set(expected_keys.keys()))
        for key in expected_keys.keys():
            # There can be multiple entries for keys suffixed with "1". We are not testing multiple
            # parameter entries here.
            self.assertTrue(set(expected_keys[key]).issubset(set(namelist_fire[key].keys())))


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

    config = Config(fcst_dir=args.fcst_dir, fcst_len=args.fcst_len, fcst_inc=args.fcst_inc)
    logging.info(f"{config=}")

    # Call unittest class
    TestExptFiles.set_config(config)
    suite = unittest.TestSuite()
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestExptFiles))

    if args.test_ufs_fire is True:
        logging.info("adding UFS-Fire tests to the runner")
        TestUfsFire.set_config(config)
        suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestUfsFire))

    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        sys.exit(1)
