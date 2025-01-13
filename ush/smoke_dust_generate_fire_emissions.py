#!/usr/bin/env python3
import fnmatch
#########################################################################
#                                                                       #
# Python script for fire emissions preprocessing from RAVE FRP and FRE  #
# (Li et al.,2022).                                                     #
# johana.romero-alvarez@noaa.gov                                        #
#                                                                       #
#########################################################################

import os
import sys
from dataclasses import dataclass
from enum import unique, StrEnum, IntEnum
import logging
import logging.config
from pathlib import Path
from typing import Tuple, List

import pandas as pd
from mpi4py import MPI
from pandas import Index

import smoke_dust_fire_emiss_tools as femmi_tools
import smoke_dust_hwp_tools as hwp_tools
import smoke_dust_interp_tools as i_tools

import datetime as dt

from smoke_dust_interpolation import NcToGrid, GridSpec


@unique
class PredefinedGrid(StrEnum):
    RRFS_CONUS_25km = "RRFS_CONUS_25km"
    RRFS_CONUS_13km = "RRFS_CONUS_13km"
    RRFS_CONUS_3km = "RRFS_CONUS_3km"
    RRFS_NA_3km = "RRFS_NA_3km"
    RRFS_NA_13km = "RRFS_NA_13km"


@unique
class EbbDCycle(IntEnum):
    ONE = 1
    TWO = 2


# @unique
# class RaveQaFilter(StrEnum):
#     NONE = "NONE"
#     HIGH = "HIGH"


@unique
class LogLevel(StrEnum):
    INFO = "INFO"
    DEBUG = "DEBUG"


@dataclass
class SmokeDustContext:
    staticdir: Path
    ravedir: Path
    intp_dir: Path
    predef_grid: PredefinedGrid
    ebb_dcycle_flag: EbbDCycle
    restart_interval: Tuple[int, ...]
    persistence: bool
    exit_on_error: bool
    log_level: LogLevel
    # rave_qa_flag_filter: RaveQaFilter

    current_day: str
    nwges_dir: Path

    beta: float = 0.3
    fg_to_ug: float = 1e6
    to_s: int = 3600
    vars_emis = ["FRP_MEAN", "FRE"]

    @property
    def veg_map(self) -> Path:
        return self.staticdir / "veg_map.nc"

    @property
    def rave_to_intp(self) -> str:
        return self.predef_grid.value + "_intp_"

    @property
    def grid_in(self) -> Path:
        return self.staticdir / "grid_in.nc"

    @property
    def weightfile(self) -> Path:
        return self.staticdir / "weight_file.nc"

    @property
    def grid_out(self) -> Path:
        return self.staticdir / "ds_out_base.nc"

    @property
    def hourly_hwpdir(self) -> Path:
        return self.nwges_dir / "hourly_hwpdir.nc"

    @classmethod
    def create_from_args(cls, args: List[str]) -> "SmokeDustContext":
        print(f"create_from_args:args={args}", flush=True)

        # Extract local arguments from args before converting values
        (
            l_staticdir,
            l_ravedir,
            l_intp_dir,
            l_predef_grid,
            l_ebb_dcycle_flag,
            l_restart_interval,
            l_persistence,
            l_exit_on_error,
            l_log_level,
        ) = args

        # Format environment-level variables
        current_day: str = os.environ["CDATE"]
        nwges_dir = cls._format_read_path_(os.environ["DATA"])

        # Convert to expected types
        kwds = dict(
            staticdir=cls._format_read_path_(l_staticdir),
            ravedir=cls._format_read_path_(l_ravedir),
            intp_dir=cls._format_write_path_(l_intp_dir),
            predef_grid=PredefinedGrid(l_predef_grid),
            ebb_dcycle_flag=EbbDCycle(int(l_ebb_dcycle_flag)),
            restart_interval=[int(num) for num in l_restart_interval.split(" ")],
            persistence=cls._str_to_bool_(l_persistence),
            exit_on_error=cls._str_to_bool_(l_exit_on_error),
            log_level=getattr(logging, l_log_level),
            current_day=current_day,
            nwges_dir=nwges_dir,
        )

        return cls(**kwds)

    @staticmethod
    def _format_read_path_(value: str) -> Path:
        path = Path(value)
        errors = []
        if not path.exists():
            errors.append(f"path does not exist: {path}")
        if not os.access(path, os.R_OK):
            errors.append(f"path is not readable: {path}")
        if not path.is_dir():
            errors.append(f"path is not a directory: {path}")
        if len(errors) > 0:
            raise OSError(errors)
        return path

    @staticmethod
    def _format_write_path_(value: str) -> Path:
        path = Path(value)
        errors = []
        if not path.exists():
            errors.append(f"path does not exist: {path}")
        if not os.access(path, os.W_OK):
            errors.append(f"path is not writable: {path}")
        if not path.is_dir():
            errors.append(f"path is not a directory: {path}")
        if len(errors) > 0:
            raise OSError(errors)
        return path

    @staticmethod
    def _str_to_bool_(value: str) -> bool:
        value = value.lower()
        if value in ["true", "t", "1"]:
            return True
        elif value in ["false", "f", "0"]:
            return False
        raise NotImplementedError(f"boolean string not recognized: {value}")


class SmokeDustPreprocessor:

    def __init__(self, args: List[str]) -> None:
        self._context = SmokeDustContext.create_from_args(args)
        self._logger = self._init_logging_()
        self.log(f"initialization complete. context={self._context}")

        self._forecast_dates = None
        # self._intp_avail_hours = None
        self._forecast_metadata = None

    @property
    def forecast_dates(self) -> pd.DatetimeIndex:
        if self._forecast_dates is not None:
            return self._forecast_dates

        fcst_datetime = dt.datetime.strptime(self._context.current_day, "%Y%m%d%H")
        match self._context.ebb_dcycle_flag:
            case EbbDCycle.ONE:
                if self._context.persistence:
                    self.log("Creating emissions for persistence method where satellite FRP persist from previous day")
                    start_datetime = fcst_datetime - dt.timedelta(days=1)
                else:
                    self.log("Creating emissions using current date satellite FRP")
                    start_datetime = fcst_datetime
            case EbbDCycle.TWO:
                self.log("Creating emissions for modulated persistence by Wildfire potential")
                start_datetime = fcst_datetime - dt.timedelta(days=1, hours=1)
            case _:
                raise NotImplementedError(self._context.ebb_dcycle_flag)
        forecast_dates = pd.date_range(start=start_datetime, periods=24, freq="h").strftime(
                    "%Y%m%d%H"
                )
        self.log(f"forecast_dates={forecast_dates}", level=logging.DEBUG)
        self._forecast_dates = forecast_dates
        return self._forecast_dates

    # @property
    # def intp_avail_hours(self) -> pd.DatetimeIndex:
    #     if self._intp_avail_hours is not None: #tdk:rm
    #         return self._intp_avail_hours
    #
    #     intp_avail_hours = []
    #     for date in self.forecast_dates:
    #         file_path = Path(self._context.intp_dir) / f"{self._context.rave_to_intp}{date}00_{date}59.nc"
    #         if file_path.exists() and file_path.is_file():
    #             try:
    #                 _ = file_path.resolve(strict=True)
    #             except FileNotFoundError:
    #                 continue
    #             else:
    #                 intp_avail_hours.append(date)
    #     self._intp_avail_hours = pd.DatetimeIndex(intp_avail_hours)
    #     self.log(
    #         f"Available interpolated files for hours: {self._intp_avail_hours}"
    #     )
    #     self.log(f"Non-available interpolated files for hours: {self.intp_non_avail_hours}")
    #     return self._intp_avail_hours
    #
    # @property
    # def intp_non_avail_hours(self) -> pd.DatetimeIndex:
    #     return self.forecast_dates[~self.forecast_dates.isin(self.intp_avail_hours)]

    @property
    def forecast_metadata(self) -> pd.DataFrame:
        if self._forecast_metadata is not None:
            return self._forecast_metadata

        intp_path = []
        rave_to_forecast = []
        for date in self.forecast_dates:
            # Check for pre-existing interpolated RAVE data
            file_path = Path(self._context.intp_dir) / f"{self._context.rave_to_intp}{date}00_{date}59.nc"
            if file_path.exists() and file_path.is_file():
                try:
                    resolved = file_path.resolve(strict=True)
                except FileNotFoundError:
                    continue
                else:
                    intp_path.append(resolved)
            else:
                intp_path.append(None)

            # Check for raw RAVE data
            wildcard_name = f"*-3km*{date}*{date}59590*.nc"
            name_retro = f"*3km*{date}*{date}*.nc" #tdk:ja: what is this for?
            found = False
            for rave_path in self._context.ravedir.iterdir():
                if fnmatch.fnmatch(str(rave_path), wildcard_name) or fnmatch.fnmatch(str(rave_path), name_retro):
                    rave_to_forecast.append(rave_path)
                    found = True
                    break
            if not found:
                rave_to_forecast.append(None)

        df = pd.DataFrame(data={'forecast_dates': self.forecast_dates,'rave_interpolated': intp_path, 'rave_raw': rave_to_forecast})
        return df

    @property
    def is_first_day(self) -> bool:
        return self.forecast_metadata['rave_interpolated'].isnull().all() and self.forecast_metadata['rave_raw'].isnull().all()

    def run(self) -> None:
        self.log(f"is_first_day={self.is_first_day}")
        if self.is_first_day:
            #tdk: implement creation of dummy emissions file
            raise NotImplementedError("is_first_day is not yet implemented")
        else:
            self.log("creating source grid from RAVE file")
            src_nc2grid = NcToGrid(
                path=self._context.grid_in,
                spec=GridSpec(
                    x_center="grid_lont",
                    y_center="grid_latt",
                    x_dim=("grid_xt",),
                    y_dim=("grid_yt",),
                    x_corner="grid_lon",
                    y_corner="grid_lat",
                    x_corner_dim=("grid_x",),
                    y_corner_dim=("grid_y",),
                ),
            )
            src_gwrap = src_nc2grid.create_grid_wrapper()
            import pdb;pdb.set_trace()

    def finalize(self) -> None:
        raise NotImplementedError

    def log(self,
            msg,
            level=logging.INFO,
            exc_info: Exception = None,
            stacklevel: int = 2,
    ):
        if exc_info is not None:
            level = logging.ERROR
        self._logger.log(level, msg, exc_info=exc_info, stacklevel=stacklevel)
        if exc_info is not None and self._context.exit_on_error:
            raise exc_info

    def _init_logging_(self) -> logging.Logger:
        project_name = "smoke-dust-preprocessor"
        rank = MPI.COMM_WORLD.Get_rank()
        logging_config: dict = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "plain": {
                    "format": f"[%(name)s][%(levelname)s][%(asctime)s][%(pathname)s:%(lineno)d][%(process)d][%(thread)d][{rank}]: %(message)s"
                },
            },
            "handlers": {
                "default": {
                    "formatter": "plain",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",
                    "filters": [],
                },
            },
            "loggers": {
                project_name: {
                    "handlers": ["default"],
                    "level": self._context.log_level,
                },
            },
        }
        logging.config.dictConfig(logging_config)
        return logging.getLogger(project_name)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Workflow
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def generate_emiss_workflow(
    args: List[str]
    # staticdir: str, #tdk:rm
    # ravedir: str,
    # intp_dir: str,
    # predef_grid: str,
    # ebb_dcycle_flag: str,
    # restart_interval: str,
    # persistence: str,
) -> None:
    """
    Prepares fire-related ICs. This is the main function that handles data movement and interpolation.

    Args:
        staticdir: Path to fix files for the smoke and dust component
        ravedir: Path to the directory containing RAVE fire data files (hourly). This is typically the working directory (DATA)
        intp_dir: Path to interpolated RAVE data files from the previous cycles (DATA_SHARE)
        predef_grid: If ``RRFS_NA_3km``, use pre-defined grid dimensions
        ebb_dcycle_flag: Select the EBB cycle to run. Valid values are ``"1"`` or ``"2"``
        restart_interval: Indicates if restart files should be copied. The actual interval values are not used
        persistence: If ``TRUE``, use satellite observations from the previous day. Otherwise, use observations from the same day.
    """

    processor = SmokeDustPreprocessor(args)
    try:
        # _ = processor.intp_non_avail_hours #tdk:rm
        # fm = processor.forecast_metadata
        # import pdb;pdb.set_trace()
        # tdk
        processor.run()
    except Exception as e:
        processor.log("unhandled error", exc_info=e)

    import pdb;pdb.set_trace()

    # ----------------------------------------------------------------------
    # Import envs from workflow and get the pre-defined grid
    # Set variable names, constants and unit conversions
    # Set predefined grid
    # Set directories
    # ----------------------------------------------------------------------

    beta = 0.3
    fg_to_ug = 1e6
    to_s = 3600
    current_day = os.environ["CDATE"]
    #   nwges_dir = os.environ.get("NWGES_DIR")
    nwges_dir = os.environ["DATA"]
    vars_emis = ["FRP_MEAN", "FRE"]
    # tdk: need dimensions for all grids
    # cols, rows = (2700, 3950) if predef_grid == "RRFS_NA_3km" else (1092, 1820)
    if predef_grid == "RRFS_NA_3km":
        cols, rows = 2700, 3950
    elif predef_grid == "RRFS_CONUS_3km":
        cols, rows = 1092, 1820
    elif predef_grid == "RRFS_CONUS_25km":
        cols, rows = 131, 219
    elif predef_grid == "RRFS_CONUS_13km":
        cols, rows = 252, 420
    elif predef_grid == "RRFS_NA_13km":
        cols, rows = 623, 912
    else:
        raise NotImplementedError(f"Unknown predefined grid type: {predef_grid}")
    print("PREDEF GRID", predef_grid, "cols,rows", cols, rows)
    # used later when working with ebb_dcyle 1 or 2
    ebb_dcycle = int(ebb_dcycle_flag)
    print(
        "WARNING, EBB_DCYCLE set to",
        ebb_dcycle,
        "and persistence=",
        persistence,
        "if persistence is false, emissions comes from same day satellite obs",
    )

    print("CDATE:", current_day)
    print("DATA:", nwges_dir)

    # This is used later when copying the rrfs restart file
    restart_interval_list = [float(num) for num in restart_interval.split()]
    len_restart_interval = len(restart_interval_list)

    # Setting the directories
    veg_map = staticdir + "/veg_map.nc"
    RAVE = ravedir
    rave_to_intp = predef_grid + "_intp_"
    grid_in = staticdir + "/grid_in.nc"
    weightfile = staticdir + "/weight_file.nc"
    grid_out = staticdir + "/ds_out_base.nc"
    hourly_hwpdir = os.path.join(nwges_dir, "RESTART")

    # ----------------------------------------------------------------------
    # Workflow
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # Sort raw RAVE, create source and target filelds, and compute emissions
    # ----------------------------------------------------------------------

    # fcst_dates = i_tools.date_range(current_day, ebb_dcycle, persistence)
    # intp_avail_hours, intp_non_avail_hours, inp_files_2use = (
    #     i_tools.check_for_intp_rave(intp_dir, fcst_dates, rave_to_intp)
    # )
    # rave_avail, rave_avail_hours, rave_nonavail_hours_test, first_day = (
    #     i_tools.check_for_raw_rave(RAVE, intp_non_avail_hours, intp_avail_hours)
    # )
    srcfield, tgtfield, tgt_latt, tgt_lont, srcgrid, tgtgrid, src_latt, tgt_area = (
        i_tools.creates_st_fields(grid_in, grid_out)
    )

    if not first_day:
        regridder, use_dummy_emiss = i_tools.generate_regridder(
            rave_avail_hours, srcfield, tgtfield, weightfile, intp_avail_hours
        )
        if use_dummy_emiss:
            print("RAVE files corrupted, no data to process")
            i_tools.create_dummy(intp_dir, current_day, tgt_latt, tgt_lont, cols, rows)
        else:
            i_tools.interpolate_rave(
                RAVE,
                rave_avail,
                rave_avail_hours,
                use_dummy_emiss,
                vars_emis,
                regridder,
                srcgrid,
                tgtgrid,
                rave_to_intp,
                intp_dir,
                tgt_latt,
                tgt_lont,
                cols,
                rows,
            )

            if ebb_dcycle == 1:
                print("Processing emissions forebb_dcyc 1")
                frp_avg_reshaped, ebb_total_reshaped = femmi_tools.averaging_FRP(
                    ebb_dcycle,
                    fcst_dates,
                    cols,
                    rows,
                    intp_dir,
                    rave_to_intp,
                    veg_map,
                    tgt_area,
                    beta,
                    fg_to_ug,
                    to_s,
                )
                femmi_tools.produce_emiss_24hr_file(
                    frp_avg_reshaped,
                    nwges_dir,
                    current_day,
                    tgt_latt,
                    tgt_lont,
                    ebb_total_reshaped,
                    cols,
                    rows,
                )
            elif ebb_dcycle == 2:
                print("Restart dates to process", fcst_dates)
                hwp_avail_hours, hwp_non_avail_hours = hwp_tools.check_restart_files(
                    hourly_hwpdir, fcst_dates
                )
                restart_avail, restart_nonavail_hours_test = (
                    hwp_tools.copy_missing_restart(
                        nwges_dir,
                        hwp_non_avail_hours,
                        hourly_hwpdir,
                        len_restart_interval,
                    )
                )
                hwp_ave_arr, xarr_hwp, totprcp_ave_arr, xarr_totprcp = (
                    hwp_tools.process_hwp(
                        fcst_dates, hourly_hwpdir, cols, rows, intp_dir, rave_to_intp
                    )
                )
                frp_avg_reshaped, ebb_total_reshaped = femmi_tools.averaging_FRP(
                    ebb_dcycle,
                    fcst_dates,
                    cols,
                    rows,
                    intp_dir,
                    rave_to_intp,
                    veg_map,
                    tgt_area,
                    beta,
                    fg_to_ug,
                    to_s,
                )
                # Fire end hours processing
                te = femmi_tools.estimate_fire_duration(
                    intp_dir, fcst_dates, current_day, cols, rows, rave_to_intp
                )
                fire_age = femmi_tools.save_fire_dur(cols, rows, te)
                # produce emiss file
                femmi_tools.produce_emiss_file(
                    xarr_hwp,
                    frp_avg_reshaped,
                    totprcp_ave_arr,
                    xarr_totprcp,
                    nwges_dir,
                    current_day,
                    tgt_latt,
                    tgt_lont,
                    ebb_total_reshaped,
                    fire_age,
                    cols,
                    rows,
                )
            else:
                raise NotImplementedError(f"ebb_dcycle={ebb_dcycle}")
    else:
        print("First day true, no RAVE files available. Use dummy emissions file")
        i_tools.create_dummy(intp_dir, current_day, tgt_latt, tgt_lont, cols, rows)


if __name__ == "__main__":
    print("")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Welcome to interpolating RAVE and processing fire emissions!")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("")
    # generate_emiss_workflow( #tdk:rm
    #     sys.argv[1],
    #     sys.argv[2],
    #     sys.argv[3],
    #     sys.argv[4],
    #     sys.argv[5],
    #     sys.argv[6],
    #     sys.argv[7],
    # )
    generate_emiss_workflow(
        sys.argv[1:]
    )
    print("")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Successful Completion. Bye!")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("")
