"""Context object for smoke/dust holding external and derived parameters."""

import datetime as dt
import logging
import logging.config
import os
from enum import unique, StrEnum, IntEnum
from pathlib import Path
from typing import Tuple, List, Union

from mpi4py import MPI
from pydantic import BaseModel, model_validator

from smoke_dust.core.common import open_nc


@unique
class PredefinedGrid(StrEnum):
    """Predefined grids supported by smoke/dust."""

    # pylint: disable=invalid-name
    RRFS_CONUS_25km = "RRFS_CONUS_25km"
    RRFS_CONUS_13km = "RRFS_CONUS_13km"
    RRFS_CONUS_3km = "RRFS_CONUS_3km"
    RRFS_NA_3km = "RRFS_NA_3km"
    RRFS_NA_13km = "RRFS_NA_13km"
    # pyline: enable=invalid-name


@unique
class EbbDCycle(IntEnum):
    """Emission forecast cycle method.

    * `1`: Estimate emissions and fire radiative power.
    * `2`: In addition to `1`, also create inputs to forecast hourly wildfire potential.
    """

    ONE = 1
    TWO = 2


@unique
class RaveQaFilter(StrEnum):
    """
    Quality assurance flag filtering to apply to input RAVE data. RAVE QA filter values range from
    one to three.

    * `NONE`: Do not apply any QA filtering.
    * `HIGH`: QA flag values less than `2` are set to zero for derived fire radiative energy fields.
    """

    NONE = "NONE"
    HIGH = "HIGH"


@unique
class LogLevel(StrEnum):
    """Logging level for the preprocessor."""

    INFO = "INFO"
    DEBUG = "DEBUG"


@unique
class EmissionVariable(StrEnum):
    """Maps RAVE and smoke/dust variable names."""

    FRE = "FRE"
    FRP = "FRP"

    def rave_name(self) -> str:
        """Convert to a RAVE name."""
        other = {self.FRP: "FRP_MEAN", self.FRE: "FRE"}
        return other[self]

    def smoke_dust_name(self) -> str:
        """Convert to a smoke/dust name."""
        other = {self.FRP: "frp_avg_hr", self.FRE: "FRE"}
        return other[self]


class SmokeDustContext(BaseModel):
    """Context object for smoke/dust."""

    # Values provided via command-line
    staticdir: Path
    ravedir: Path
    intp_dir: Path
    predef_grid: PredefinedGrid
    ebb_dcycle_flag: EbbDCycle
    restart_interval: Tuple[int, ...]
    persistence: bool
    rave_qa_filter: RaveQaFilter
    exit_on_error: bool
    log_level: LogLevel

    # Values provided via environment
    current_day: str
    nwges_dir: Path

    # Fixed parameters
    should_calc_desc_stats: bool = False
    vars_emis: tuple[str] = ("FRP_MEAN", "FRE")
    beta: float = 0.3
    fg_to_ug: float = 1e6
    to_s: int = 3600
    rank: int = MPI.COMM_WORLD.Get_rank()
    grid_out_shape: Tuple[int, int] = (0, 0)  # Set in _finalize_model_
    esmpy_debug: bool = False
    regrid_in_memory: bool = False
    _logger: Union[logging.Logger, None] = None

    @model_validator(mode="after")
    def _finalize_model_(self) -> "SmokeDustContext":
        self._logger = self._init_logging_()

        with open_nc(self.grid_out, parallel=False) as ds:
            self.grid_out_shape = (
                ds.dimensions["grid_yt"].size,
                ds.dimensions["grid_xt"].size,
            )
        self.log(f"{self.grid_out_shape=}")
        return self

    @classmethod
    def create_from_args(
        cls, args: List[str], extra: Union[dict | None] = None
    ) -> "SmokeDustContext":
        """
        Create a context object from an arguments list.

        Args:
            args: Source arguments list.
            extra: If provided, pass these keyword arguments to the context constructor.

        Returns:
            A context object.
        """
        print(f"create_from_args: {args=}", flush=True)

        # Extract local arguments from args before converting values
        (
            l_staticdir,
            l_ravedir,
            l_intp_dir,
            l_predef_grid,
            l_ebb_dcycle_flag,
            l_restart_interval,
            l_persistence,
            l_rave_qa_filter,
            l_exit_on_error,
            l_log_level,
        ) = args

        # Format environment-level variables
        current_day: str = os.environ["CDATE"]
        nwges_dir = cls._format_read_path_(os.environ["DATA"])

        # Convert to expected types
        kwds = {
            "staticdir": cls._format_read_path_(l_staticdir),
            "ravedir": cls._format_read_path_(l_ravedir),
            "intp_dir": cls._format_write_path_(l_intp_dir),
            "predef_grid": PredefinedGrid(l_predef_grid),
            "ebb_dcycle_flag": EbbDCycle(int(l_ebb_dcycle_flag)),
            "restart_interval": [int(num) for num in l_restart_interval.split(" ")],
            "persistence": cls._str_to_bool_(l_persistence),
            "rave_qa_filter": RaveQaFilter(l_rave_qa_filter.upper()),
            "exit_on_error": cls._str_to_bool_(l_exit_on_error),
            "log_level": l_log_level.upper(),
            "current_day": current_day,
            "nwges_dir": nwges_dir,
        }

        if extra is not None:
            kwds.update(extra)

        return cls(**kwds)

    @property
    def veg_map(self) -> Path:
        """Path to the vegetation map netCDF file which contains emission factors."""
        return self.staticdir / "veg_map.nc"

    @property
    def rave_to_intp(self) -> str:
        """File prefix for interpolated RAVE files."""
        return self.predef_grid.value + "_intp_"

    @property
    def grid_in(self) -> Path:
        """Path to the grid definition for RAVE data."""
        return self.staticdir / "grid_in.nc"

    @property
    def weightfile(self) -> Path:
        """Path to pre-calculated ESMF weights file mapping the RAVE grid to forecast grid."""
        return self.staticdir / "weight_file.nc"

    @property
    def grid_out(self) -> Path:
        """Path to the forecast grid definition."""
        return self.staticdir / "ds_out_base.nc"

    @property
    def hourly_hwpdir(self) -> Path:
        """Path to the directory containing restart files for `EBB_DCYCLE=2`."""
        return self.nwges_dir / "RESTART"

    @property
    def emissions_path(self) -> Path:
        """Path to the output emissions files containing ICs for smoke/dust."""
        return self.intp_dir / f"SMOKE_RRFS_data_{self.current_day}00.nc"

    @property
    def fcst_datetime(self) -> dt.datetime:
        """The starting datetime object parsed from the `current_day`."""
        return dt.datetime.strptime(self.current_day, "%Y%m%d%H")

    def log(
        self,
        msg,
        level=logging.INFO,
        exc_info: Exception = None,
        stacklevel: int = 2,
    ):
        """
        Log a message.

        Args:
            msg: The message to log.
            level: An optional override for the message level.
            exc_info: If provided, log this exception and raise an error if `self.exit_on_error`
                is `True`.
            stacklevel: If greater than 1, the corresponding number of stack frames are skipped
                when computing the line number and function name.
        """
        if exc_info is not None:
            level = logging.ERROR
        self._logger.log(level, msg, exc_info=exc_info, stacklevel=stacklevel)
        if exc_info is not None and self.exit_on_error:
            raise exc_info

    @staticmethod
    def _format_path_(value: Union[Path, str]) -> Path:
        return Path(value).expanduser().resolve(strict=True)

    @classmethod
    def _format_read_path_(cls, value: str) -> Path:
        path = cls._format_path_(value)
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

    @classmethod
    def _format_write_path_(cls, value: str) -> Path:
        path = cls._format_path_(value)
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
        if value in ["true", "t", "1"]:  # pylint: disable=no-else-return
            return True
        elif value in ["false", "f", "0"]:
            return False
        else:
            raise NotImplementedError(f"boolean string not recognized: {value}")

    def _init_logging_(self) -> logging.Logger:
        project_name = "smoke-dust-preprocessor"

        logging_config: dict = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "plain": {
                    # pylint: disable=line-too-long
                    # Uncomment to report verbose output in logs; try to keep these two in sync
                    # "format": f"[%(name)s][%(levelname)s][%(asctime)s][%(pathname)s:%(lineno)d][%(process)d][%(thread)d][rank={self._rank}]: %(message)s"
                    "format": f"[%(name)s][%(levelname)s][%(asctime)s][%(filename)s:%(lineno)d][rank={self.rank}]: %(message)s"
                    # pylint: enable=line-too-long
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
                    "level": getattr(logging, self.log_level.value),
                },
            },
        }
        logging.config.dictConfig(logging_config)
        return logging.getLogger(project_name)
