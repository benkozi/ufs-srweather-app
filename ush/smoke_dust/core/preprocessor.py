"""Smoke/dust preprocessor core implementation."""

import fnmatch
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from smoke_dust.core.common import (
    open_nc,
    create_template_emissions_file,
    create_sd_variable,
)
from smoke_dust.core.context import SmokeDustContext
from smoke_dust.core.cycle import create_cycle_processor
from smoke_dust.core.regrid.processor import SmokeDustRegridProcessor
from smoke_dust.core.variable import SD_VARS


class SmokeDustPreprocessor:
    """Implements smoke/dust preprocessing such as regridding and IC value calculations."""

    def __init__(self, context: SmokeDustContext) -> None:
        self._context = context
        self.log("__init__: enter")

        # Processes regridding from source data to destination analysis grid
        self._regrid_processor = SmokeDustRegridProcessor(context)
        # Processes cycle-specific data transformations
        self._cycle_processor = create_cycle_processor(context)

        # On-demand/cached property values
        self._forecast_metadata = None
        self._forecast_dates = None

        self.log(f"{self._context=}")
        self.log("__init__: exit")

    def log(self, *args: Any, **kwargs: Any) -> None:
        """See `SmokeDustContext.log`."""
        self._context.log(*args, **kwargs)

    @property
    def forecast_dates(self) -> pd.DatetimeIndex:
        """Create the forecast dates for cycle."""
        if self._forecast_dates is not None:
            return self._forecast_dates
        start_datetime = self._cycle_processor.create_start_datetime()
        self.log(f"{start_datetime=}")
        forecast_dates = pd.date_range(start=start_datetime, periods=24, freq="h").strftime(
            "%Y%m%d%H"
        )
        self._forecast_dates = forecast_dates
        return self._forecast_dates

    @property
    def forecast_metadata(self) -> pd.DataFrame:
        """Create forecast metadata consisting of:

        * `forecast_date`: The forecast timestep as a `datetime` object.
        * `rave_interpolated`: To the date's corresponding interpolated RAVE file. Null if not
            found.
        * `rave_raw`: Raw RAVE data before interpolation. Null if not found.
        """
        if self._forecast_metadata is not None:
            return self._forecast_metadata

        # Collect metadata on data files related to forecast dates
        self.log("creating forecast metadata")
        intp_path = []
        rave_to_forecast = []
        for date in self.forecast_dates:
            # Check for pre-existing interpolated RAVE data
            file_path = (
                Path(self._context.intp_dir) / f"{self._context.rave_to_intp}{date}00_{date}59.nc"
            )
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
            name_retro = f"*3km*{date}*{date}*.nc"
            found = False
            for rave_path in self._context.ravedir.iterdir():
                if fnmatch.fnmatch(str(rave_path), wildcard_name) or fnmatch.fnmatch(
                    str(rave_path), name_retro
                ):
                    rave_to_forecast.append(rave_path)
                    found = True
                    break
            if not found:
                rave_to_forecast.append(None)

        self.log(f"{self.forecast_dates}", level=logging.DEBUG)
        self.log(f"{intp_path=}", level=logging.DEBUG)
        self.log(f"{rave_to_forecast=}", level=logging.DEBUG)
        df = pd.DataFrame(
            data={
                "forecast_date": self.forecast_dates,
                "rave_interpolated": intp_path,
                "rave_raw": rave_to_forecast,
            }
        )
        self._forecast_metadata = df
        return df

    @property
    def is_first_day(self) -> bool:
        """`True` if this is considered the "first day" of the simulation where there is no
        interpolated or raw RAVE data available."""

        is_first_day = (
            self.forecast_metadata["rave_interpolated"].isnull().all()
            and self.forecast_metadata["rave_raw"].isnull().all()
        )
        self.log(f"{is_first_day=}")
        return is_first_day

    def run(self) -> None:
        """Run the preprocessor."""
        self.log("run: entering")
        if self.is_first_day:
            if self._context.rank == 0:
                self.create_dummy_emissions_file()
        else:
            self._regrid_processor.run(self.forecast_metadata)
            if self._context.rank == 0:
                self._cycle_processor.process_emissions(self.forecast_metadata)
        self.log("run: exiting")

    def create_dummy_emissions_file(self) -> None:
        """Create a dummy emissions file. This occurs if it is the first day of the forecast or
        there is an exception and the context is set to not exit on error."""
        self.log("create_dummy_emissions_file: enter")
        self.log(f"{self._context.emissions_path=}")
        with open_nc(self._context.emissions_path, "w", parallel=False, clobber=True) as ds:
            create_template_emissions_file(ds, self._context.grid_out_shape, is_dummy=True)
            with open_nc(self._context.grid_out, parallel=False) as ds_src:
                ds.variables["geolat"][:] = ds_src.variables["grid_latt"][:]
                ds.variables["geolon"][:] = ds_src.variables["grid_lont"][:]

            for varname in [
                "frp_davg",
                "ebb_rate",
                "fire_end_hr",
                "hwp_davg",
                "totprcp_24hrs",
            ]:
                create_sd_variable(ds, SD_VARS.get(varname))
        self.log("create_dummy_emissions_file: exit")

    def finalize(self) -> None:
        """Finalize the preprocessor."""
        self.log("finalize: exiting")
