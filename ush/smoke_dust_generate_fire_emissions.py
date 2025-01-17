#!/usr/bin/env python3

#########################################################################
#                                                                       #
# Python script for fire emissions preprocessing from RAVE FRP and FRE  #
# (Li et al.,2022).                                                     #
# johana.romero-alvarez@noaa.gov                                        #
#                                                                       #
#########################################################################

import sys
import fnmatch
from copy import copy, deepcopy
from pathlib import Path
from typing import List, Any

import esmpy
import pandas as pd


from smoke_dust_interpolation import NcToGrid, GridSpec, NcToField, create_template_emissions_file, \
    create_descriptive_statistics
from smoke_dust_interpolation import open_nc, create_sd_variable

import numpy as np

from smoke_dust_interp_tools import mask_edges

from smoke_dust_context import SmokeDustContext
from smoke_dust_cycle import SmokeDustCycleTwo
import logging

from smoke_dust_context import EbbDCycle
from smoke_dust_cycle import SmokeDustCycleOne


class SmokeDustPreprocessor:

    def __init__(self, context: SmokeDustContext) -> None:
        self._context = context
        match self._context.ebb_dcycle_flag:
            case EbbDCycle.ONE:
                self._cycle = SmokeDustCycleOne(context)
            case EbbDCycle.TWO:
                self._cycle = SmokeDustCycleTwo(context)
            case _:
                raise NotImplementedError(self._context.ebb_dcycle_flag)

        # On-demand/cached property values
        self._forecast_metadata = None

        self.log(f"initialization complete. {self._context=}")

    def log(self, *args: Any, **kwargs: Any) -> None:
        self._context.log(*args, **kwargs)

    @property
    def forecast_metadata(self) -> pd.DataFrame:
        if self._forecast_metadata is not None:
            return self._forecast_metadata

        start_datetime = self._cycle.create_start_datetime()
        forecast_dates = pd.date_range(start=start_datetime, periods=24, freq="h").strftime(
            "%Y%m%d%H"
        )

        # Collect metadata on RAVE input files
        intp_path = []
        rave_to_forecast = []
        for date in forecast_dates:
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

        df = pd.DataFrame(data={'forecast_date': forecast_dates,'rave_interpolated': intp_path, 'rave_raw': rave_to_forecast})
        self._forecast_metadata = df
        return df

    @property
    def is_first_day(self) -> bool:
        return self.forecast_metadata['rave_interpolated'].isnull().all() and self.forecast_metadata['rave_raw'].isnull().all()

    def run(self) -> None:
        self.log("run: entering")
        if self.is_first_day:
            #tdk: implement creation of dummy emissions file
            raise NotImplementedError
        else:
            #tdk: need try/catch to use dummy emissions if regridding fails or no rave data is available
            self._run_interpolation_()
            if self._context.rank == 0:
                self._cycle.process_emissions(self.forecast_metadata)
                if self._context.should_calc_desc_stats:
                    self._cycle.create_derived_statistics()
        self.log("run: exiting")

    # def _run_interpolation_(self):
        #tdk:last: refactor to method

        # # Select which RAVE files to interpolate
        # rave_to_interpolate = self.forecast_metadata[
        #     self.forecast_metadata['rave_interpolated'].isnull() & ~self.forecast_metadata['rave_raw'].isnull()]
        #
        # if len(rave_to_interpolate) == 0:
        #     self.log("all rave files have been interpolated")
        #     return
        #
        # first = True
        # for row_idx, row_data in rave_to_interpolate.iterrows():
        #     row_dict = row_data.to_dict()
        #     self.log(f"processing RAVE interpolation row: {row_idx}, {row_dict}")
        #
        #     if first:
        #         self.log("creating destination grid from RRFS grid file")
        #         dst_nc2grid = NcToGrid(
        #             path=self._context.grid_out,
        #             spec=GridSpec(
        #                 x_center="grid_lont",
        #                 y_center="grid_latt",
        #                 x_dim=("grid_xt",),
        #                 y_dim=("grid_yt",),
        #                 x_corner="grid_lon",
        #                 y_corner="grid_lat",
        #                 x_corner_dim=("grid_x",),
        #                 y_corner_dim=("grid_y",),
        #             ),
        #         )
        #         dst_gwrap = dst_nc2grid.create_grid_wrapper()
        #
        #         # We are translating metadata and some structure for the destination grid.
        #         dst_output_gwrap = copy(dst_gwrap)
        #         dst_output_gwrap.corner_dims = None
        #         dst_output_gwrap.spec = GridSpec(x_center="geolon", y_center="geolat", x_dim=('lon',), y_dim=('lat',))
        #         dst_output_gwrap.dims = deepcopy(dst_gwrap.dims)
        #         dst_output_gwrap.dims.value[0].name = ('lon',)
        #         dst_output_gwrap.dims.value[1].name = ('lat',)
        #
        #     forecast_date = row_data['forecast_date']
        #     output_file_path = self._context.intp_dir / f"{self._context.rave_to_intp}{forecast_date}00_{forecast_date}59.nc"
        #     self.log(f"creating output file: {output_file_path}")
        #     with open_nc(output_file_path, "w") as ds:
        #         create_template_emissions_file(ds, self._context.grid_out_shape)
        #
        #         create_sd_variable(ds, "frp_avg_hr", "Mean Fire Radiative Power", "MW", fill_value_str="0.f",
        #                            fill_value_float=0.0)
        #         create_sd_variable(ds, "FRE", "FRE", "MJ", fill_value_str="0.f", fill_value_float=0.0)
        #
        #     dst_output_gwrap.fill_nc_variables(output_file_path)
        #
        #     for field_name in self._context.vars_emis:
        #
        #         # tdk: clean this up
        #         match field_name:
        #             case "FRP_MEAN":
        #                 dst_field_name = "frp_avg_hr"
        #             case "FRE":
        #                 dst_field_name = "FRE"
        #             case _:
        #                 raise NotImplementedError(field_name)
        #
        #         self.log("creating destination field", level=logging.DEBUG)
        #         dst_nc2field = NcToField(path=output_file_path, name=dst_field_name, gwrap=dst_output_gwrap,
        #                                  dim_time=('t',))
        #         dst_fwrap = dst_nc2field.create_field_wrapper()
        #
        #         if first:
        #             self.log("creating source grid from RAVE file")
        #             src_nc2grid = NcToGrid(
        #                 path=self._context.grid_in,
        #                 spec=GridSpec(
        #                     x_center="grid_lont",
        #                     y_center="grid_latt",
        #                     x_dim=("grid_xt",),
        #                     y_dim=("grid_yt",),
        #                     x_corner="grid_lon",
        #                     y_corner="grid_lat",
        #                     x_corner_dim=("grid_x",),
        #                     y_corner_dim=("grid_y",),
        #                 ),
        #             )
        #             src_gwrap = src_nc2grid.create_grid_wrapper()
        #
        #         self.log("creating source field", level=logging.DEBUG)
        #         src_nc2field = NcToField(path=row_data['rave_raw'], name=field_name, gwrap=src_gwrap, dim_time=('time',))
        #         src_fwrap = src_nc2field.create_field_wrapper()
        #
        #         if first:
        #             self.log("creating regridder")
        #             self.log(f"{src_fwrap.value.data.shape=}", level=logging.DEBUG)
        #             self.log(f"{dst_fwrap.value.data.shape=}", level=logging.DEBUG)
        #             regridder = esmpy.RegridFromFile(src_fwrap.value, dst_fwrap.value,
        #                                              filename=str(self._context.weightfile))
        #             first = False
        #
        #         # tdk: make this smoother; automatically fill masked data maybe
        #         src_data = src_fwrap.value.data
        #         match field_name:
        #             case "FRP_MEAN":
        #                 src_data[:] = np.where(src_data == -1.0, 0.0, src_data)
        #             case "FRE":
        #                 src_data[:] = np.where(src_data > 1000., src_data, 0.0)
        #             case _:
        #                 raise NotImplementedError(field_name)
        #
        #         # Execute the ESMF regridding
        #         self.log(f"run regridding", level=logging.DEBUG)
        #         _ = regridder(src_fwrap.value, dst_fwrap.value)
        #
        #         # Persist the destination field
        #         self.log(f"filling netcdf", level=logging.DEBUG)
        #         dst_fwrap.fill_nc_variable(output_file_path)
        #
        #     # Update the forecast metadata with the interpolated RAVE file data
        #     self.forecast_metadata.loc[row_idx, 'rave_interpolated'] = output_file_path
        #     row_data['rave_interpolated'] = output_file_path
        #
        #     if self._context.rank == 0:
        #         self._interpolation_postprocessing_(row_data)
        #
        # if self._context.rank == 0 and self._context.should_calc_desc_stats and self._interpolation_stats is not None:
        #     self.log(f"writing interpolation statistics: {self._context.interpolation_statistics_path}")
        #     self._interpolation_stats.to_csv(self._context.interpolation_statistics_path, index=False)

    # def _interpolation_postprocessing_(self, row_data: pd.Series) -> None:
    #     self.log("_run_interpolation_postprocessing: enter", level=logging.DEBUG)
    #
    #     calc_stats = self._context.should_calc_desc_stats
    #
    #     field_names_dst = ["frp_avg_hr", "FRE"] #tdk: make this a property or something
    #     with open_nc(row_data["rave_interpolated"], parallel=False) as ds:
    #         dst_data = {ii: ds.variables[ii][:] for ii in field_names_dst}
    #     if calc_stats:
    #         # Do these calculations before we modify the arrays since edge masking is inplace
    #         dst_desc_unmasked = create_descriptive_statistics(dst_data, "dst_unmasked", None)
    #
    #     # Mask edges to reduce model edge effects
    #     self.log("masking edges", level=logging.DEBUG)
    #     for v in dst_data.values():
    #         # Operation is inplace
    #         mask_edges(v[0, :, :])
    #
    #     # Persist masked data to disk
    #     with open_nc(row_data["rave_interpolated"], parallel=False, mode="a") as ds:
    #         for k, v in dst_data.items():
    #             ds.variables[k][:] = v
    #
    #     if calc_stats:
    #         with open_nc(row_data["rave_raw"], parallel=False) as ds:
    #             src_desc = create_descriptive_statistics({ii: ds.variables[ii][:] for ii in self._context.vars_emis}, "src", row_data["rave_raw"])
    #             src_desc.rename(columns={'FRP_MEAN': 'frp_avg_hr'}, inplace=True)
    #         dst_desc_masked = create_descriptive_statistics(dst_data, "dst_masked", row_data["rave_interpolated"])
    #         summary = pd.concat([ii.transpose() for ii in [src_desc, dst_desc_unmasked, dst_desc_masked]])
    #         summary.index.name = "variable"
    #         summary['forecast_date'] = row_data['forecast_date']
    #         summary.reset_index(inplace=True)
    #         if self._interpolation_stats is None:
    #             self._interpolation_stats = summary
    #         else:
    #             self._interpolation_stats = pd.concat([self._interpolation_stats, summary])
    #
    #     self.log("_run_interpolation_postprocessing: exit", level=logging.DEBUG)

    def _create_dummy_emissions_file_(self) -> None:
        self.log("_create_dummy_emissions_file_: enter")
        self.log(f"{self._context.emissions_path=}")
        with open_nc(self._context.emissions_path, "w", parallel=False, clobber=True) as ds:
            create_template_emissions_file(ds, self._context.grid_out_shape)
            with open_nc(self._context.grid_out, parallel=False) as ds_src:
                ds.variables["geolat"][:] = ds_src.variables["grid_latt"][:]
                ds.variables["geolon"][:] = ds_src.variables["grid_lont"][:]

            create_sd_variable(ds, "frp_davg", "Daily mean Fire Radiative Power", "MW", "0.f", 0.)
            create_sd_variable(ds, "ebb_rate", "Total EBB emission", "ug m-2 s-1", "0.f", 0.)
            create_sd_variable(ds, "fire_end_hr", "Hours since fire was last detected", "hrs", "0.f", 0.)
            create_sd_variable(
                ds, "hwp_davg", "Daily mean Hourly Wildfire Potential", "none", "0.f", 0.
            )
            create_sd_variable(ds, "totprcp_24hrs", "Sum of precipitation", "m", "0.f", 0.)
        self.log("_create_dummy_emissions_file_: exit")

    def finalize(self) -> None:
        self.log('finalize: exiting')


def generate_emiss_workflow(
    args: List[str]
) -> None:
    """
    Prepares fire-related ICs. This is the main function that handles data movement and interpolation.
    #tdk: doc
    Args:
        staticdir: Path to fix files for the smoke and dust component
        ravedir: Path to the directory containing RAVE fire data files (hourly). This is typically the working directory (DATA)
        intp_dir: Path to interpolated RAVE data files from the previous cycles (DATA_SHARE)
        predef_grid: If ``RRFS_NA_3km``, use pre-defined grid dimensions
        ebb_dcycle_flag: Select the EBB cycle to run. Valid values are ``"1"`` or ``"2"``
        restart_interval: Indicates if restart files should be copied. The actual interval values are not used
        persistence: If ``TRUE``, use satellite observations from the previous day. Otherwise, use observations from the same day.
    """

    context = SmokeDustContext.create_from_args(args)
    processor = SmokeDustPreprocessor(context)
    try:
        processor.run()
        processor.finalize()
    except Exception as e:
        processor.log("unhandled error", exc_info=e)

if __name__ == "__main__":
    print("")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Welcome to interpolating RAVE and processing fire emissions!")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("")
    #tdk:story: use argparse
    generate_emiss_workflow(
        sys.argv[1:]
    )
    print("")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Exiting. Bye!")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("")
