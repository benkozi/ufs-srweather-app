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
from typing import List, Literal, Dict, Any

import esmpy
import pandas as pd
from numpy.ma.core import MaskedArray


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
        # self._grid_out_shape = None

        # Holds interpolation descriptive statistics
        self._interpolation_stats = None

        self.log(f"initialization complete. {self._context=}")

    def log(self, *args: Any, **kwargs: Any) -> None:
        self._context.log(*args, **kwargs)

    @property
    def forecast_metadata(self) -> pd.DataFrame:
        if self._forecast_metadata is not None:
            return self._forecast_metadata

        # Create forecast times

        # fcst_datetime = dt.datetime.strptime(self._context.current_day, "%Y%m%d%H")
        # match self._context.ebb_dcycle_flag:
        #     case EbbDCycle.ONE:
        #         if self._context.persistence:
        #             self.log("Creating emissions for persistence method where satellite FRP persist from previous day")
        #             start_datetime = fcst_datetime - dt.timedelta(days=1)
        #         else:
        #             self.log("Creating emissions using current date satellite FRP")
        #             start_datetime = fcst_datetime
        #     case EbbDCycle.TWO:
        #         self.log("Creating emissions for modulated persistence by Wildfire potential")
        #         start_datetime = fcst_datetime - dt.timedelta(days=1, hours=1)
        #     case _:
        #         raise NotImplementedError(self._context.ebb_dcycle_flag)

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

    # @property
    # def grid_out_shape(self) -> Tuple[int, int]:
    #     if self._grid_out_shape is not None:
    #         return self._grid_out_shape
    #     with open_nc(self._context.grid_out, parallel=False) as ds:
    #         grid_out_shape = ds.dimensions["grid_yt"].size, ds.dimensions["grid_xt"].size
    #     self.log(f"{grid_out_shape=}")
    #     self._grid_out_shape = grid_out_shape
    #     return self._grid_out_shape

    def run(self) -> None:
        self.log("run: entering")
        if self.is_first_day:
            #tdk: implement creation of dummy emissions file
            raise NotImplementedError("is_first_day=True not implemented")
        else:
            #tdk: need try/catch to use dummy emissions if regridding fails or no rave data is available
            self._run_interpolation_()
            if self._context.rank == 0:
                self._cycle.process_emissions(self.forecast_metadata)
                # match self._context.ebb_dcycle_flag:
                #     case EbbDCycle.ONE:
                #         self._run_average_frp_()
                #     case EbbDCycle.TWO:
                #         self._run_emissions_forecast_()
                #     case _:
                #         raise NotImplementedError(self._context.ebb_dcycle_flag)
                if self._context.calculate_descriptive_interpolation_statistics:
                    self._cycle.create_derived_statistics()
        self.log("run: exiting")

    def _run_interpolation_(self):
        #tdk:last: refactor to method

        # Select which RAVE files need to be interpolated
        rave_to_interpolate = self.forecast_metadata[
            self.forecast_metadata['rave_interpolated'].isnull() & ~self.forecast_metadata['rave_raw'].isnull()]

        if len(rave_to_interpolate) == 0:
            self.log("all rave files have been interpolated")
            return

        first = True
        for row_idx, row_data in rave_to_interpolate.iterrows():
            row_dict = row_data.to_dict()
            self.log(f"processing RAVE interpolation row: {row_idx}, {row_dict}")

            if first:
                self.log("creating destination grid from RRFS grid file")
                dst_nc2grid = NcToGrid(
                    path=self._context.grid_out,
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
                dst_gwrap = dst_nc2grid.create_grid_wrapper()

                # We are translating metadata and some structure for the destination grid.
                dst_output_gwrap = copy(dst_gwrap)
                dst_output_gwrap.corner_dims = None
                dst_output_gwrap.spec = GridSpec(x_center="geolon", y_center="geolat", x_dim=('lon',), y_dim=('lat',))
                dst_output_gwrap.dims = deepcopy(dst_gwrap.dims)
                dst_output_gwrap.dims.value[0].name = ('lon',)
                dst_output_gwrap.dims.value[1].name = ('lat',)

            forecast_date = row_data['forecast_date']
            output_file_path = self._context.intp_dir / f"{self._context.rave_to_intp}{forecast_date}00_{forecast_date}59.nc"
            self.log(f"creating output file: {output_file_path}")
            with open_nc(output_file_path, "w") as ds:
                create_template_emissions_file(ds, self._context.grid_out_shape)

                create_sd_variable(ds, "frp_avg_hr", "Mean Fire Radiative Power", "MW", fill_value_str="0.f",
                                   fill_value_float=0.0)
                create_sd_variable(ds, "FRE", "FRE", "MJ", fill_value_str="0.f", fill_value_float=0.0)

            dst_output_gwrap.fill_nc_variables(output_file_path)

            for field_name in self._context.vars_emis:

                # tdk: clean this up
                match field_name:
                    case "FRP_MEAN":
                        dst_field_name = "frp_avg_hr"
                    case "FRE":
                        dst_field_name = "FRE"
                    case _:
                        raise NotImplementedError(field_name)

                self.log("creating destination field", level=logging.DEBUG)
                dst_nc2field = NcToField(path=output_file_path, name=dst_field_name, gwrap=dst_output_gwrap,
                                         dim_time=('t',))
                dst_fwrap = dst_nc2field.create_field_wrapper()

                if first:
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

                self.log("creating source field", level=logging.DEBUG)
                src_nc2field = NcToField(path=row_data['rave_raw'], name=field_name, gwrap=src_gwrap, dim_time=('time',))
                src_fwrap = src_nc2field.create_field_wrapper()

                if first:
                    self.log("creating regridder")
                    self.log(f"{src_fwrap.value.data.shape=}", level=logging.DEBUG)
                    self.log(f"{dst_fwrap.value.data.shape=}", level=logging.DEBUG)
                    regridder = esmpy.RegridFromFile(src_fwrap.value, dst_fwrap.value,
                                                     filename=str(self._context.weightfile))
                    first = False

                # tdk: make this smoother; automatically fill masked data maybe
                src_data = src_fwrap.value.data
                match field_name:
                    case "FRP_MEAN":
                        src_data[:] = np.where(src_data == -1.0, 0.0, src_data)
                    case "FRE":
                        src_data[:] = np.where(src_data > 1000., src_data, 0.0)
                    case _:
                        raise NotImplementedError(field_name)

                # Execute the ESMF regridding
                self.log(f"run regridding", level=logging.DEBUG)
                _ = regridder(src_fwrap.value, dst_fwrap.value)

                # Persist the destination field
                self.log(f"filling netcdf", level=logging.DEBUG)
                dst_fwrap.fill_nc_variable(output_file_path)

            # Update the forecast metadata with the interpolated RAVE file data
            self.forecast_metadata.loc[row_idx, 'rave_interpolated'] = output_file_path
            row_data['rave_interpolated'] = output_file_path

            if self._context.rank == 0:
                self._interpolation_postprocessing_(row_data)

        if self._context.rank == 0 and self._context.calculate_descriptive_interpolation_statistics and self._interpolation_stats is not None:
            self.log(f"writing interpolation statistics: {self._context.interpolation_statistics_path}")
            self._interpolation_stats.to_csv(self._context.interpolation_statistics_path, index=False)

    def _interpolation_postprocessing_(self, row_data: pd.Series) -> None:
        self.log("_run_interpolation_postprocessing: enter", level=logging.DEBUG)

        calc_stats = self._context.calculate_descriptive_interpolation_statistics

        field_names_dst = ["frp_avg_hr", "FRE"] #tdk: make this a property or something
        with open_nc(row_data["rave_interpolated"], parallel=False) as ds:
            dst_data = {ii: ds.variables[ii][:] for ii in field_names_dst}
        if calc_stats:
            # Do these calculations before we modify the arrays since edge masking is inplace
            dst_desc_unmasked = create_descriptive_statistics(dst_data, "dst_unmasked", None)

        # Mask edges to reduce model edge effects
        self.log("masking edges", level=logging.DEBUG)
        for v in dst_data.values():
            # Operation is inplace
            mask_edges(v[0, :, :])

        # Persist masked data to disk
        with open_nc(row_data["rave_interpolated"], parallel=False, mode="a") as ds:
            for k, v in dst_data.items():
                ds.variables[k][:] = v

        if calc_stats:
            with open_nc(row_data["rave_raw"], parallel=False) as ds:
                src_desc = create_descriptive_statistics({ii: ds.variables[ii][:] for ii in self._context.vars_emis}, "src", row_data["rave_raw"])
                src_desc.rename(columns={'FRP_MEAN': 'frp_avg_hr'}, inplace=True)
            dst_desc_masked = create_descriptive_statistics(dst_data, "dst_masked", row_data["rave_interpolated"])
            summary = pd.concat([ii.transpose() for ii in [src_desc, dst_desc_unmasked, dst_desc_masked]])
            summary.index.name = "variable"
            summary['forecast_date'] = row_data['forecast_date']
            summary.reset_index(inplace=True)
            if self._interpolation_stats is None:
                self._interpolation_stats = summary
            else:
                self._interpolation_stats = pd.concat([self._interpolation_stats, summary])

        self.log("_run_interpolation_postprocessing: exit", level=logging.DEBUG)

    # @staticmethod
    # def _create_descriptive_statistics_(container: Dict[str, MaskedArray], origin: Literal["src", "dst_unmasked", "dst_masked"], path: Path) -> pd.DataFrame:
    #     df = pd.DataFrame.from_dict({k: v.filled(np.nan).ravel() for k, v in container.items()})
    #     desc = df.describe()
    #     adds = {}
    #     for field_name in container.keys():
    #         adds[field_name] = [df[field_name].sum(), df[field_name].isnull().sum(), origin, path]
    #     desc = pd.concat([desc, pd.DataFrame(data=adds, index=['sum', 'count_null', "origin", "path"])])
    #     return desc

    # def _run_average_frp_(self):
    #     self.log("averaging FRP")
        #tdk:story: need fail-over option to return empty arrays

        # frp_daily = np.zeros(self.grid_out_shape)
        # ebb_smoke_total = []
        # frp_avg_hr = []
        #
        # with xr.open_dataset(self._context.veg_map) as ds:
        #     emiss_factor = ds['emiss_factor'].values
        # with xr.open_dataset(self._context.grid_out) as ds:
        #     target_area = ds['area'].values
        #
        # for row_idx, row_df in self.forecast_metadata.iterrows():
        #     self.log(f"processing emissions: {row_idx}, {row_df.to_dict()}")
        #     with xr.open_dataset(row_df['rave_interpolated']) as ds:
        #         fre = ds['FRE'][0, :, :].values
        #         frp = ds['frp_avg_hr'][0, :, :].values
        #
        #         match self._context.ebb_dcycle_flag:
        #             #tdk:ja: can we give these cycles more explanatory names?
        #             case EbbDCycle.ONE:
        #                 frp_avg_hr.append(frp)
        #                 ebb_hourly = (fre * emiss_factor * self._context.beta * self._context.fg_to_ug) / (
        #                         target_area * self._context.to_s
        #                 )
        #                 ebb_smoke_total.append(
        #                     np.where(frp > 0, ebb_hourly, 0)
        #                 )
        #             case EbbDCycle.TWO:
        #                 ebb_hourly = (
        #                         fre * emiss_factor * self._context.beta * self._context.fg_to_ug / target_area
        #                 )
        #                 ebb_smoke_total.append(
        #                     np.where(frp > 0, ebb_hourly, 0).ravel()
        #                 )
        #                 frp_daily += np.where(frp > 0, frp, 0).ravel()
        #             case _:
        #                 raise NotImplementedError(self._context.ebb_dcycle_flag)
        #
        # self.log("reshaping arrays")
        # match self._context.ebb_dcycle_flag:
        #     case EbbDCycle.ONE:
        #         frp_avg_reshaped = np.stack(frp_avg_hr, axis=0)
        #         ebb_total_reshaped = np.stack(ebb_smoke_total, axis=0)
        #     case EbbDCycle.TWO:
        #         summed_array = np.sum(np.array(ebb_smoke_total), axis=0)
        #         num_zeros = len(ebb_smoke_total) - np.sum(
        #             [arr == 0 for arr in ebb_smoke_total], axis=0
        #         )
        #         safe_zero_count = np.where(num_zeros == 0, 1, num_zeros)
        #         result_array = np.array(
        #             [
        #                 (
        #                     summed_array[i] / 2
        #                     if safe_zero_count[i] == 1
        #                     else summed_array[i] / safe_zero_count[i]
        #                 )
        #                 for i in range(len(safe_zero_count))
        #             ]
        #         )
        #         result_array[num_zeros == 0] = summed_array[num_zeros == 0]
        #         ebb_total = result_array.reshape(self.grid_out_shape)
        #         ebb_total_reshaped = ebb_total / 3600
        #         temp_frp = np.array(
        #             [
        #                 (
        #                     frp_daily[i] / 2
        #                     if safe_zero_count[i] == 1
        #                     else frp_daily[i] / safe_zero_count[i]
        #                 )
        #                 for i in range(len(safe_zero_count))
        #             ]
        #         )
        #         temp_frp[num_zeros == 0] = frp_daily[num_zeros == 0]
        #         frp_avg_reshaped = temp_frp.reshape(*self.grid_out_shape)
        #     case _:
        #         raise NotImplementedError(self._context.ebb_dcycle_flag)
        #
        # np.nan_to_num(frp_avg_reshaped, copy=False, nan=0.0)

        # self.log(f"frp_avg_reshaped nan count={np.isnan(frp_avg_reshaped).sum()}")
        # self.log(f"ebb_total_reshaped nan count={np.isnan(ebb_total_reshaped).sum()}")

        # derived = self._cycle.average_frp(self.forecast_metadata)
        #
        # self.log(f"creating emissions file: {self._context.emissions_path}")
        # with open_nc(self._context.emissions_path, "w", parallel=False, clobber=True) as ds_out:
        #     self._create_template_emissions_file_(ds_out)
        #     with open_nc(self._context.grid_out, parallel=False) as ds_src:
        #         ds_out.variables["geolat"][:] = ds_src.variables["grid_latt"][:]
        #         ds_out.variables["geolon"][:] = ds_src.variables["grid_lont"][:]
        #     create_sd_variable(
        #         ds_out, "frp_avg_hr", "mean Fire Radiative Power", "MW", "0.f", 0.
        #     )
        #     ds_out.variables["frp_avg_hr"][:] = derived[DerivedVariable.FRP_AVG]
        #     create_sd_variable(
        #         ds_out, "ebb_smoke_hr", "EBB emissions", "ug m-2 s-1", "0.f", 0.
        #     )
        #     ds_out.variables["ebb_smoke_hr"][:] = derived[DerivedVariable.EBB_TOTAL]

    # def _run_emissions_forecast_(self) -> None:
    #     self.log("_run_emissions_forecast_: enter")
    #     #tdk: story emissions forecast
    #     raise NotImplementedError(EbbDCycle.TWO)
    #     # self.log("_run_emissions_forecast_: exit")

    # def _create_template_emissions_file_(self, ds: netCDF4.Dataset, grid_shape: Tuple[int, int]):
    #     ds.createDimension("t", None)
    #     ds.createDimension("lat", grid_shape[0])
    #     ds.createDimension("lon", grid_shape[1])
    #     setattr(ds, "PRODUCT_ALGORITHM_VERSION", "Beta")
    #     setattr(ds, "TIME_RANGE", "1 hour")
    #
    #     create_sd_coordinate_variable(ds, "geolat", "cell center latitude", "degrees_north", "-9999.f", -9999.0)
    #     create_sd_coordinate_variable(ds, "geolon", "cell center longitude", "degrees_east", "-9999.f", -9999.0)

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


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Workflow
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
