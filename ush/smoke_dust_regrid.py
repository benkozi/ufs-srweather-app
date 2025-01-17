import logging
from copy import copy, deepcopy
from typing import Any

import esmpy
import numpy as np

from smoke_dust_context import SmokeDustContext
import pandas as pd

from smoke_dust_interpolation import NcToGrid, GridSpec, open_nc, create_template_emissions_file, \
    create_sd_variable, NcToField, create_descriptive_statistics
from smoke_dust_interp_tools import mask_edges


class SmokeDustRegridProcessor:

    def __init__(self, context: SmokeDustContext, forecast_metadata: pd.DataFrame):
        self._context = context
        self._forecast_metadata = forecast_metadata

        # Holds interpolation descriptive statistics
        self._interpolation_stats = None

    def log(self, *args: Any, **kwargs: Any) -> None:
        #tdk: superclass for processors
        self._context.log(*args, **kwargs)

    def run(self) -> None:
        # Select which RAVE files to interpolate
        rave_to_interpolate = self._forecast_metadata[
            self._forecast_metadata['rave_interpolated'].isnull() & ~self._forecast_metadata['rave_raw'].isnull()]

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
                src_nc2field = NcToField(path=row_data['rave_raw'], name=field_name, gwrap=src_gwrap,
                                         dim_time=('time',))
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
            self._forecast_metadata.loc[row_idx, 'rave_interpolated'] = output_file_path
            row_data['rave_interpolated'] = output_file_path

            if self._context.rank == 0:
                self._interpolation_postprocessing_(row_data)

        if self._context.rank == 0 and self._context.should_calc_desc_stats and self._interpolation_stats is not None:
            self.log(f"writing interpolation statistics: {self._context.interpolation_statistics_path}")
            self._interpolation_stats.to_csv(self._context.interpolation_statistics_path, index=False)

    def _interpolation_postprocessing_(self, row_data: pd.Series) -> None:
        self.log("_run_interpolation_postprocessing: enter", level=logging.DEBUG)

        calc_stats = self._context.should_calc_desc_stats

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
