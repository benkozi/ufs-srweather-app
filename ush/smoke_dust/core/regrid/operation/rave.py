import copy
import logging
from datetime import datetime, timezone
from functools import cached_property
from pathlib import Path

import esmpy
import numpy as np
import pandas as pd

from pydantic import ConfigDict

from smoke_dust.core.common import AbstractSmokeDustObject, open_nc, create_template_emissions_file, \
    create_sd_variable, create_descriptive_statistics, copy_nc_variable, ncdump, \
    create_sd_variable_mpas
from smoke_dust.core.context import RaveQaFilter, PredefinedGrid
from smoke_dust.core.regrid.common import FieldWrapper, NcToField, GridWrapper, NcToGrid, GridSpec, \
    load_variable_data, mask_edges, MeshWrapper, NcToMesh

from smoke_dust.core.regrid.operation.common import RegridOperationContext, EsmpyContext, \
    RegridFieldName
from smoke_dust.core.variable import SD_VARS


class RaveToGeomStrategyContext(RegridOperationContext):
    model_config = ConfigDict(frozen=False)

    field_names: tuple[RegridFieldName, ...]  # tdk:last: move to RegridOperationContext?
    dst_path: Path


class RaveToGridStrategy(AbstractSmokeDustObject):

    def __init__(self, context: RaveToGeomStrategyContext) -> None:
        self._context = context

        # Holds the current operation's required info
        self._curr_src_path: Path | None = None
        self._curr_output_path: Path | None = None

    def run(self, src_path: Path, output_path: Path) -> None:
        self._curr_src_path = src_path
        self._curr_output_path = output_path

        self.log(f"creating output file: {self._curr_output_path}")
        self._create_template_emissions_file_()

        for field_name in self._context.field_names:
            src_fwrap = self._create_src_field_wrapper_(field_name.src)

            # Execute the ESMF regridding
            dst_fwrap = self._create_dst_field_wrapper_(field_name.dst)
            self.log("run regridding", level=logging.DEBUG)
            self._regridder(src_fwrap.value, dst_fwrap.value)

            # Persist the destination field
            self.log("filling netcdf", level=logging.DEBUG)
            dst_fwrap.fill_nc_variable(self._curr_output_path)

    def _create_template_emissions_file_(self) -> None:
        with open_nc(self._curr_output_path, "w") as nc_ds:
            create_template_emissions_file(nc_ds, self._context.grid_out_shape)
            for field_name in self._context.field_names:
                create_sd_variable(nc_ds, SD_VARS.get(field_name.dst))
        self._dst_gwrap_output.fill_nc_variables(self._curr_output_path)

    def finalize(self) -> None:
        self.log("finalize")

    @cached_property
    def _esmpy_context(self) -> EsmpyContext:
        return EsmpyContext(
            regrid_method=esmpy.RegridMethod.CONSERVE,
            zero_region=esmpy.Region.TOTAL,
            debug=self._context.esmpy_debug,
            ignore_degenerate=True,
            unmapped_action=esmpy.UnmappedAction.IGNORE,
        )

    @cached_property
    def _dst_fwrap(self) -> FieldWrapper:
        nc_to_field = NcToField(
            path=self._curr_output_path,
            name=self._context.field_names[0].dst,
            gwrap=self._dst_gwrap_output,
            dim_time=("t",),
        )
        return nc_to_field.create_field_wrapper()

    def _create_dst_field_wrapper_(self, name: str) -> FieldWrapper:
        self.log("creating destination field")
        fwrap = self._dst_fwrap
        fwrap.name = name
        return fwrap


    @cached_property
    def _dst_gwrap(self) -> GridWrapper:
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
        return dst_nc2grid.create_grid_wrapper()

    @cached_property
    def _dst_gwrap_output(self) -> GridWrapper:
        self.log("creating destination grid wrapper")
        # We are translating metadata and some structure for the destination grid.
        dst_output_gwrap = copy.copy(self._dst_gwrap)
        dst_output_gwrap.corner_dims = None
        dst_output_gwrap.spec = GridSpec(
            x_center="geolon", y_center="geolat", x_dim=("lon",), y_dim=("lat",)
        )
        dst_output_gwrap.dims = copy.deepcopy(self._dst_gwrap.dims)
        dst_output_gwrap.dims.value[0].name = ("lon",)
        dst_output_gwrap.dims.value[1].name = ("lat",)
        return dst_output_gwrap

    @cached_property
    def _src_fwrap(self) -> FieldWrapper:
        nc_to_field = NcToField(
            path=self._curr_src_path,
            name=self._context.field_names[0].src,
            gwrap=self._src_gwrap,
            dim_time=("time",),
        )
        return nc_to_field.create_field_wrapper()

    def _create_src_field_wrapper_(self, name: str) -> FieldWrapper:
        self.log(f"creating source field: {name=}")
        src_path = self._curr_src_path
        fwrap = NcToField.create_field_wrapper_from_template(src_path, self._src_fwrap, name)
        assert fwrap.name == name
        src_data = fwrap.value.data
        match name:
            case "FRP_MEAN":
                src_data[:] = np.where(src_data == -1.0, 0.0, src_data)
            case "FRE":
                src_data[:] = np.where(src_data > 1000.0, src_data, 0.0)
            case _:
                raise NotImplementedError(name)
        rave_qa_filter = self._context.rave_qa_filter
        if rave_qa_filter == RaveQaFilter.HIGH:
            with open_nc(src_path, parallel=True) as rave_ds:
                rave_qa = load_variable_data(
                    rave_ds.variables["QA"],  # pylint: disable=unsubscriptable-object
                    fwrap.dims,
                )
            set_to_zero = rave_qa < 2
            self.log(
                f"RAVE QA filter applied: {rave_qa_filter=}; "
                f"{set_to_zero.size=}; {np.sum(set_to_zero)=}"
            )
            src_data[set_to_zero] = 0.0
        else:
            if rave_qa_filter != RaveQaFilter.NONE:
                raise NotImplementedError
        return fwrap

    @cached_property
    def _src_gwrap(self) -> GridWrapper:
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
        return src_nc2grid.create_grid_wrapper()

    @cached_property
    def _regridder(self) -> esmpy.Regrid | esmpy.RegridFromFile:
        self.log("creating regridder")
        src_fwrap = self._src_fwrap
        dst_fwrap = self._dst_fwrap
        self.log(f"{src_fwrap.value.data.shape=}")
        self.log(f"{dst_fwrap.value.data.shape=}")

        if (
            self._context.predef_grid == PredefinedGrid.RRFS_NA_13KM
            or self._context.regrid_in_memory
        ):
            # ESMF does not like reading the weights for this field combination (rc=-1). The
            # error can be bypassed by creating weights in-memory.
            self.log("creating regridder in-memory")
            if self._context.create_weight_file:
                if self._context.weightfile.exists():
                    raise ValueError(f"cannot overwrite weight file: {self._context.weightfile}")
                filename = str(self._context.weightfile)
            else:
                filename = None
                # filename = "/opt/project/benkozi-data/weights-nproc8.nc" #tdk:rm
            esmpy_context = self._esmpy_context
            regridder = esmpy.Regrid(
                src_fwrap.value,
                dst_fwrap.value,
                regrid_method=esmpy_context.regrid_method,
                unmapped_action=esmpy_context.unmapped_action,
                ignore_degenerate=esmpy_context.ignore_degenerate,
                # Can be used to create a weight file for testing
                filename=filename,
            )
        else:
            self.log("creating regridder from file")
            regridder = esmpy.RegridFromFile(
                src_fwrap.value,
                dst_fwrap.value,
                filename=str(self._context.weightfile),
            )
        return regridder

    def mask_edges(self, dst_data):
        # Mask edges to reduce model edge effects
        self.log("masking edges", level=logging.DEBUG)
        for value in dst_data.values():
            # Operation is inplace
            mask_edges(value[0, :, :])


class RaveToMpasStrategy(RaveToGridStrategy):
    #tdk: story to integrate pyremap

    @cached_property
    def _dst_fwrap(self) -> FieldWrapper:
        nc_to_field = NcToField(
            path=self._curr_output_path,
            name=self._context.field_names[0].dst,
            gwrap=self._dst_gwrap_output,
            dim_time=("Time",),
            meshloc=esmpy.MeshLoc.ELEMENT
        )
        output = ncdump(self._curr_output_path) #tdk:rm
        return nc_to_field.create_field_wrapper()

    @cached_property
    def _dst_gwrap(self) -> MeshWrapper:
        gwrap = NcToMesh(path=self._context.grid_out, n_elements=self._context.grid_out_shape[0]).create_mesh_wrapper()
        return gwrap

    @cached_property
    def _dst_gwrap_output(self) -> MeshWrapper:
        return self._dst_gwrap

    def _create_template_emissions_file_(self) -> None:
        if self._context.rank == 0:
            with open_nc(self._curr_output_path, mode="w", parallel=False) as dst_nc:
                dst_nc.createDimension("nCells", self._context.grid_out_shape[0])
                dst_nc.createDimension("nkfire", 1)
                dst_nc.createDimension("Time")
                dst_nc.setncattr("created_at", str(datetime.now(timezone.utc)))
                dst_nc.setncattr("src_path", str(self._curr_src_path))
                dst_nc.setncattr("dst_path", str(self._curr_output_path))
                for field_name in self._context.field_names:
                    create_sd_variable_mpas(dst_nc, SD_VARS.get(field_name.dst))
                with open_nc(self._context.grid_out, mode="r", parallel=False) as src_nc:
                    for varname, new_name in zip(("grid_center_lat", "grid_center_lon"), ("latCell", "lonCell")):
                        copy_nc_variable(src_nc, dst_nc, varname, copy_data=True, new_name=new_name)

    def mask_edges(self, dst_data):
        self.log("no masking with MPAS")


class RaveToGeomProcessor(AbstractSmokeDustObject):

    def __init__(self, context: RegridOperationContext):
        self._context = context

        # Holds interpolation descriptive statistics
        self._interpolation_stats: None | pd.DataFrame = None

    def run(self) -> None:
        cycle_metadata = self._context.cycle_metadata
        # Select which RAVE files to interpolate
        rave_to_interpolate = cycle_metadata[
            cycle_metadata["rave_interpolated"].isnull() & ~cycle_metadata["rave_raw"].isnull()
        ]
        if len(rave_to_interpolate) == 0:
            self.log("all rave files have been interpolated")
            return

        self._run_impl_(rave_to_interpolate)

    def finalize(self) -> None:
        self.log("finalize")

    @cached_property
    def _strategy(self) -> RaveToGridStrategy:
        field_names = (
            RegridFieldName(src="FRP_MEAN", dst="frp_avg_hr"),
            RegridFieldName(src="FRE", dst="FRE"),
        )
        kwds = self._context.model_dump()
        kwds.update({'field_names': field_names,
                     'dst_path': self._context.grid_in,
                     })
        context = RaveToGeomStrategyContext.model_validate(kwds)

        match self._context.predef_grid:
            case PredefinedGrid.MPAS_NA_15KM:
                strategy_klass = RaveToMpasStrategy
            case _:
                strategy_klass = RaveToGridStrategy

        self.log(f"{strategy_klass=}")
        strategy = strategy_klass(context=context)
        return strategy

    def _run_impl_(self, rave_to_interpolate: pd.Series) -> None:
        cycle_metadata = self._context.cycle_metadata
        for row_idx, row_data in rave_to_interpolate.iterrows():
            row_dict = row_data.to_dict()
            self.log(f"processing RAVE interpolation row: {row_idx}, {row_dict}")

            forecast_date = row_data["forecast_date"]
            output_file_path = (
                self._context.intp_dir
                / f"{self._context.rave_to_intp}{forecast_date}00_{forecast_date}59.nc"
            )

            self._strategy.run(row_data['rave_raw'], output_file_path)
            self._strategy.finalize()

            # Update the forecast metadata with the interpolated RAVE file data
            cycle_metadata.loc[row_idx, "rave_interpolated"] = output_file_path
            row_data["rave_interpolated"] = output_file_path

            if self._context.rank == 0:
                self._regrid_postprocessing_(row_data)

        if (
            self._context.rank == 0
            and self._context.should_calc_desc_stats
            and self._interpolation_stats is not None
        ):
            cycle_dates = cycle_metadata["forecast_date"]
            stats_path = (
                self._context.intp_dir
                / f"stats_regridding_{cycle_dates.min()}_{cycle_dates.max()}.csv"
            )
            self.log(f"writing interpolation statistics: {stats_path=}")
            self._interpolation_stats.to_csv(stats_path, index=False)

    def _regrid_postprocessing_(self, row_data: pd.Series) -> None:
        self.log("_run_interpolation_postprocessing: enter", level=logging.DEBUG)

        calc_stats = self._context.should_calc_desc_stats

        field_names_dst = [
            "frp_avg_hr",
            "FRE",
        ]
        with open_nc(row_data["rave_interpolated"], parallel=False) as nc_ds:
            dst_data = {ii: nc_ds.variables[ii][:] for ii in field_names_dst}
        if calc_stats:
            # Do these calculations before we modify the arrays since edge masking is inplace
            dst_desc_unmasked = create_descriptive_statistics(dst_data, "dst_unmasked", None)

        self._strategy.mask_edges(dst_data)

        # Persist masked data to disk
        with open_nc(row_data["rave_interpolated"], parallel=False, mode="a") as nc_ds:
            for key, value in dst_data.items():
                nc_ds.variables[key][:] = value

        if calc_stats:
            with open_nc(row_data["rave_raw"], parallel=False) as nc_ds:
                src_desc = create_descriptive_statistics(
                    {ii: nc_ds.variables[ii][:] for ii in self._context.vars_emis},
                    "src",
                    row_data["rave_raw"],
                )
                src_desc.rename(columns={"FRP_MEAN": "frp_avg_hr"}, inplace=True)
            dst_desc_masked = create_descriptive_statistics(
                dst_data, "dst_masked", row_data["rave_interpolated"]
            )
            summary = pd.concat(
                [ii.transpose() for ii in [src_desc, dst_desc_unmasked, dst_desc_masked]]
            )
            summary.index.name = "variable"
            summary["forecast_date"] = row_data["forecast_date"]
            summary.reset_index(inplace=True)
            if self._interpolation_stats is None:
                self._interpolation_stats = summary
            else:
                self._interpolation_stats = pd.concat([self._interpolation_stats, summary])

        self.log("_run_interpolation_postprocessing: exit", level=logging.DEBUG)
