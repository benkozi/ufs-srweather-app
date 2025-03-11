from functools import cached_property
import copy
from pathlib import Path
from typing import Any

import esmpy
import pandas as pd
from pydantic import BaseModel, ConfigDict

from smoke_dust.core.common import open_nc, create_template_emissions_file, create_sd_variable
from smoke_dust.core.context import SmokeDustContext, PredefinedGrid
from smoke_dust.core.regrid.common import NcToGrid, GridSpec, GridWrapper, FieldWrapper, NcToField
from smoke_dust.core.variable import SD_VARS


class EsmpyContext(BaseModel):
    model_config = ConfigDict(frozen=True)

    regrid_method: int
    zero_region: int
    debug: bool = False
    ignore_degenerate: bool = False
    unmapped_action: int = esmpy.UnmappedAction.ERROR


class RegridOptimizations(BaseModel):
    src_fwrap: FieldWrapper | None = None
    dst_fwrap: FieldWrapper | None = None
    regridder: esmpy.Regrid | None = None


class RegridOperationContext(BaseModel):
    model_config = ConfigDict(frozen=True)

    smoke_dust_context: SmokeDustContext
    esmpy_context: EsmpyContext
    cycle_metadata: pd.DataFrame
    create_weight_file: bool = False


class RegridFieldName(BaseModel):
    src: str
    dst: str


class RegridOperationSpec(BaseModel):
    #tdk: doc
    field_names: tuple[RegridFieldName, ...] #tdk:last: move to RegridOperationContext?
    src_path: Path
    dst_path: Path
    weight_path: Path | None = None
    optimizations: RegridOptimizations | None = None

    def get_optimization(self, name: str) -> FieldWrapper | esmpy.Regrid | None:
        if self.optimizations is None:
            return
        return getattr(self.optimizations, name)


class RaveToGridOperation:

    def __init__(self, context: RegridOperationContext, spec: RegridOperationSpec) -> None:
        self._context = context
        self._spec = spec

    def log(self, *args: Any, **kwargs: Any) -> None:
        """See ``SmokeDustContext.log``."""
        self._context.smoke_dust_context.log(*args, **kwargs)

    def run(self) -> RegridOptimizations | None:
        ...

    def finalize(self) -> None:
        self.log("finalize")

    @cached_property
    def _dst_fwrap(self) -> FieldWrapper:
        optimization = self._context.get_optimization("dst_fwrap")
        if optimization is not None:
            self.log("using optimization for dst_fwrap")
            assert isinstance(optimization, FieldWrapper)
            return optimization

        self.log("creating destination field")
        nc_to_field = NcToField(
                    path=self._spec.dst_path,
                    name=self._spec.dst_field_name,
                    gwrap=self._dst_output_gwrap,
                    dim_time=("t",),
                )
        return nc_to_field.create_field_wrapper()

    @cached_property
    def _dst_gwrap(self) -> GridWrapper:
        self.log("creating destination grid from RRFS grid file")
        dst_nc2grid = NcToGrid(
            path=self._context.smoke_dust_context.grid_out,
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
        self.log(f"creating source field: {self._context.src_field_name}")
        nc_to_field = NcToField(
            path=self._spec.src_path,
            name=self._spec.src_field_name,
            gwrap=self._src_gwrap,
            dim_time=("time",),
        )
        # tdk: optimization to provide an esmpy field and load into it
        return nc_to_field.create_field_wrapper()

    @cached_property
    def _src_gwrap(self) -> GridWrapper:
        optimization = self._context.get_optimization("src_fwrap")
        if optimization is not None:
            self.log("using optimization for src_gwrap")
            assert isinstance(optimization, FieldWrapper)
            return optimization.gwrap

        self.log("creating source grid from RAVE file")
        src_nc2grid = NcToGrid(
            path=self._context.smoke_dust_context.grid_in,
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
    def _regridder(self) -> esmpy.Regrid:
        optimization = self._context.get_optimization("regridder")
        if optimization is not None:
            self.log("using regridder optimization")
            assert isinstance(optimization, esmpy.Regrid)
            return optimization

        self.log("creating regridder")
        src_fwrap = self._src_fwrap
        dst_fwrap = self._dst_fwrap
        self.log(f"{src_fwrap.value.data.shape=}")
        self.log(f"{dst_fwrap.value.data.shape=}")
        if (
                self._context.smoke_dust_context.predef_grid == PredefinedGrid.RRFS_NA_13KM
                or self._context.smoke_dust_context.regrid_in_memory
        ):
            # ESMF does not like reading the weights for this field combination (rc=-1). The
            # error can be bypassed by creating weights in-memory.
            self.log("creating regridding in-memory")
            if self._context.create_weight_file:
                filename = str(self._context.weight_path)
            else:
                filename = None
            regridder = esmpy.Regrid(
                src_fwrap.value,
                dst_fwrap.value,
                regrid_method=self._context.esmpy_context.regrid_method,
                unmapped_action=self._context.esmpy_context.unmapped_action,
                ignore_degenerate=self._context.esmpy_context.ignore_degenerate,
                # Can be used to create a weight file for testing
                filename=filename
            )
        else:
            self.log("creating regridder from file")
            regridder = esmpy.RegridFromFile(
                src_fwrap.value,
                dst_fwrap.value,
                filename=str(self._context.weight_path),
            )
        return regridder


class RaveToGridProcessor:

    def __init__(self, context: RegridOperationContext):
        self._context = context

    def log(self, *args: Any, **kwargs: Any) -> None:
        """See ``SmokeDustContext.log``."""
        self._context.log(*args, **kwargs)

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

    def _run_impl_(self, rave_to_interpolate: pd.Series) -> None:
        esmpy_context = EsmpyContext(regrid_method=esmpy.RegridMethod.CONSERVE,
                                     zero_region=esmpy.Region.TOTAL,
                                     debug=self._context.esmpy_debug,
                                     ignore_degenerate=True,
                                     unmapped_action=esmpy.UnmappedAction.IGNORE)
        optimizations = RegridOptimizations()
        field_names = (RegridFieldName(src="FRP_MEAN", dst="frp_avg_hr"),
                       RegridFieldName(src="FRE", dst="FRE"))

        for row_idx, row_data in rave_to_interpolate.iterrows():
            row_dict = row_data.to_dict()
            self.log(f"processing RAVE interpolation row: {row_idx}, {row_dict}")

            forecast_date = row_data["forecast_date"]
            output_file_path = (
                self._context.intp_dir
                / f"{self._context.rave_to_intp}{forecast_date}00_{forecast_date}59.nc"
            )
            self.log(f"creating output file: {output_file_path}")
            with open_nc(output_file_path, "w") as nc_ds:
                create_template_emissions_file(nc_ds, self._context.grid_out_shape)
                for varname in ["frp_avg_hr", "FRE"]:
                    create_sd_variable(nc_ds, SD_VARS.get(varname))

            spec = RegridOperationSpec(field_names=field_names, src_path=row_data["rave_raw"],
                                       dst_path=output_file_path, optimizations=optimizations)
            operation = RaveToGridOperation(context=self._context, spec=spec)

            operation._dst_output_gwrap.fill_nc_variables(output_file_path)

