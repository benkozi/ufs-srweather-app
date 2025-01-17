import abc
import logging
from copy import copy, deepcopy
from pathlib import Path
from typing import Any, Union, Dict, Sequence, Tuple, Literal

import esmpy
import netCDF4 as nc
import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator, field_validator

from smoke_dust_context import SmokeDustContext
import pandas as pd

from smoke_dust_common import create_template_emissions_file, \
    create_sd_variable, create_descriptive_statistics
from smoke_dust_common import open_nc


class SmokeDustRegridProcessor:

    def __init__(self, context: SmokeDustContext):
        self._context = context

        # Holds interpolation descriptive statistics
        self._interpolation_stats = None

    def log(self, *args: Any, **kwargs: Any) -> None:
        #tdk: superclass for processors
        self._context.log(*args, **kwargs)

    def run(self, forecast_metadata: pd.DataFrame) -> None:
        # Select which RAVE files to interpolate
        rave_to_interpolate = forecast_metadata[
            forecast_metadata['rave_interpolated'].isnull() & ~forecast_metadata['rave_raw'].isnull()]

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
            forecast_metadata.loc[row_idx, 'rave_interpolated'] = output_file_path
            row_data['rave_interpolated'] = output_file_path

            if self._context.rank == 0:
                self._interpolation_postprocessing_(row_data)

        if self._context.rank == 0 and self._context.should_calc_desc_stats and self._interpolation_stats is not None:
            forecast_dates = forecast_metadata['forecast_date']
            stats_path = self._context.intp_dir / f"stats_regridding_{forecast_dates.min()}_{forecast_dates.max()}.csv"
            self.log(f"writing interpolation statistics: {stats_path=}")
            self._interpolation_stats.to_csv(stats_path, index=False)

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


HasNcAttrsType = Union[nc.Dataset, nc.Variable]


def copy_nc_attrs(src: HasNcAttrsType, dst: HasNcAttrsType) -> None:
    for attr in src.ncattrs():
        if attr.startswith("_"):
            continue
        setattr(dst, attr, getattr(src, attr))


def resize_nc(
    src_path: Path,
    dst_path: Path,
    new_sizes: Dict[str, int],
    copy_values_for: Sequence[str] | None = None,
) -> None:
    with open_nc(src_path, mode="r") as src:
        with open_nc(dst_path, mode="w") as dst:
            copy_nc_attrs(src, dst)
            for dim in src.dimensions:
                size = get_aliased_key(new_sizes, dim)
                dst.createDimension(dim, size=size)
            for varname, var in src.variables.items():
                fill_value = (
                    getattr(var, "_FillValue") if hasattr(var, "_FillValue") else None
                )
                new_var = dst.createVariable(
                    varname, var.dtype, var.dimensions, fill_value=fill_value
                )
                copy_nc_attrs(var, new_var)
                if copy_values_for and varname in copy_values_for:
                    new_var[:] = var[:]


NameListType = Tuple[str, ...]


def get_aliased_key(source: Dict, keys: NameListType | str) -> Any:
    if isinstance(keys, str):
        keys_to_find = (keys,)
    else:
        keys_to_find = keys
    for key in keys_to_find:
        try:
            return source[key]
        except KeyError:
            continue
    raise ValueError(f"key not found: {keys}")


def get_nc_dimension(ds: nc.Dataset, names: NameListType) -> nc.Dimension:
    return get_aliased_key(ds.dimensions, names)


class Dimension(BaseModel):
    name: NameListType
    size: int
    lower: int
    upper: int
    staggerloc: int
    coordinate_type: Literal["y", "x", "time"]


class DimensionCollection(BaseModel):
    value: Tuple[Dimension, ...]

    def get(self, name: str | NameListType) -> Dimension:
        if isinstance(name, str):
            name_to_find = (name,)
        else:
            name_to_find = name
        for jj in name_to_find:
            for ii in self.value:
                if jj in ii.name:
                    return ii
        raise ValueError(f"dimension not found: {name}")


def create_dimension_map(dims: DimensionCollection) -> Dict[str, int]:
    ret = {}
    for idx, dim in enumerate(dims.value):
        for name in dim.name:
            ret[name] = idx
    return ret


def load_variable_data(
    var: nc.Variable, target_dims: DimensionCollection
) -> np.ndarray:
    slices = [
        slice(target_dims.get(ii).lower, target_dims.get(ii).upper)
        for ii in var.dimensions
    ]
    raw_data = var[*slices]
    dim_map = {dim: ii for ii, dim in enumerate(var.dimensions)}
    axes = [get_aliased_key(dim_map, ii.name) for ii in target_dims.value]
    transposed_data = raw_data.transpose(axes)
    return transposed_data


def set_variable_data(
    var: nc.Variable, target_dims: DimensionCollection, target_data: np.ndarray
) -> np.ndarray:
    dim_map = create_dimension_map(target_dims)
    axes = [get_aliased_key(dim_map, ii) for ii in var.dimensions]
    transposed_data = target_data.transpose(axes)
    slices = [
        slice(target_dims.get(ii).lower, target_dims.get(ii).upper)
        for ii in var.dimensions
    ]
    var[*slices] = transposed_data
    return transposed_data


class AbstractWrapper(abc.ABC, BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    dims: DimensionCollection


class GridSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    x_center: str
    y_center: str
    x_dim: NameListType
    y_dim: NameListType
    x_corner: str | None = None
    y_corner: str | None = None
    x_corner_dim: NameListType | None = None
    y_corner_dim: NameListType | None = None
    x_index: int = 0
    y_index: int = 1

    @model_validator(mode="after")
    def _validate_model_(self) -> "GridSpec":
        corner_meta = [
            self.x_corner,
            self.y_corner,
            self.x_corner_dim,
            self.y_corner_dim,
        ]
        is_given_sum = sum([ii is not None for ii in corner_meta])
        if is_given_sum > 0 and is_given_sum != len(corner_meta):
            raise ValueError(
                "if one corner name is supplied, then all must be supplied"
            )
        return self

    @property
    def has_corners(self) -> bool:
        return self.x_corner is not None

    def get_x_corner(self) -> str:
        if self.x_corner is None:
            raise ValueError
        return self.x_corner

    def get_y_corner(self) -> str:
        if self.y_corner is None:
            raise ValueError
        return self.y_corner

    def get_x_data(self, grid: esmpy.Grid, staggerloc: esmpy.StaggerLoc) -> np.ndarray:
        return grid.get_coords(self.x_index, staggerloc=staggerloc)

    def get_y_data(self, grid: esmpy.Grid, staggerloc: esmpy.StaggerLoc) -> np.ndarray:
        return grid.get_coords(self.y_index, staggerloc=staggerloc)

    def create_grid_dims(
        self, ds: nc.Dataset, grid: esmpy.Grid, staggerloc: esmpy.StaggerLoc
    ) -> DimensionCollection:
        if staggerloc == esmpy.StaggerLoc.CENTER:
            x_dim, y_dim = self.x_dim, self.y_dim
        elif staggerloc == esmpy.StaggerLoc.CORNER:
            x_dim, y_dim = self.x_corner_dim, self.y_corner_dim
        else:
            raise NotImplementedError(staggerloc)
        x_dimobj = Dimension(
            name=x_dim,
            size=get_nc_dimension(ds, x_dim).size,
            lower=grid.lower_bounds[staggerloc][self.x_index],
            upper=grid.upper_bounds[staggerloc][self.x_index],
            staggerloc=staggerloc,
            coordinate_type="x",
        )
        y_dimobj = Dimension(
            name=y_dim,
            size=get_nc_dimension(ds, y_dim).size,
            lower=grid.lower_bounds[staggerloc][self.y_index],
            upper=grid.upper_bounds[staggerloc][self.y_index],
            staggerloc=staggerloc,
            coordinate_type="y",
        )
        if self.x_index == 0:
            value = [x_dimobj, y_dimobj]
        elif self.x_index == 1:
            value = [y_dimobj, x_dimobj]
        else:
            raise NotImplementedError(self.x_index, self.y_index)
        return DimensionCollection(value=value)


class GridWrapper(AbstractWrapper):
    value: esmpy.Grid
    spec: GridSpec
    corner_dims: DimensionCollection | None = None

    def fill_nc_variables(self, path: Path):
        if self.corner_dims is not None:
            raise NotImplementedError
        with open_nc(path, "a") as ds:
            staggerloc = esmpy.StaggerLoc.CENTER
            x_center_data = self.spec.get_x_data(self.value, staggerloc)
            set_variable_data(
                ds.variables[self.spec.x_center], self.dims, x_center_data
            )
            y_center_data = self.spec.get_y_data(self.value, staggerloc)
            set_variable_data(
                ds.variables[self.spec.y_center], self.dims, y_center_data
            )


class NcToGrid(BaseModel):
    path: Path
    spec: GridSpec

    def create_grid_wrapper(self) -> GridWrapper:
        with open_nc(self.path, "r") as ds:
            grid_shape = self._create_grid_shape_(ds)
            staggerloc = esmpy.StaggerLoc.CENTER
            grid = esmpy.Grid(
                grid_shape,
                staggerloc=staggerloc,
                coord_sys=esmpy.CoordSys.SPH_DEG,
            )
            dims = self.spec.create_grid_dims(ds, grid, staggerloc)
            grid_x_center_coords = self.spec.get_x_data(grid, staggerloc)
            grid_x_center_coords[:] = load_variable_data(
                ds.variables[self.spec.x_center], dims
            )
            grid_y_center_coords = self.spec.get_y_data(grid, staggerloc)
            grid_y_center_coords[:] = load_variable_data(
                ds.variables[self.spec.y_center], dims
            )

            if self.spec.has_corners:
                corner_dims = self._add_corner_coords_(ds, grid)
            else:
                corner_dims = None

            gwrap = GridWrapper(
                value=grid, dims=dims, spec=self.spec, corner_dims=corner_dims
            )
            return gwrap

    def _create_grid_shape_(self, ds: nc.Dataset) -> np.ndarray:
        x_size = get_nc_dimension(ds, self.spec.x_dim).size
        y_size = get_nc_dimension(ds, self.spec.y_dim).size
        if self.spec.x_index == 0:
            grid_shape = (x_size, y_size)
        elif self.spec.x_index == 1:
            grid_shape = (y_size, x_size)
        else:
            raise NotImplementedError(self.spec.x_index, self.spec.y_index)
        return np.array(grid_shape)

    def _add_corner_coords_(
        self, ds: nc.Dataset, grid: esmpy.Grid
    ) -> DimensionCollection:
        staggerloc = esmpy.StaggerLoc.CORNER
        grid.add_coords(staggerloc)
        dims = self.spec.create_grid_dims(ds, grid, staggerloc)
        grid_x_corner_coords = self.spec.get_x_data(grid, staggerloc)
        grid_x_corner_coords[:] = load_variable_data(
            ds.variables[self.spec.x_corner], dims
        )
        grid_y_corner_coords = self.spec.get_y_data(grid, staggerloc)
        grid_y_corner_coords[:] = load_variable_data(
            ds.variables[self.spec.y_corner], dims
        )
        return dims


class FieldWrapper(AbstractWrapper):
    value: esmpy.Field
    gwrap: GridWrapper

    def fill_nc_variable(self, path: Path):
        with open_nc(path, "a") as ds:
            var = ds.variables[self.value.name]
            set_variable_data(var, self.dims, self.value.data)


class NcToField(BaseModel):
    path: Path
    name: str
    gwrap: GridWrapper
    dim_time: NameListType | None = None
    staggerloc: int = esmpy.StaggerLoc.CENTER

    def create_field_wrapper(self) -> FieldWrapper:
        with open_nc(self.path, "r") as ds:
            if self.dim_time is None:
                ndbounds = None
                target_dims = self.gwrap.dims
            else:
                ndbounds = (len(get_nc_dimension(ds, self.dim_time)),)
                time_dim = Dimension(
                    name=self.dim_time,
                    size=ndbounds[0],
                    lower=0,
                    upper=ndbounds[0],
                    staggerloc=self.staggerloc,
                    coordinate_type="time",
                )
                target_dims = DimensionCollection(
                    value=list(self.gwrap.dims.value) + [time_dim]
                )
            field = esmpy.Field(
                self.gwrap.value,
                name=self.name,
                ndbounds=ndbounds,
                staggerloc=self.staggerloc,
            )
            field.data[:] = load_variable_data(ds.variables[self.name], target_dims)
            fwrap = FieldWrapper(value=field, dims=target_dims, gwrap=self.gwrap)
            return fwrap


class FieldWrapperCollection(BaseModel):
    value: Tuple[FieldWrapper, ...]

    def fill_nc_variables(self, path: Path) -> None:
        for fwrap in self.value:
            fwrap.fill_nc_variable(path)

    @field_validator("value", mode="before")
    @classmethod
    def _validate_value_(
        cls, value: Tuple[FieldWrapper, ...]
    ) -> Tuple[FieldWrapper, ...]:
        if len(set([id(ii.value.grid) for ii in value])) != 1:
            raise ValueError("all fields must share the same grid")
        return value


def mask_edges(data: np.ma.MaskedArray, mask_width: int = 1) -> None:
    """
    Mask edges of domain for interpolation.

    Args:
        data: The masked array to alter
        mask_width: The width of the mask at each edge

    Returns:
        A numpy array of the masked edges
    """
    if data.ndim != 2:
        raise ValueError(f"{data.ndim=}")

    original_shape = data.shape
    if mask_width < 1:
        return  # No masking if mask_width is less than 1

    target = data.mask
    # Mask top and bottom rows
    target[:mask_width, :] = True
    target[-mask_width:, :] = True

    # Mask left and right columns
    target[:, :mask_width] = True
    target[:, -mask_width:] = True

    if data.shape != original_shape:
        raise ValueError("Data shape altered during masking.")
