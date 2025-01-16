import abc
from contextlib import contextmanager
from pathlib import Path
from typing import Tuple, Literal, Dict, Sequence, Any, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator, model_validator
import esmpy
import netCDF4 as nc

from mpi4py import MPI




@contextmanager
def open_nc(
    path: Path,
    mode: Literal["r", "w", "a"] = "r",
    clobber: bool = False,
    parallel: bool = True,
) -> nc.Dataset:
    ds = nc.Dataset(
        path,
        mode=mode,
        clobber=clobber,
        parallel=parallel,
        comm=MPI.COMM_WORLD,
        info=MPI.Info(),
    )
    try:
        yield ds
    finally:
        ds.close()

def create_sd_coordinate_variable(
    ds: nc.Dataset, varname: str, long_name: str, units: str, fill_value_str: str, fill_value_float: float
) -> None:
    """
    Create a smoke/dust netCDF spatial coordinate variable.

    Args:
        ds: Dataset to update
        varname: Variable name to create
        long_name: Variable long name
        units: Variable units
        fill_value_str: The string representation of the fill value
        fill_value_float: The float representation of the fill value
    """
    var_out = ds.createVariable(varname, "f4", ("lat", "lon"), fill_value=fill_value_float)
    var_out.units = units
    var_out.long_name = long_name
    var_out.standard_name = varname
    var_out.FillValue = fill_value_str
    var_out.coordinates = "geolat geolon"


def create_sd_variable(
    ds: nc.Dataset, varname: str, long_name: str, units: str, fill_value_str: str, fill_value_float: float, fill_first_time_index: bool = True
) -> None:
    """
    Create a smoke/dust netCDF variable.

    Args:
        ds: Dataset to update
        varname: Name of the variable to create
        long_name: Long name of the variable to create
        units: Units of the variable to create
        fill_value_str: The string representation of the fill value
        fill_value_float: The float representation of the fill value
        fill_first_time_index: If True, fill the first time index with provided `fill_value_float`
    """
    var_out = ds.createVariable(varname, "f4", ("t", "lat", "lon"), fill_value=fill_value_float)
    var_out.units = units
    var_out.long_name = long_name
    var_out.standard_name = long_name
    var_out.FillValue = fill_value_str
    var_out.coordinates = "t geolat geolon"
    if fill_first_time_index:
        try:
            var_out.set_collective(True)
        except RuntimeError:
            # Allow this function to work with parallel and non-parallel datasets. If the dataset is not opened in parallel
            # this error message is returned: RuntimeError: NetCDF: Parallel operation on file opened for non-parallel access
            pass
        var_out[0, :, :] = fill_value_float
        try:
            var_out.set_collective(False)
        except RuntimeError:
            pass

def create_template_emissions_file(ds: nc.Dataset, grid_shape: Tuple[int, int]):
    ds.createDimension("t", None)
    ds.createDimension("lat", grid_shape[0])
    ds.createDimension("lon", grid_shape[1])
    setattr(ds, "PRODUCT_ALGORITHM_VERSION", "Beta")
    setattr(ds, "TIME_RANGE", "1 hour")

    create_sd_coordinate_variable(ds, "geolat", "cell center latitude", "degrees_north", "-9999.f", -9999.0)
    create_sd_coordinate_variable(ds, "geolon", "cell center longitude", "degrees_east", "-9999.f", -9999.0)

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
