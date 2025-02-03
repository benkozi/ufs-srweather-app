import abc
from pathlib import Path
from typing import Tuple, Literal, Union, Dict, Any

import esmpy
import netCDF4 as nc
import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator

from smoke_dust.core.common import open_nc

NameListType = Tuple[str, ...]


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
            raise ValueError("if one corner name is supplied, then all must be supplied")
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
            set_variable_data(ds.variables[self.spec.x_center], self.dims, x_center_data)
            y_center_data = self.spec.get_y_data(self.value, staggerloc)
            set_variable_data(ds.variables[self.spec.y_center], self.dims, y_center_data)


class FieldWrapper(AbstractWrapper):
    value: esmpy.Field
    gwrap: GridWrapper

    def fill_nc_variable(self, path: Path):
        with open_nc(path, "a") as ds:
            var = ds.variables[self.value.name]
            set_variable_data(var, self.dims, self.value.data)


HasNcAttrsType = Union[nc.Dataset, nc.Variable]


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


def create_dimension_map(dims: DimensionCollection) -> Dict[str, int]:
    ret = {}
    for idx, dim in enumerate(dims.value):
        for name in dim.name:
            ret[name] = idx
    return ret


def load_variable_data(var: nc.Variable, target_dims: DimensionCollection) -> np.ndarray:
    slices = [slice(target_dims.get(ii).lower, target_dims.get(ii).upper) for ii in var.dimensions]
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
    slices = [slice(target_dims.get(ii).lower, target_dims.get(ii).upper) for ii in var.dimensions]
    var[*slices] = transposed_data
    return transposed_data


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
            grid_x_center_coords[:] = load_variable_data(ds.variables[self.spec.x_center], dims)
            grid_y_center_coords = self.spec.get_y_data(grid, staggerloc)
            grid_y_center_coords[:] = load_variable_data(ds.variables[self.spec.y_center], dims)

            if self.spec.has_corners:
                corner_dims = self._add_corner_coords_(ds, grid)
            else:
                corner_dims = None

            gwrap = GridWrapper(value=grid, dims=dims, spec=self.spec, corner_dims=corner_dims)
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

    def _add_corner_coords_(self, ds: nc.Dataset, grid: esmpy.Grid) -> DimensionCollection:
        staggerloc = esmpy.StaggerLoc.CORNER
        grid.add_coords(staggerloc)
        dims = self.spec.create_grid_dims(ds, grid, staggerloc)
        grid_x_corner_coords = self.spec.get_x_data(grid, staggerloc)
        grid_x_corner_coords[:] = load_variable_data(ds.variables[self.spec.x_corner], dims)
        grid_y_corner_coords = self.spec.get_y_data(grid, staggerloc)
        grid_y_corner_coords[:] = load_variable_data(ds.variables[self.spec.y_corner], dims)
        return dims


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
                target_dims = DimensionCollection(value=list(self.gwrap.dims.value) + [time_dim])
            field = esmpy.Field(
                self.gwrap.value,
                name=self.name,
                ndbounds=ndbounds,
                staggerloc=self.staggerloc,
            )
            field.data[:] = load_variable_data(ds.variables[self.name], target_dims)
            fwrap = FieldWrapper(value=field, dims=target_dims, gwrap=self.gwrap)
            return fwrap


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
    if isinstance(target, np.bool_):
        data.mask = np.zeros_like(data, dtype=bool)
        target = data.mask
    # Mask top and bottom rows
    target[:mask_width, :] = True
    target[-mask_width:, :] = True

    # Mask left and right columns
    target[:, :mask_width] = True
    target[:, -mask_width:] = True

    if data.shape != original_shape:
        raise ValueError("Data shape altered during masking.")
