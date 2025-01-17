from contextlib import contextmanager
from pathlib import Path
from typing import Tuple, Literal, Dict

import numpy as np
import netCDF4 as nc

import pandas as pd

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


def create_descriptive_statistics(container: Dict[str, np.ma.MaskedArray], origin: Literal["src", "dst_unmasked", "dst_masked", "derived"], path: Path) -> pd.DataFrame:
    df = pd.DataFrame.from_dict({k: v.filled(np.nan).ravel() for k, v in container.items()})
    desc = df.describe()
    adds = {}
    for field_name in container.keys():
        adds[field_name] = [df[field_name].sum(), df[field_name].isnull().sum(), origin, path]
    desc = pd.concat([desc, pd.DataFrame(data=adds, index=['sum', 'count_null', "origin", "path"])])
    return desc


