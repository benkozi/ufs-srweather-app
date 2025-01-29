from contextlib import contextmanager
from pathlib import Path
from typing import Tuple, Literal, Dict

import netCDF4 as nc
import numpy as np
import pandas as pd
from mpi4py import MPI

from smoke_dust.core.variable import SmokeDustVariable, SD_VARS


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
    ds: nc.Dataset,
    sd_variable: SmokeDustVariable,
) -> None:
    """
    Create a smoke/dust netCDF spatial coordinate variable.

    Args:
        ds: Dataset to update
        sd_variable: Contains variable metadata
    """
    var_out = ds.createVariable(
        sd_variable.name, "f4", ("lat", "lon"), fill_value=sd_variable.fill_value_float
    )
    var_out.units = sd_variable.units
    var_out.long_name = sd_variable.long_name
    var_out.standard_name = sd_variable.name
    var_out.FillValue = sd_variable.fill_value_str
    var_out.coordinates = "geolat geolon"


def create_sd_variable(
    ds: nc.Dataset,
    sd_variable: SmokeDustVariable,
    fill_first_time_index: bool = True,
) -> None:
    """
    Create a smoke/dust netCDF variable.

    Args:
        ds: Dataset to update
        sd_variable: Contains variable metadata
        fill_first_time_index: If True, fill the first time index with provided `fill_value_float`
    """
    var_out = ds.createVariable(
        sd_variable.name,
        "f4",
        ("t", "lat", "lon"),
        fill_value=sd_variable.fill_value_float,
    )
    var_out.units = sd_variable.units
    var_out.long_name = sd_variable.long_name
    var_out.standard_name = sd_variable.long_name
    var_out.FillValue = sd_variable.fill_value_str
    var_out.coordinates = "t geolat geolon"
    if fill_first_time_index:
        try:
            var_out.set_collective(True)
        except RuntimeError:
            # Allow this function to work with parallel and non-parallel datasets. If the dataset is not opened in parallel
            # this error message is returned: RuntimeError: NetCDF: Parallel operation on file opened for non-parallel access
            pass
        var_out[0, :, :] = sd_variable.fill_value_float
        try:
            var_out.set_collective(False)
        except RuntimeError:
            pass


def create_template_emissions_file(
    ds: nc.Dataset, grid_shape: Tuple[int, int], is_dummy: bool = False
):
    ds.createDimension("t", None)
    ds.createDimension("lat", grid_shape[0])
    ds.createDimension("lon", grid_shape[1])
    setattr(ds, "PRODUCT_ALGORITHM_VERSION", "Beta")
    setattr(ds, "TIME_RANGE", "1 hour")
    setattr(ds, "is_dummy", str(is_dummy))

    for varname in ["geolat", "geolon"]:
        create_sd_coordinate_variable(ds, SD_VARS.get(varname))


def create_descriptive_statistics(
    container: Dict[str, np.ma.MaskedArray],
    origin: Literal["src", "dst_unmasked", "dst_masked", "derived"],
    path: Path,
) -> pd.DataFrame:
    df = pd.DataFrame.from_dict(
        {k: v.filled(np.nan).ravel() for k, v in container.items()}
    )
    desc = df.describe()
    adds = {}
    for field_name in container.keys():
        adds[field_name] = [
            df[field_name].sum(),
            df[field_name].isnull().sum(),
            origin,
            path,
        ]
    desc = pd.concat(
        [desc, pd.DataFrame(data=adds, index=["sum", "count_null", "origin", "path"])]
    )
    return desc
