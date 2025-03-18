"""Tests the regrid processor."""

import glob
import itertools
import shutil
from pathlib import Path
from typing import Union, Iterator, Any
from unittest import case

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from _pytest.fixtures import SubRequest
from pydantic import BaseModel, Field, ValidationError, computed_field
from pytest_mock import MockerFixture

from smoke_dust.core.comm import COMM
from smoke_dust.core.common import ncdump, nccmp
from smoke_dust.core.context import SmokeDustContext, PredefinedGrid
from smoke_dust.core.describe import DescribeParams, describe
from smoke_dust.core.preprocessor import SmokeDustPreprocessor
from smoke_dust.core.regrid.operation.rave import RaveToGridStrategy, RaveToGeomProcessor
from smoke_dust.core.regrid.processor import SmokeDustRegridProcessor
from test_python.test_smoke_dust.conftest import (
    FakeGridOutShape,
    create_fake_context,
    create_file_hash,
)


class DataForTest(BaseModel):
    """Model holds objects needed for testing."""

    model_config = {"arbitrary_types_allowed": True}
    context: SmokeDustContext
    preprocessor: SmokeDustPreprocessor

    @computed_field
    def baseline_filename(self) -> str:
        match self.context.predef_grid:
            case PredefinedGrid.MPAS_NA_15KM:
                return "MPAS_NA_15km_intp_baseline.nc"
            case PredefinedGrid.RRFS_CONUS_3KM:
                return "RRFS_CONUS_3km_intp_baseline.nc"
            case _:
                raise NotImplementedError(self.context.predef_grid)

class FakeGridParams(BaseModel):
    """Model for a fake RAVE/RRFS data file definition."""

    path: Path = Field(description="Path to the output data file.")
    shape: FakeGridOutShape = Field(description="Output grid shape.")
    with_corners: bool = Field(
        description="If True, create the output grid with corners", default=True
    )
    fields: Union[list[str], None] = Field(
        description="If provided, a list of field names to create in the output file.", default=None
    )
    min_lon: int = Field(
        description="The minimum longitude value as origin for grid generation.", default=230
    )
    min_lat: int = Field(
        description="The minimum latitude value as origin for grid generation.", default=25
    )
    ntime: Union[int, None] = Field(
        description="If provided, create the output fields with this many time steps.", default=1
    )


def iterate_params() -> Iterator[dict[str, Any]]:

    def element_iterator(key: str, value: list[Any]) -> Iterator[dict[str, Any]]:
        for element in value:
            yield {key: element}

    #tdk:uncomm
    parms = {'regrid_in_memory': [
        True,
                                  False
                                  ],
    'predef_grid': [
        PredefinedGrid.RRFS_CONUS_3KM,
                    PredefinedGrid.MPAS_NA_15KM
                    ]}
    iterators = (element_iterator(k, v) for k, v in parms.items())
    for elements in itertools.product(*iterators):
        yld = {}
        for element in elements:
            yld.update(element)
        yield yld



@pytest.fixture(params=iterate_params(), ids=lambda p: f"params={p}")
def data_for_test(
    request: SubRequest,
    tmp_path_shared: Path,
    fake_grid_out_shape: FakeGridOutShape,
    bin_dir: Path,
) -> DataForTest:
    """Create test data including any required data files."""

    match request.param['predef_grid']:
        case PredefinedGrid.MPAS_NA_15KM:
            if COMM.rank == 0:
                shutil.copy(bin_dir / "na15km.init.scrip.nc", tmp_path_shared / "ds_out_base.nc")
                _ = create_fake_rave_and_rrfs_like_data(
                    FakeGridParams(path=tmp_path_shared / "grid_in.nc", shape=fake_grid_out_shape, fields=["area"], ntime=None)
                )
        case _:
            weight_file = "weight_file.nc"
            if COMM.rank == 0:
                shutil.copy(bin_dir / weight_file, tmp_path_shared / "weight_file.nc")
                for name in ["ds_out_base.nc", "grid_in.nc"]:
                    path = tmp_path_shared / name
                    _ = create_fake_rave_and_rrfs_like_data(
                        FakeGridParams(path=path, shape=fake_grid_out_shape, fields=["area"], ntime=None)
                    )
    try:
        context = create_fake_context(tmp_path_shared, overrides=request.param)
    except ValidationError:
        assert request.param['predef_grid'] == PredefinedGrid.MPAS_NA_15KM
        assert request.param["regrid_in_memory"] == False
        pytest.xfail("validation error expected")
    preprocessor = SmokeDustPreprocessor(context)
    if COMM.rank == 0:
        for date in preprocessor.cycle_dates:
            path = tmp_path_shared / f"Hourly_Emissions_3km_{date}_{date}.nc"
            _ = create_fake_rave_and_rrfs_like_data(
                FakeGridParams(path=path, shape=fake_grid_out_shape, fields=["FRP_MEAN", "FRE"])
            )
    return DataForTest(context=context, preprocessor=preprocessor)


def create_analytic_data_array(
    dims: list[str],
    lon_mesh: np.ndarray,
    lat_mesh: np.ndarray,
    ntime: Union[int, None] = None,
) -> xr.DataArray:
    """
    Create an analytic data array using lat/lon values.

    Args:
        dims: Names of the lat/lon dimensions. For example `["lat", "lon"]`.
        lon_mesh: A two-dimensional array of longitude values.
        lat_mesh: A two-dimensional array of latitude values.
        ntime: If provided, create the output data array with the provided number of time steps.

    Returns:
        An analytic data array.
    """
    # tdk:last: remove duplicate create_data_array
    deg_to_rad = 3.141592653589793 / 180.0
    analytic_data = 2.0 + np.cos(deg_to_rad * lon_mesh) ** 2 * np.cos(
        2.0 * deg_to_rad * (90.0 - lat_mesh)
    )
    if ntime is not None:
        time_modifier = np.arange(1, ntime + 1).reshape(ntime, 1, 1)
        analytic_data = analytic_data.reshape([1] + list(analytic_data.shape))
        analytic_data = np.repeat(analytic_data, ntime, axis=0)
        analytic_data = time_modifier * analytic_data
    return xr.DataArray(
        analytic_data,
        dims=dims,
    )


def create_fake_rave_and_rrfs_like_data(params: FakeGridParams) -> xr.Dataset:
    """
    Create fake RAVE and RRFS data. These data files share a common grid.

    Returns:
        The created dataset object.
    """
    if params.path.exists():
        raise ValueError(f"path exists: {params.path}")
    lon = np.arange(params.shape.x_size, dtype=float) + params.min_lon
    lat = np.arange(params.shape.y_size, dtype=float) + params.min_lat
    lon_mesh, lat_mesh = np.meshgrid(lon, lat)
    nc_ds = xr.Dataset()
    dims = ["grid_yt", "grid_xt"]
    nc_ds["grid_lont"] = xr.DataArray(lon_mesh, dims=dims)
    nc_ds["grid_latt"] = xr.DataArray(lat_mesh, dims=dims)
    if params.with_corners:
        lonc = np.hstack((lon - 0.5, [lon[-1] + 0.5]))
        latc = np.hstack((lat - 0.5, [lat[-1] + 0.5]))
        lonc_mesh, latc_mesh = np.meshgrid(lonc, latc)
        nc_ds["grid_lon"] = xr.DataArray(lonc_mesh, dims=["grid_y", "grid_x"])
        nc_ds["grid_lat"] = xr.DataArray(latc_mesh, dims=["grid_y", "grid_x"])
    if params.fields is not None:
        if params.ntime is not None:
            field_dims = ["time"] + dims
        else:
            field_dims = dims
        for field in params.fields:
            nc_ds[field] = create_analytic_data_array(
                field_dims, lon_mesh, lat_mesh, ntime=params.ntime
            )
    nc_ds.to_netcdf(params.path)
    return nc_ds


def describe_mpas_output(path: Path) -> pd.DataFrame:
    params = DescribeParams(
        namespace="mpas.rave",
        files=(path,),
        varnames=(
            "frp_avg_hr", "FRE", "latCell", "lonCell"
        ),
    )
    return describe(params)


def describe_output(path: Path) -> pd.DataFrame:
    params = DescribeParams(
        namespace="grid.rave",
        files=(path,),
        varnames=(
            "frp_avg_hr", "FRE",
        ),
    )
    return describe(params)


class TestSmokeDustRegridProcessor:  # pylint: disable=too-few-public-methods
    """Tests for the smoke/dust regrid processor."""

    @pytest.mark.mpi
    def test_run(
        self,
        data_for_test: DataForTest,  # pylint: disable=redefined-outer-name
        mocker: MockerFixture,
        tmp_path_shared: Path,
        bin_dir: Path
    ) -> None:
        """Test the regrid processor."""
        #tdk:story: regrid other smoke/dust inputs (vegmap, etc)
        COMM.barrier()
        spy1 = mocker.spy(RaveToGeomProcessor, "run")
        spy2 = mocker.spy(RaveToGridStrategy, "run")
        regrid_processor = SmokeDustRegridProcessor(data_for_test.context)
        regrid_processor.run(data_for_test.preprocessor.cycle_metadata)
        spy1.assert_called_once()
        assert spy2.call_count == 24

        if COMM.rank == 0:
            interpolated_files = glob.glob(
                f"*{data_for_test.context.rave_to_intp}*nc", root_dir=tmp_path_shared
            )
            assert len(interpolated_files) == 24
            control_file = bin_dir / "baseline" / data_for_test.baseline_filename
            for intp_file in interpolated_files:
                fpath = tmp_path_shared / intp_file
                # ncdump(fpath, header_only=True) #tdk:rm
                # df = describe_mpas_output(fpath) #tdk:rm
                # df = describe_output(fpath) #tdk:rm
                # print(df['sum']) #tdk:rm
                # return #tdk:rm

                # assert create_file_hash(fpath) == data_for_test.hash

                nccmp(control_file, Path(fpath))
