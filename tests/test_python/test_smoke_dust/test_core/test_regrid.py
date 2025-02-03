import glob
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr
from _pytest.fixtures import SubRequest
from pydantic import BaseModel
from pytest_mock import MockerFixture

from smoke_dust.core.context import SmokeDustContext
from smoke_dust.core.preprocessor import SmokeDustPreprocessor
from smoke_dust.core.regrid.processor import SmokeDustRegridProcessor
from test_python.test_smoke_dust.conftest import (
    FakeGridOutShape,
    create_fake_context,
    create_file_hash,
)


def ncdump(path: Path, header_only: bool = True) -> Any:
    args = ["ncdump"]
    if header_only:
        args.append("-h")
    args.append(str(path))
    ret = subprocess.check_output(args)
    print(ret.decode(), flush=True)
    return ret


class DataForTest(BaseModel):
    model_config = dict(arbitrary_types_allowed=True)
    context: SmokeDustContext
    preprocessor: SmokeDustPreprocessor


@pytest.fixture(params=[True, False], ids=lambda p: f"regrid_in_memory={p}")
def data_for_test(
    request: SubRequest,
    tmp_path: Path,
    fake_grid_out_shape: FakeGridOutShape,
    bin_dir: Path,
) -> DataForTest:
    weight_file = "weight_file.nc"
    shutil.copy(bin_dir / weight_file, tmp_path / "weight_file.nc")
    for name in ["ds_out_base.nc", "grid_in.nc"]:
        path = tmp_path / name
        create_fake_rave_and_rrfs_like_data(path, fake_grid_out_shape, fields=["area"], ntime=None)
    context = create_fake_context(tmp_path, extra=dict(regrid_in_memory=request.param))
    preprocessor = SmokeDustPreprocessor(context)
    for date in preprocessor.forecast_dates:
        path = tmp_path / f"Hourly_Emissions_3km_{date}_{date}.nc"
        create_fake_rave_and_rrfs_like_data(path, fake_grid_out_shape, fields=["FRP_MEAN", "FRE"])
    return DataForTest(context=context, preprocessor=preprocessor)


def create_analytic_data_array(
    dims: list[str],
    lon_mesh: np.ndarray,
    lat_mesh: np.ndarray,
    ntime: int | None = None,
) -> xr.DataArray:
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


def create_fake_rave_and_rrfs_like_data(
    path: Path,
    shape: FakeGridOutShape,
    with_corners: bool = True,
    fields: list[str] | None = None,
    min_lon: int = 230,
    min_lat: int = 25,
    ntime: int | None = 1,
) -> xr.Dataset:
    if path.exists():
        raise ValueError(f"path exists: {path}")
    lon = np.arange(shape.x_size, dtype=float) + min_lon
    lat = np.arange(shape.y_size, dtype=float) + min_lat
    lon_mesh, lat_mesh = np.meshgrid(lon, lat)
    ds = xr.Dataset()
    dims = ["grid_yt", "grid_xt"]
    ds["grid_lont"] = xr.DataArray(lon_mesh, dims=dims)
    ds["grid_latt"] = xr.DataArray(lat_mesh, dims=dims)
    if with_corners:
        lonc = np.hstack((lon - 0.5, [lon[-1] + 0.5]))
        latc = np.hstack((lat - 0.5, [lat[-1] + 0.5]))
        lonc_mesh, latc_mesh = np.meshgrid(lonc, latc)
        ds["grid_lon"] = xr.DataArray(lonc_mesh, dims=["grid_y", "grid_x"])
        ds["grid_lat"] = xr.DataArray(latc_mesh, dims=["grid_y", "grid_x"])
    if fields is not None:
        if ntime is not None:
            field_dims = ["time"] + dims
        else:
            field_dims = dims
        for field in fields:
            ds[field] = create_analytic_data_array(field_dims, lon_mesh, lat_mesh, ntime=ntime)
    ds.to_netcdf(path)
    return ds


class TestSmokeDustRegridProcessor:
    def test_run(self, data_for_test: DataForTest, mocker: MockerFixture, tmp_path: Path) -> None:
        spy1 = mocker.spy(SmokeDustRegridProcessor, "_run_impl_")
        regrid_processor = SmokeDustRegridProcessor(data_for_test.context)
        regrid_processor.run(data_for_test.preprocessor.forecast_metadata)
        spy1.assert_called_once()
        interpolated_files = glob.glob(
            f"*{data_for_test.context.rave_to_intp}*nc", root_dir=tmp_path
        )
        assert len(interpolated_files) == 24
        for f in interpolated_files:
            fpath = tmp_path / f
            assert create_file_hash(fpath) == "8e90b769137aad054a2e49559d209c4d"
