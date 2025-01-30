import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Type

import netCDF4 as nc
import numpy as np
import pandas as pd
import pytest
from _pytest.fixtures import SubRequest
from pydantic import BaseModel
from pytest_mock import MockerFixture

from smoke_dust.core.context import SmokeDustContext
from smoke_dust.core.cycle import (
    AbstractSmokeDustCycleProcessor,
    SmokeDustCycleOne,
    SmokeDustCycleTwo,
)
from smoke_dust.core.preprocessor import SmokeDustPreprocessor


@dataclass
class FakeGridOutShape:
    y_size: int = 5
    x_size: int = 10

    @property
    def as_tuple(self) -> tuple[int, int]:
        return self.y_size, self.x_size


@pytest.fixture
def fake_grid_out_shape() -> FakeGridOutShape:
    return FakeGridOutShape()


def create_restart_files(
    root_dir: Path, forecast_dates: pd.DatetimeIndex, shape: FakeGridOutShape
) -> None:
    restart_dir = root_dir / "RESTART"
    restart_dir.mkdir()
    for date in forecast_dates:
        restart_file = restart_dir / f"{date[:8]}.{date[8:10]}0000.phy_data.nc"
        with nc.Dataset(restart_file, "w") as ds:
            ds.createDimension("Time")
            ds.createDimension("yaxis_1", shape.y_size)
            ds.createDimension("xaxis_1", shape.x_size)
            totprcp_ave = ds.createVariable(
                "totprcp_ave", "f4", ("Time", "yaxis_1", "xaxis_1")
            )
            totprcp_ave[0, ...] = np.ones(shape.as_tuple)
            rrfs_hwp_ave = ds.createVariable(
                "rrfs_hwp_ave", "f4", ("Time", "yaxis_1", "xaxis_1")
            )
            rrfs_hwp_ave[0, ...] = totprcp_ave[:] + 2


def create_rave_interpolated(
    root_dir: Path,
    forecast_dates: pd.DatetimeIndex,
    shape: FakeGridOutShape,
    rave_to_intp: str,
) -> None:
    for date in forecast_dates:
        intp_file = root_dir / f"{rave_to_intp}{date}00_{date}59.nc"
        dims = ("t", "lat", "lon")
        with nc.Dataset(intp_file, "w") as ds:
            ds.createDimension("t")
            ds.createDimension("lat", shape.y_size)
            ds.createDimension("lon", shape.x_size)
            for varname in ["frp_avg_hr", "FRE"]:
                var = ds.createVariable(varname, "f4", dims)
                var[0, ...] = np.ones(shape.as_tuple)


def create_grid_out(root_dir: Path, shape: FakeGridOutShape) -> None:
    with nc.Dataset(root_dir / "ds_out_base.nc", "w") as ds:
        ds.createDimension("grid_yt", shape.y_size)
        ds.createDimension("grid_xt", shape.x_size)
        for varname in ["area", "grid_latt", "grid_lont"]:
            var = ds.createVariable(varname, "f4", ("grid_yt", "grid_xt"))
            var[:] = np.ones((shape.y_size, shape.x_size))


def create_veg_map(root_dir: Path, shape: FakeGridOutShape) -> None:
    with nc.Dataset(root_dir / "veg_map.nc", "w") as ds:
        ds.createDimension("grid_yt", shape.y_size)
        ds.createDimension("grid_xt", shape.x_size)
        emiss_factor = ds.createVariable("emiss_factor", "f4", ("grid_yt", "grid_xt"))
        emiss_factor[:] = np.ones((shape.y_size, shape.x_size))


def create_context(root_dir: Path, overrides: dict | None = None) -> SmokeDustContext:
    current_day = "2019072200"
    nwges_dir = root_dir
    os.environ["CDATE"] = current_day
    os.environ["DATA"] = str(nwges_dir)
    kwds = dict(
        staticdir=root_dir,
        ravedir=root_dir,
        intp_dir=root_dir,
        predef_grid="RRFS_CONUS_3km",
        ebb_dcycle_flag="2",
        restart_interval="6 12 18 24",
        persistence="FALSE",
        rave_qa_filter="NONE",
        exit_on_error="TRUE",
        log_level="DEBUG",
    )
    if overrides is not None:
        kwds.update(overrides)
    context = SmokeDustContext.create_from_args(kwds.values())
    return context


class ExpectedData(BaseModel):
    flag: str
    klass: Type[AbstractSmokeDustCycleProcessor]
    hash: str


class DataForTest(BaseModel):
    model_config = dict(arbitrary_types_allowed=True)
    context: SmokeDustContext
    preprocessor: SmokeDustPreprocessor
    expected: ExpectedData


@pytest.fixture(
    params=[
        ExpectedData(
            flag="1", klass=SmokeDustCycleOne, hash="d124734dfce7ca914391e35a02e4a7d2"
        ),
        ExpectedData(
            flag="2", klass=SmokeDustCycleTwo, hash="6752199f1039edc936a942f3885af38b"
        ),
    ]
)
def data_for_test(
    request: SubRequest, tmp_path: Path, fake_grid_out_shape: FakeGridOutShape
) -> DataForTest:
    try:
        create_grid_out(tmp_path, fake_grid_out_shape)
        create_veg_map(tmp_path, fake_grid_out_shape)
        context = create_context(
            tmp_path, overrides=dict(ebb_dcycle_flag=request.param.flag)
        )
        preprocessor = SmokeDustPreprocessor(context)
        create_restart_files(tmp_path, preprocessor.forecast_dates, fake_grid_out_shape)
        create_rave_interpolated(
            tmp_path,
            preprocessor.forecast_dates,
            fake_grid_out_shape,
            context.predef_grid.value + "_intp_",
        )
        return DataForTest(
            context=context, preprocessor=preprocessor, expected=request.param
        )
    finally:
        for ii in ["CDATE", "DATA"]:
            os.unsetenv(ii)


def create_file_hash(path: Path) -> str:
    with open(path, "rb") as f:
        file_hash = hashlib.md5()
        while chunk := f.read(8192):
            file_hash.update(chunk)
    return file_hash.hexdigest()


class TestSmokeDustPreprocessor:

    def test_run(self, data_for_test: DataForTest, mocker: MockerFixture) -> None:
        """Test core capabilities of the preprocessor. Note this does not test regridding."""
        preprocessor = data_for_test.preprocessor
        spy1 = mocker.spy(preprocessor, "create_dummy_emissions_file")
        regrid_processor_class = preprocessor._regrid_processor.__class__
        spy2 = mocker.spy(regrid_processor_class, "_run_impl_")
        spy3 = mocker.spy(regrid_processor_class, "run")
        cycle_processor_class = preprocessor._cycle_processor.__class__
        spy4 = mocker.spy(cycle_processor_class, "process_emissions")
        spy5 = mocker.spy(cycle_processor_class, "average_frp")

        assert isinstance(preprocessor._cycle_processor, data_for_test.expected.klass)
        assert preprocessor._forecast_metadata is None
        assert not data_for_test.context.emissions_path.exists()

        preprocessor.run()
        spy1.assert_not_called()
        spy2.assert_not_called()
        spy3.assert_called_once()
        spy4.assert_called_once()
        spy5.assert_called_once()

        assert data_for_test.context.emissions_path.exists()
        assert (
            create_file_hash(data_for_test.context.emissions_path)
            == data_for_test.expected.hash
        )
