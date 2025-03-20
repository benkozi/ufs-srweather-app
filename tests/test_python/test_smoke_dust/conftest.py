"""Shared pytest fixtures and test functions."""

import hashlib
import os
import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import pytest
from netCDF4 import Dataset

from smoke_dust.core.comm import COMM
from smoke_dust.core.context import SmokeDustContext


@dataclass
class FakeGridOutShape:
    """Explicitly defines the test grid shape."""

    y_size: int = 50
    x_size: int = 100

    @property
    def as_tuple(self) -> tuple[int, int]:
        """
        Convert the grid shape to a tuple.
        """
        return self.y_size, self.x_size


@pytest.fixture
def fake_grid_out_shape() -> FakeGridOutShape:
    """Fixture creating the test grid shape."""
    return FakeGridOutShape()


@pytest.fixture
def bin_dir() -> Path:
    """Fixture returning the path to the binary test directory for this package."""
    path = Path(__file__).parent / "bin"
    if COMM.rank == 0:
        if not path.exists():
            path.mkdir(exist_ok=False, parents=False)
            zip_name = "srw-smokedust-test-data-20250320.zip"
            current_dir = os.getcwd()
            os.chdir(path)
            try:
                subprocess.check_call(["wget",
                                       "-O", zip_name,
                                       f"https://www.dropbox.com/scl/fi/qxukhxtchuof161be4uoe/srw-smokedust-test-data-20250320.zip?rlkey=tmoiti07hq296hjvfyg8wxex4&st=ja2ktc23&dl=1"])
            finally:
                os.chdir(current_dir)
            try:
                with zipfile.ZipFile(str(path / zip_name), 'r') as zip_ref:
                    zip_ref.extractall(path)
            finally:
                os.remove(path/zip_name)
    COMM.barrier()
    return path.expanduser().resolve(strict=True)


def create_fake_grid_out(root_dir: Path, shape: FakeGridOutShape) -> None:
    """Create the output grid netCDF file. The output grid is the domain grid for the experiment.

    Args:
        root_dir: Directory to write grid to.
        shape: Grid output shape.
    """
    with Dataset(root_dir / "ds_out_base.nc", "w") as nc_ds:
        nc_ds.createDimension("grid_yt", shape.y_size)
        nc_ds.createDimension("grid_xt", shape.x_size)
        for varname in ["area", "grid_latt", "grid_lont"]:
            var = nc_ds.createVariable(varname, "f4", ("grid_yt", "grid_xt"))
            var[:] = create_analytic_array(shape)


def create_fake_context(root_dir: Path, overrides: Union[dict, None] = None) -> SmokeDustContext:
    """
    Create a fake context for the test runner.

    Args:
        root_dir: Path to write fake test files to.
        overrides: If provided, override the required context arguments - the arguments provided to
            the CLI program.

    Returns:
        A fake context to use for testing.
    """
    current_day = "2019072200"
    comin = root_dir / current_day
    if COMM.rank == 0:
        comin.mkdir(exist_ok=True)
    COMM.barrier()
    os.environ["CDATE"] = current_day
    os.environ["COMIN_SMOKE_DUST_COMMUNITY"] = str(comin)
    kwds = {
        "staticdir": root_dir,
        "ravedir": root_dir,
        "intp_dir": root_dir,
        # "intp_dir": "/opt/project/benkozi-data/baseline", #tdk: how to generate baselines?
        "predef_grid": "RRFS_CONUS_3km",
        "ebb_dcycle": "2",
        "restart_interval": "6 12 18 24",
        "persistence": "false",
        "rave_qa_filter": "none",
        "exit_on_error": "TRUE",
        "log_level": "debug",
    }
    if overrides is not None:
        kwds.update(overrides)
    try:
        context = SmokeDustContext.model_validate(kwds)
    except:
        for env_var in ["CDATE", "COMIN_SMOKE_DUST_COMMUNITY"]:
            os.unsetenv(env_var)
        raise
    return context


def create_file_hash(path: Path) -> str:
    """
    Create a unique file hash to use for bit-for-bit comparison.
    Args:
        path: Target binary file to hash.

    Returns:
        The file's hex digest.
    """
    with open(path, "rb") as target_file:
        file_hash = hashlib.md5()
        while chunk := target_file.read(8192):
            file_hash.update(chunk)
    return file_hash.hexdigest()


def create_fake_restart_files(
    root_dir: Path, cycle_dates: pd.DatetimeIndex, shape: FakeGridOutShape
) -> None:
    """
    Create fake restart files expected for EBB_DCYLE=2.

    Args:
        root_dir: Directory to create fake files in.
        cycle_dates: The series of dates to create the restart files for.
        shape: Output grid shape.
    """
    for date in cycle_dates:
        restart_dir = root_dir / date / "RESTART"
        restart_dir.mkdir(exist_ok=True, parents=True)
        restart_file = restart_dir / f"{date[:8]}.{date[8:10]}0000.phy_data.nc"
        with Dataset(restart_file, "w") as nc_ds:
            nc_ds.createDimension("Time")
            nc_ds.createDimension("yaxis_1", shape.y_size)
            nc_ds.createDimension("xaxis_1", shape.x_size)
            totprcp_ave = nc_ds.createVariable("totprcp_ave", "f4", ("Time", "yaxis_1", "xaxis_1"))
            totprcp_ave[0, ...] = np.ones(shape.as_tuple)
            rrfs_hwp_ave = nc_ds.createVariable(
                "rrfs_hwp_ave", "f4", ("Time", "yaxis_1", "xaxis_1")
            )
            rrfs_hwp_ave[0, ...] = totprcp_ave[:] + 2


def create_analytic_array(shape: FakeGridOutShape) -> np.ndarray:
    lat_vec = np.linspace(-80.0, 80.0, num=shape.y_size)
    lon_vec = np.linspace(-170.0, 170.0, num=shape.x_size)
    lat_mesh, lon_mesh = np.meshgrid(lon_vec, lat_vec)
    deg_to_rad = 3.141592653589793 / 180.0
    analytic_data = 2.0 + np.cos(deg_to_rad * lon_mesh) ** 2 * np.cos(
        2.0 * deg_to_rad * (90.0 - lat_mesh)
    )
    return analytic_data


@pytest.fixture
def tmp_path_shared(tmp_path: Path) -> Path:
    return Path(COMM.bcast({"path": str(tmp_path)}, root=0)["path"])


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "mpi: mark test to run under MPI runs"
    )