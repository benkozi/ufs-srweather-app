"""Shared pytest fixtures and test functions."""

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import netCDF4 as nc
import numpy as np
import pytest

from smoke_dust.core.context import SmokeDustContext


@dataclass
class FakeGridOutShape:
    """Explicitly defines the test grid shape."""

    y_size: int = 5
    x_size: int = 10

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
    return (Path(__file__).parent / "bin").expanduser().resolve(strict=True)


def create_fake_grid_out(root_dir: Path, shape: FakeGridOutShape) -> None:
    """Create the output grid netCDF file. The output grid is the domain grid for the experiment.

    Args:
        root_dir: Directory to write grid to.
        shape: Grid output shape.
    """
    with nc.Dataset(root_dir / "ds_out_base.nc", "w") as ds:
        ds.createDimension("grid_yt", shape.y_size)
        ds.createDimension("grid_xt", shape.x_size)
        for varname in ["area", "grid_latt", "grid_lont"]:
            var = ds.createVariable(varname, "f4", ("grid_yt", "grid_xt"))
            var[:] = np.ones((shape.y_size, shape.x_size))


def create_fake_context(
    root_dir: Path, overrides: Union[dict, None] = None, extra: Union[dict, None] = None
) -> SmokeDustContext:
    """
    Create a fake context for the test runner.
    Args:
        root_dir: Path to write fake test files to.
        overrides: If provided, override the required context arguments - the arguments provided to
            the CLI program.
        extra: If provided, override context parameters not used in the CLI.

    Returns:
        A fake context to use for testing.
    """
    current_day = "2019072200"
    nwges_dir = root_dir
    os.environ["CDATE"] = current_day
    os.environ["DATA"] = str(nwges_dir)
    kwds = {
        "staticdir": root_dir,
        "ravedir": root_dir,
        "intp_dir": root_dir,
        "predef_grid": "RRFS_CONUS_3km",
        "ebb_dcycle_flag": "2",
        "restart_interval": "6 12 18 24",
        "persistence": "FALSE",
        "rave_qa_filter": "NONE",
        "exit_on_error": "TRUE",
        "log_level": "debug",
    }
    if overrides is not None:
        kwds.update(overrides)
    context = SmokeDustContext.create_from_args(kwds.values(), extra=extra)
    for ii in ["CDATE", "DATA"]:
        os.unsetenv(ii)
    return context


def create_file_hash(path: Path) -> str:
    """
    Create a unique file hash to use for bit-for-bit comparison.
    Args:
        path: Target binary file to hash.

    Returns:
        The file's hex digest.
    """
    with open(path, "rb") as f:
        file_hash = hashlib.md5()
        while chunk := f.read(8192):
            file_hash.update(chunk)
    return file_hash.hexdigest()
