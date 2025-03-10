from pathlib import Path
from subprocess import CalledProcessError

import pytest

from smoke_dust.core.common import open_nc, nccmp


def create_nc_file(path: Path) -> None:
    with open_nc(path, mode="w", parallel=False) as nc_ds:
        dim = nc_ds.createDimension("dim", 1)
        nc_ds.createVariable("var", "f4", (dim,))


def test_nccmp_happy_path(tmp_path: Path) -> None:
    nc_path = tmp_path / "nccmp_happy.nc"
    create_nc_file(nc_path)
    nccmp(nc_path, nc_path)


def test_nccmp_with_diff(tmp_path: Path) -> None:
    lhs = tmp_path / "lhs.nc"
    rhs = tmp_path / "rhs.nc"
    create_nc_file(lhs)
    create_nc_file(rhs)
    with open_nc(rhs, mode="a", parallel=False) as nc_ds:
        nc_ds.variables["var"][:] = 10.
    with pytest.raises(CalledProcessError):
        nccmp(lhs, rhs)