from pathlib import Path

import pytest
from _pytest.fixtures import SubRequest
import netCDF4 as nc

from smoke_dust.core.common import open_nc
from smoke_dust.core.context import SmokeDustContext
from smoke_dust.core.cycle import SmokeDustCycleTwo
from smoke_dust.core.preprocessor import SmokeDustPreprocessor
from test_python.test_smoke_dust.conftest import create_fake_context, create_fake_grid_out, \
    FakeGridOutShape


@pytest.fixture(params=[True, False], ids= lambda p: f"allow_dummy_restart={p}")
def context_for_dummy_test(request: SubRequest, tmp_path: Path, fake_grid_out_shape: FakeGridOutShape) -> SmokeDustContext:
    create_fake_grid_out(tmp_path, fake_grid_out_shape)
    context = create_fake_context(tmp_path, overrides={"allow_dummy_restart": request.param})
    return context


def create_restart_ncfile(path: Path, varnames: list[str]) -> None:
    with open_nc(path, mode="w") as nc_ds:
        dim = nc_ds.createDimension("foo")
        for varname in varnames:
            nc_ds.createVariable(varname, "f4", (dim.name,))


class TestSmokeDustCycleTwo:

    def test_writes_dummy_emissions_with_no_restart_files(
            self, context_for_dummy_test: SmokeDustContext) -> None:
        cycle = SmokeDustCycleTwo(context_for_dummy_test)
        assert not context_for_dummy_test.emissions_path.exists()
        try:
            cycle.run()
        except FileNotFoundError:
            assert not context_for_dummy_test.allow_dummy_restart
        else:
            assert context_for_dummy_test.emissions_path.exists()

    def test_iter_restart_files(self, tmp_path: Path, fake_grid_out_shape: FakeGridOutShape) -> None:
        create_fake_grid_out(tmp_path, fake_grid_out_shape)
        context = create_fake_context(tmp_path)
        cycle = SmokeDustCycleTwo(context)
        expected_vars = ("totprcp_ave", "rrfs_hwp_ave")
        restart_slug = "phy_data"
        outdir = tmp_path / "RESTART"
        outdir.mkdir()
        create_restart_ncfile(outdir / f"foobar.nonsense.{restart_slug}.0000.nc", expected_vars)
        create_restart_ncfile(outdir / f"foobar.nonsense.{restart_slug}.1111.nc", [])
        create_restart_ncfile(outdir / "foobar.nonsense.nc", [])
        for root_dir in [outdir, tmp_path]:
            print(root_dir)
            restart_files = list(cycle._iter_restart_files_(root_dir, expected_vars, ))
            assert len(restart_files) == 1
