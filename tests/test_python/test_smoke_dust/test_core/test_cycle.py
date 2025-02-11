"""Tests related to the smoke/dust cycle processor."""
from pathlib import Path

import pytest
from _pytest.fixtures import SubRequest

from smoke_dust.core.common import open_nc
from smoke_dust.core.context import SmokeDustContext
from smoke_dust.core.cycle import SmokeDustCycleTwo
from test_python.test_smoke_dust.conftest import (
    create_fake_context,
    create_fake_grid_out,
    FakeGridOutShape, create_fake_restart_files,
)
from datetime import datetime, timedelta


@pytest.fixture(params=[True, False], ids=lambda p: f"allow_dummy_restart={p}")
def context_for_dummy_test(
    request: SubRequest, tmp_path: Path, fake_grid_out_shape: FakeGridOutShape
) -> SmokeDustContext:
    """Create a context for the dummy restart files test."""
    create_fake_grid_out(tmp_path, fake_grid_out_shape)
    context = create_fake_context(tmp_path, overrides={"allow_dummy_restart": request.param})
    return context


def create_restart_ncfile(path: Path, varnames: list[str]) -> None:
    """Create a physics-related restart netCDF file."""
    with open_nc(path, mode="w") as nc_ds:
        dim = nc_ds.createDimension("foo")
        for varname in varnames:
            nc_ds.createVariable(varname, "f4", (dim.name,))


class TestSmokeDustCycleTwo:
    """..."""

    def test_writes_dummy_emissions_with_no_restart_files(
        self, context_for_dummy_test: SmokeDustContext  # pylint: disable=redefined-outer-name
    ) -> None:
        """Test that dummy emissions are handled appropriately when no restart files are present."""
        cycle = SmokeDustCycleTwo(context_for_dummy_test)
        assert not context_for_dummy_test.emissions_path.exists()
        try:
            cycle.run()
        except FileNotFoundError:
            assert not context_for_dummy_test.allow_dummy_restart
        else:
            assert context_for_dummy_test.emissions_path.exists()

    def test_iter_restart_files(
        self, tmp_path: Path, fake_grid_out_shape: FakeGridOutShape
    ) -> None:
        """Test iterating over restart files."""
        create_fake_grid_out(tmp_path, fake_grid_out_shape)
        context = create_fake_context(tmp_path)
        assert context.hourly_hwpdir.name.endswith('RESTART')
        cycle = SmokeDustCycleTwo(context)
        assert cycle._root_restart_dir == context.hourly_hwpdir.parent.parent
        create_fake_restart_files(context.nwges_dir, cycle.cycle_dates, fake_grid_out_shape)
        create_fake_restart_files(context.nwges_dir, [str(datetime.strptime(date, "%Y%m%d%H") + timedelta(days=10)) for date in cycle.cycle_dates], fake_grid_out_shape)
        actual = list(cycle._iter_restart_files_())
        assert len(actual) == len(cycle.cycle_dates)
