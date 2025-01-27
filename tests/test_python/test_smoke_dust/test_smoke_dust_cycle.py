import sys
from pathlib import Path

sys.path.append(str(Path("../../../ush")))

import pytest
import xarray as xr
import netCDF4 as nc

from smoke_dust_context import SmokeDustContext
from smoke_dust_cycle import SmokeDustCycleTwo


class TestSmokeDustCycleTwo:

    def test_process_emissions(self, tmp_path: Path) -> None:

        with nc.Dataset(tmp_path / 'ds_out_base.nc', 'w') as ds:
            ds.createDimension('grid_yt', 5)
            ds.createDimension('grid_xt', 10)

        context = SmokeDustContext(staticdir=tmp_path,
                                   ravedir=tmp_path,
                                   intp_dir=tmp_path,
                                   predef_grid='RRFS_CONUS_3km',
                                   ebb_dcycle_flag='2',
                                   restart_interval='',
                                   persistence='FALSE',
                                   rave_qa_filter='NONE',
                                   exit_on_error='TRUE',
                                   log_level='DEBUG',
                                   current_day='2019072200',
                                   nwges_dir=tmp_path)

        cycle = SmokeDustCycleTwo(context)