import os
import sys
from pathlib import Path

sys.path.append(str(Path("../../../ush")))

import netCDF4 as nc
import pandas as pd
import numpy as np

from smoke_dust_context import SmokeDustContext
from smoke_dust_cycle import SmokeDustCycleTwo
from smoke_dust_main import SmokeDustPreprocessor


def create_restart_files(root_dir: Path, forecast_metadata: pd.DataFrame, grid_out_shape: tuple[int, int]) -> None:
    restart_dir = root_dir / 'RESTART'
    restart_dir.mkdir()
    for date in forecast_metadata['forecast_date']:
        restart_file = restart_dir / f'{date[:8]}.{date[8:10]}0000.phy_data.nc'
        with nc.Dataset(restart_file, 'w') as ds:
            ds.createDimension('Time')
            ds.createDimension('yaxis_1', grid_out_shape[0])
            ds.createDimension('xaxis_1', grid_out_shape[1])
            totprcp_ave = ds.createVariable('totprcp_ave', 'f4', ('Time', 'yaxis_1', 'xaxis_1'))
            totprcp_ave[0, ...] = np.ones(grid_out_shape)
            rrfs_hwp_ave = ds.createVariable('rrfs_hwp_ave', 'f4', ('Time', 'yaxis_1', 'xaxis_1'))
            rrfs_hwp_ave[0, ...] = totprcp_ave[:] + 2


def create_rave_interpolated(root_dir: Path, forecast_metadata: pd.DataFrame, grid_out_shape: tuple[int, int],
                             rave_to_intp: str) -> None:
    for date in forecast_metadata['forecast_date']:
        intp_file = root_dir / f"{rave_to_intp}{date}00_{date}59.nc"
        dims = ('t', 'lat', 'lon')
        with nc.Dataset(intp_file, 'w') as ds:
            ds.createDimension('t')
            ds.createDimension('lat', grid_out_shape[0])
            ds.createDimension('lon', grid_out_shape[1])
            for varname in ['frp_avg_hr', 'FRE']:
                var = ds.createVariable(varname, 'f4', dims)
                var[0, ...] = np.ones(grid_out_shape)


class TestSmokeDustCycleTwo:

    def test_process_emissions(self, tmp_path: Path) -> None:
        y_size = 5
        x_size = 10
        with nc.Dataset(tmp_path / 'ds_out_base.nc', 'w') as ds:
            ds.createDimension('grid_yt', y_size)
            ds.createDimension('grid_xt', x_size)
            for varname in ['area', 'grid_latt', 'grid_lont']:
                var = ds.createVariable(varname, 'f4', ('grid_yt', 'grid_xt'))
                var[:] = np.ones((y_size, x_size))
        with nc.Dataset(tmp_path / 'veg_map.nc', 'w') as ds:
            ds.createDimension('grid_yt', y_size)
            ds.createDimension('grid_xt', x_size)
            emiss_factor = ds.createVariable('emiss_factor', 'f4', ('grid_yt', 'grid_xt'))
            emiss_factor[:] = np.ones((y_size, x_size))

        current_day = '2019072200'
        nwges_dir = tmp_path
        os.environ['CDATE'] = current_day
        os.environ['DATA'] = str(nwges_dir)

        kwds = dict(staticdir=tmp_path,
                    ravedir=tmp_path,
                    intp_dir=tmp_path,
                    predef_grid='RRFS_CONUS_3km',
                    ebb_dcycle_flag='2',
                    restart_interval='6 12 18 24',
                    persistence='FALSE',
                    rave_qa_filter='NONE',
                    exit_on_error='TRUE',
                    log_level='DEBUG',
                    )
        context = SmokeDustContext.create_from_args(kwds.values())

        preprocessor = SmokeDustPreprocessor(context)
        cycle = SmokeDustCycleTwo(context)

        create_restart_files(tmp_path, preprocessor.forecast_metadata, context.grid_out_shape)
        create_rave_interpolated(tmp_path, preprocessor.forecast_metadata, context.grid_out_shape,
                                 context.predef_grid.value + "_intp_")
        preprocessor._forecast_metadata = None

        cycle.process_emissions(preprocessor.forecast_metadata)

        assert context.emissions_path.exists()
