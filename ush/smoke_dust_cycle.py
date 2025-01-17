import abc
import datetime as dt
from enum import StrEnum, unique
from typing import Dict, Any

import pandas as pd

from smoke_dust_context import SmokeDustContext, EmissionVariable, EbbDCycle
import numpy as np
import xarray as xr

from smoke_dust_interpolation import open_nc, create_sd_variable, create_template_emissions_file
from smoke_dust_interpolation import create_descriptive_statistics


@unique
class DerivedVariable(StrEnum):
    FRP_AVG = "frp_avg_hr"
    EBB_TOTAL = "ebb_smoke_hr"


class AbstractSmokeDustCycleProcessor(abc.ABC):

    def __init__(self, context: SmokeDustContext):
        self._context = context

    def log(self, *args: Any, **kwargs: Any) -> None:
        self._context.log(*args, **kwargs)

    def create_derived_statistics(self) -> None:
        with open_nc(self._context.emissions_path, 'r', parallel=False) as ds:
            df = create_descriptive_statistics({ii.value: ds.variables[ii.value][:] for ii in DerivedVariable}, "derived", self._context.emissions_path)
        derived_stats_out = self._context.intp_dir / "derived_variable_statistics.csv" #tdk: add forecast date info to filenames
        self.log(f"writing {derived_stats_out}")
        df = df.transpose()
        df.index.name = "variable"
        df.reset_index(inplace=True)
        df.to_csv(derived_stats_out, index=False)

    @abc.abstractmethod
    def flag(self) -> EbbDCycle:
        ...

    @abc.abstractmethod
    def create_start_datetime(self) -> dt.datetime:
        ...

    @abc.abstractmethod
    def average_frp(self, forecast_metadata: pd.DataFrame) -> Dict[DerivedVariable, np.ndarray]:
        ...

    @abc.abstractmethod
    def process_emissions(self, forecast_metadata: pd.DataFrame) -> None:
        ...


class SmokeDustCycleOne(AbstractSmokeDustCycleProcessor):
    flag = EbbDCycle.ONE

    def create_start_datetime(self) -> dt.datetime:
        if self._context.persistence:
            self.log("Creating emissions for persistence method where satellite FRP persist from previous day")
            start_datetime = self._context.fcst_datetime - dt.timedelta(days=1)
        else:
            self.log("Creating emissions using current date satellite FRP")
            start_datetime = self._context.fcst_datetime
        return start_datetime

    def process_emissions(self, forecast_metadata: pd.DataFrame) -> None:
        derived = self.average_frp(forecast_metadata)
        self.log(f"creating 24-hour emissions file: {self._context.emissions_path}")
        with open_nc(self._context.emissions_path, "w", parallel=False, clobber=True) as ds_out:
            create_template_emissions_file(ds_out, self._context.grid_out_shape)
            with open_nc(self._context.grid_out, parallel=False) as ds_src:
                ds_out.variables["geolat"][:] = ds_src.variables["grid_latt"][:]
                ds_out.variables["geolon"][:] = ds_src.variables["grid_lont"][:]
            create_sd_variable(
                ds_out, DerivedVariable.FRP_AVG.value, "mean Fire Radiative Power", "MW", "0.f", 0.
            )
            ds_out.variables[DerivedVariable.FRP_AVG.value][:] = derived[DerivedVariable.FRP_AVG]
            create_sd_variable(
                ds_out, DerivedVariable.EBB_TOTAL.value, "EBB emissions", "ug m-2 s-1", "0.f", 0.
            )
            ds_out.variables[DerivedVariable.EBB_TOTAL.value][:] = derived[DerivedVariable.EBB_TOTAL]

    def average_frp(self, forecast_metadata: pd.DataFrame) -> Dict[DerivedVariable, np.ndarray]:
        #tdk:story: refactor to share code with other cycle
        ebb_smoke_total = []
        frp_avg_hr = []

        with xr.open_dataset(self._context.veg_map) as ds:
            emiss_factor = ds['emiss_factor'].values
        with xr.open_dataset(self._context.grid_out) as ds:
            target_area = ds['area'].values

        for row_idx, row_df in forecast_metadata.iterrows():
            self.log(f"processing emissions: {row_idx}, {row_df.to_dict()}")
            with xr.open_dataset(row_df['rave_interpolated']) as ds:
                fre = ds[EmissionVariable.FRE.smoke_dust_name()][0, :, :].values
                frp = ds[EmissionVariable.FRP.smoke_dust_name()][0, :, :].values

            frp_avg_hr.append(frp)
            ebb_hourly = (fre * emiss_factor * self._context.beta * self._context.fg_to_ug) / (
                    target_area * self._context.to_s
            )
            ebb_smoke_total.append(
                np.where(frp > 0, ebb_hourly, 0)
            )

        frp_avg_reshaped = np.stack(frp_avg_hr, axis=0)
        ebb_total_reshaped = np.stack(ebb_smoke_total, axis=0)

        np.nan_to_num(frp_avg_reshaped, copy=False, nan=0.0)

        return {DerivedVariable.FRP_AVG:frp_avg_reshaped, DerivedVariable.EBB_TOTAL:ebb_total_reshaped}

class SmokeDustCycleTwo(AbstractSmokeDustCycleProcessor):
    flag = EbbDCycle.TWO

    def create_start_datetime(self) -> dt.datetime:
        self.log("Creating emissions for modulated persistence by Wildfire potential")
        return self._context.fcst_datetime - dt.timedelta(days=1, hours=1)

    def process_emissions(self, forecast_metadata: pd.DataFrame) -> None:
        #tdk:story: implement emissions processing when we can test
        raise NotImplementedError

    def average_frp(self, forecast_metadata: pd.DataFrame) -> Dict[DerivedVariable, np.ndarray]:
        frp_daily = np.zeros(self._context.grid_out_shape)
        ebb_smoke_total = []

        with xr.open_dataset(self._context.veg_map) as ds:
            emiss_factor = ds['emiss_factor'].values
        with xr.open_dataset(self._context.grid_out) as ds:
            target_area = ds['area'].values

        for row_idx, row_df in forecast_metadata.iterrows():
            self.log(f"processing emissions: {row_idx}, {row_df.to_dict()}")
            with xr.open_dataset(row_df['rave_interpolated']) as ds:
                fre = ds[EmissionVariable.FRE.smoke_dust_name()][0, :, :].values
                frp = ds[EmissionVariable.FRP.smoke_dust_name()][0, :, :].values

            ebb_hourly = (
                    fre * emiss_factor * self._context.beta * self._context.fg_to_ug / target_area
            )
            ebb_smoke_total.append(
                np.where(frp > 0, ebb_hourly, 0).ravel()
            )
            frp_daily += np.where(frp > 0, frp, 0).ravel()

        summed_array = np.sum(np.array(ebb_smoke_total), axis=0)
        num_zeros = len(ebb_smoke_total) - np.sum(
            [arr == 0 for arr in ebb_smoke_total], axis=0
        )
        safe_zero_count = np.where(num_zeros == 0, 1, num_zeros)
        result_array = np.array(
            [
                (
                    summed_array[i] / 2
                    if safe_zero_count[i] == 1
                    else summed_array[i] / safe_zero_count[i]
                )
                for i in range(len(safe_zero_count))
            ]
        )
        result_array[num_zeros == 0] = summed_array[num_zeros == 0]
        ebb_total = result_array.reshape(self._context.grid_out_shape)
        ebb_total_reshaped = ebb_total / 3600
        temp_frp = np.array(
            [
                (
                    frp_daily[i] / 2
                    if safe_zero_count[i] == 1
                    else frp_daily[i] / safe_zero_count[i]
                )
                for i in range(len(safe_zero_count))
            ]
        )
        temp_frp[num_zeros == 0] = frp_daily[num_zeros == 0]
        frp_avg_reshaped = temp_frp.reshape(*self._context.grid_out_shape)

        np.nan_to_num(frp_avg_reshaped, copy=False, nan=0.0)

        return {DerivedVariable.FRP_AVG: frp_avg_reshaped, DerivedVariable.EBB_TOTAL: ebb_total_reshaped}