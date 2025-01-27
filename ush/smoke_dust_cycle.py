import abc
import datetime as dt
from enum import StrEnum, unique
from typing import Dict, Any

import numpy as np
import pandas as pd
import xarray as xr

from smoke_dust_common import (
    open_nc,
    create_sd_variable,
    create_template_emissions_file,
    create_descriptive_statistics,
)
from smoke_dust_context import SmokeDustContext, EmissionVariable, EbbDCycle


@unique
class FrpVariable(StrEnum):
    FRP_AVG = "frp_avg_hr"
    EBB_TOTAL = "ebb_smoke_hr"


class AbstractSmokeDustCycleProcessor(abc.ABC):

    def __init__(self, context: SmokeDustContext):
        self._context = context

    def log(self, *args: Any, **kwargs: Any) -> None:
        self._context.log(*args, **kwargs)

    def create_derived_statistics(self, forecast_metadata: pd.DataFrame) -> None:
        with open_nc(self._context.emissions_path, "r", parallel=False) as ds:
            df = create_descriptive_statistics(
                {ii.value: ds.variables[ii.value][:] for ii in FrpVariable},
                "derived",
                self._context.emissions_path,
            )
        df = df.transpose()
        df.index.name = "variable"
        df.reset_index(inplace=True)
        forecast_dates = forecast_metadata["forecast_date"]
        stats_path = (
            self._context.intp_dir
            / f"stats_derived_{forecast_dates.min()}_{forecast_dates.max()}.csv"
        )
        self.log(f"writing {stats_path=}")
        df.to_csv(stats_path, index=False)

    @abc.abstractmethod
    def flag(self) -> EbbDCycle: ...

    @abc.abstractmethod
    def create_start_datetime(self) -> dt.datetime: ...

    @abc.abstractmethod
    def average_frp(
        self, forecast_metadata: pd.DataFrame
    ) -> Dict[FrpVariable, np.ndarray]: ...

    @abc.abstractmethod
    def process_emissions(self, forecast_metadata: pd.DataFrame) -> None: ...


class SmokeDustCycleOne(AbstractSmokeDustCycleProcessor):
    flag = EbbDCycle.ONE

    def create_start_datetime(self) -> dt.datetime:
        if self._context.persistence:
            self.log(
                "Creating emissions for persistence method where satellite FRP persist from previous day"
            )
            start_datetime = self._context.fcst_datetime - dt.timedelta(days=1)
        else:
            self.log("Creating emissions using current date satellite FRP")
            start_datetime = self._context.fcst_datetime
        return start_datetime

    def process_emissions(self, forecast_metadata: pd.DataFrame) -> None:
        derived = self.average_frp(forecast_metadata)
        self.log(f"creating 24-hour emissions file: {self._context.emissions_path}")
        with open_nc(
            self._context.emissions_path, "w", parallel=False, clobber=True
        ) as ds_out:
            create_template_emissions_file(ds_out, self._context.grid_out_shape)
            with open_nc(self._context.grid_out, parallel=False) as ds_src:
                ds_out.variables["geolat"][:] = ds_src.variables["grid_latt"][:]
                ds_out.variables["geolon"][:] = ds_src.variables["grid_lont"][:]
            create_sd_variable(
                ds_out,
                FrpVariable.FRP_AVG.value,
                "mean Fire Radiative Power",
                "MW",
                "0.f",
                0.0,
            )
            ds_out.variables[FrpVariable.FRP_AVG.value][:] = derived[
                FrpVariable.FRP_AVG
            ]
            create_sd_variable(
                ds_out,
                FrpVariable.EBB_TOTAL.value,
                "EBB emissions",
                "ug m-2 s-1",
                "0.f",
                0.0,
            )
            ds_out.variables[FrpVariable.EBB_TOTAL.value][:] = derived[
                FrpVariable.EBB_TOTAL
            ]

    def average_frp(
        self, forecast_metadata: pd.DataFrame
    ) -> Dict[FrpVariable, np.ndarray]:
        # tdk:story: refactor to share code with other cycle
        ebb_smoke_total = []
        frp_avg_hr = []

        with xr.open_dataset(self._context.veg_map) as ds:
            emiss_factor = ds["emiss_factor"].values
        with xr.open_dataset(self._context.grid_out) as ds:
            target_area = ds["area"].values

        for row_idx, row_df in forecast_metadata.iterrows():
            self.log(f"processing emissions: {row_idx}, {row_df.to_dict()}")
            with xr.open_dataset(row_df["rave_interpolated"]) as ds:
                fre = ds[EmissionVariable.FRE.smoke_dust_name()][0, :, :].values
                frp = ds[EmissionVariable.FRP.smoke_dust_name()][0, :, :].values

            frp_avg_hr.append(frp)
            ebb_hourly = (
                fre * emiss_factor * self._context.beta * self._context.fg_to_ug
            ) / (target_area * self._context.to_s)
            ebb_smoke_total.append(np.where(frp > 0, ebb_hourly, 0))

        frp_avg_reshaped = np.stack(frp_avg_hr, axis=0)
        ebb_total_reshaped = np.stack(ebb_smoke_total, axis=0)

        np.nan_to_num(frp_avg_reshaped, copy=False, nan=0.0)

        return {
            FrpVariable.FRP_AVG: frp_avg_reshaped,
            FrpVariable.EBB_TOTAL: ebb_total_reshaped,
        }


class SmokeDustCycleTwo(AbstractSmokeDustCycleProcessor):
    flag = EbbDCycle.TWO

    def create_start_datetime(self) -> dt.datetime:
        self.log("Creating emissions for modulated persistence by Wildfire potential")
        return self._context.fcst_datetime - dt.timedelta(days=1, hours=1)

    def process_emissions(self, forecast_metadata: pd.DataFrame) -> None:
        #tdk:story: figure out restart file copying
        self.log("process_emissions: enter")

        hwp_ave = []
        totprcp = np.zeros(self._context.grid_out_shape).ravel()
        for date in forecast_metadata['forecast_date']:
            phy_data_path = self._context.hourly_hwpdir / f"{date[:8]}.{date[8:10]}0000.phy_data.nc"
            rave_path = self._context.intp_dir / f"{self._context.rave_to_intp}{date}00_{date}59.nc"
            self.log(f"processing emissions for: {phy_data_path=}, {rave_path=}")
            with xr.open_dataset(phy_data_path) as ds:
                hwp_values = ds.rrfs_hwp_ave.values.ravel()
                tprcp_values = ds.totprcp_ave.values.ravel()
                totprcp += np.where(tprcp_values > 0, tprcp_values, 0)
                hwp_ave.append(hwp_values)
        hwp_ave_arr = np.nanmean(hwp_ave, axis=0).reshape(*self._context.grid_out_shape)
        totprcp_ave_arr = totprcp.reshape(*self._context.grid_out_shape)
        xarr_hwp = xr.DataArray(hwp_ave_arr)
        xarr_totprcp = xr.DataArray(totprcp_ave_arr)

        derived = self.average_frp(forecast_metadata)

        t_fire = np.zeros(self._context.grid_out_shape)
        for date in forecast_metadata['forecast_date']:
            rave_path = self._context.intp_dir / f"{self._context.rave_to_intp}{date}00_{date}59.nc"
            with xr.open_dataset(rave_path) as ds:
                frp = ds.frp_avg_hr[0, :, :].values
            dates_filtered = np.where(frp > 0, int(date[:10]), 0)
            t_fire = np.maximum(t_fire, dates_filtered)
        t_fire_flattened = [int(i) if i != 0 else 0 for i in t_fire.flatten()]
        hr_ends = [
            dt.datetime.strptime(str(hr), "%Y%m%d%H") if hr != 0 else 0
            for hr in t_fire_flattened
        ]
        te = np.array(
            [(self._context.fcst_datetime - i).total_seconds() / 3600 if i != 0 else 0 for i in hr_ends]
        )
        fire_age = np.array(te).reshape(self._context.grid_out_shape)

        # Ensure arrays are not negative or NaN
        frp_avg_reshaped = np.clip(derived[FrpVariable.FRP_AVG], 0, None)
        frp_avg_reshaped = np.nan_to_num(frp_avg_reshaped)

        ebb_tot_reshaped = np.clip(derived[FrpVariable.EBB_TOTAL], 0, None)
        ebb_tot_reshaped = np.nan_to_num(ebb_tot_reshaped)

        fire_age = np.clip(fire_age, 0, None)
        fire_age = np.nan_to_num(fire_age)

        # Filter HWP Prcp arrays to be non-negative and replace NaNs
        filtered_hwp = xarr_hwp.where(frp_avg_reshaped > 0, 0).fillna(0)
        filtered_prcp = xarr_totprcp.where(frp_avg_reshaped > 0, 0).fillna(0)

        # Filter based on ebb_rate
        ebb_rate_threshold = 0  # Define an appropriate threshold if needed
        mask = ebb_tot_reshaped > ebb_rate_threshold

        filtered_hwp = filtered_hwp.where(mask, 0).fillna(0)
        filtered_prcp = filtered_prcp.where(mask, 0).fillna(0)
        frp_avg_reshaped = frp_avg_reshaped * mask
        ebb_tot_reshaped = ebb_tot_reshaped * mask
        fire_age = fire_age * mask

        self.log(f"creating emissions file: {self._context.emissions_path}")
        with open_nc(self._context.emissions_path, "w", parallel=False) as ds_out:
            create_template_emissions_file(ds_out, self._context.grid_out_shape)
            with open_nc(self._context.grid_out, parallel=False) as ds_src:
                ds_out.variables["geolat"][:] = ds_src.variables["grid_latt"][:]
                ds_out.variables["geolon"][:] = ds_src.variables["grid_lont"][:]

            create_sd_variable(
                ds_out, "frp_davg", "Daily mean Fire Radiative Power", "MW", "0.f", 0.0
            )
            ds_out.variables["frp_davg"][0, :, :] = frp_avg_reshaped
            create_sd_variable(
                ds_out, "ebb_rate", "Total EBB emission", "ug m-2 s-1", "0.f", 0.0
            )
            ds_out.variables["ebb_rate"][0, :, :] = ebb_tot_reshaped
            create_sd_variable(
                ds_out, "fire_end_hr", "Hours since fire was last detected", "hrs", "0.f", 0.0
            )
            ds_out.variables["fire_end_hr"][0, :, :] = fire_age
            create_sd_variable(
                ds_out, "hwp_davg", "Daily mean Hourly Wildfire Potential", "none", "0.f", 0.0
            )
            ds_out.variables["hwp_davg"][0, :, :] = filtered_hwp
            create_sd_variable(
                ds_out, "totprcp_24hrs", "Sum of precipitation", "m", "0.f", 0.0
            )
            ds_out.variables["totprcp_24hrs"][0, :, :] = filtered_prcp

        self.log("process_emissions: exit")

    def average_frp(
        self, forecast_metadata: pd.DataFrame
    ) -> Dict[FrpVariable, np.ndarray]:
        self.log(f"average_frp: entering")

        frp_daily = np.zeros(self._context.grid_out_shape).ravel()
        ebb_smoke_total = []

        with xr.open_dataset(self._context.veg_map) as ds:
            emiss_factor = ds["emiss_factor"].values
        with xr.open_dataset(self._context.grid_out) as ds:
            target_area = ds["area"].values

        for row_idx, row_df in forecast_metadata.iterrows():
            self.log(f"processing emissions: {row_idx}, {row_df.to_dict()}")
            with xr.open_dataset(row_df["rave_interpolated"]) as ds:
                fre = ds[EmissionVariable.FRE.smoke_dust_name()][0, :, :].values
                frp = ds[EmissionVariable.FRP.smoke_dust_name()][0, :, :].values

            ebb_hourly = (
                fre
                * emiss_factor
                * self._context.beta
                * self._context.fg_to_ug
                / target_area
            )
            ebb_smoke_total.append(np.where(frp > 0, ebb_hourly, 0).ravel())
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

        self.log("average_frp: exiting")
        return {
            FrpVariable.FRP_AVG: frp_avg_reshaped,
            FrpVariable.EBB_TOTAL: ebb_total_reshaped,
        }


def create_cycle_processor(
    context: SmokeDustContext,
) -> AbstractSmokeDustCycleProcessor:
    match context.ebb_dcycle_flag:
        case EbbDCycle.ONE:
            return SmokeDustCycleOne(context)
        case EbbDCycle.TWO:
            return SmokeDustCycleTwo(context)
        case _:
            raise NotImplementedError(context.ebb_dcycle_flag)
