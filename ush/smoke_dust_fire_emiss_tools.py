#!/usr/bin/env python3

import os
from typing import Tuple, Any

import numpy as np
import xarray as xr
from datetime import datetime
from netCDF4 import Dataset
from pandas import Index
from xarray import DataArray

def estimate_fire_duration(
    intp_dir: str,
    fcst_dates: Index,
    current_day: str,
    cols: int,
    rows: int,
    rave_to_intp: str,
) -> np.ndarray:
    """
    Estimate fire duration potentially using data from previous cycles.

    There are two steps here.
        1) First day simulation no RAVE from previous 24 hours available (fire age is set to zero).
        2) Previous files are present (estimate fire age as the difference between the date of the current cycle and the date whe the fire was last observed within 24 hours).

    Args:
        intp_dir: Path to interpolated RAVE data
        fcst_dates: Forecast hours used in the current cycle
        current_day: The current day hour
        cols: Number of columns
        rows: Number of rows
        rave_to_intp: Prefix of the target RAVE files
    """
    t_fire = np.zeros((cols, rows))

    for date_str in fcst_dates:
        try:
            assert isinstance(date_str, str)
            date_file = int(date_str[:10])
            print("Date processing for fire duration", date_file)
            file_path = os.path.join(
                intp_dir, f"{rave_to_intp}{date_str}00_{date_str}59.nc"
            )

            if os.path.exists(file_path):
                try:
                    with xr.open_dataset(file_path) as open_intp:
                        FRP = open_intp.frp_avg_hr[0, :, :].values
                        dates_filtered = np.where(FRP > 0, date_file, 0)
                        t_fire = np.maximum(t_fire, dates_filtered)
                except (
                    FileNotFoundError,
                    IOError,
                    OSError,
                    RuntimeError,
                    ValueError,
                    TypeError,
                    KeyError,
                    IndexError,
                    MemoryError,
                ) as e:
                    print(f"Error processing NetCDF file {file_path}: {e}")
        except Exception as e:
            print(f"Error processing date {date_str}: {e}")

    t_fire_flattened = [int(i) if i != 0 else 0 for i in t_fire.flatten()]

    try:
        fcst_t = datetime.strptime(current_day, "%Y%m%d%H")
        hr_ends = [
            datetime.strptime(str(hr), "%Y%m%d%H") if hr != 0 else 0
            for hr in t_fire_flattened
        ]
        te = np.array(
            [(fcst_t - i).total_seconds() / 3600 if i != 0 else 0 for i in hr_ends]
        )
    except ValueError as e:
        print(f"Error processing forecast time {current_day}: {e}")
        te = np.zeros((rows, cols))

    return te


def save_fire_dur(cols: int, rows: int, te: np.ndarray) -> np.ndarray:
    """
    Reshape the fire duration array.

    Args:
        cols: Number of columns
        rows: Number of rows
        te: Target array to reshape

    Returns:
        The reshaped fire duration array
    """
    fire_dur = np.array(te).reshape(cols, rows)
    return fire_dur


def produce_emiss_file(
    xarr_hwp: DataArray,
    frp_avg_reshaped: np.ndarray,
    totprcp_ave_arr: Any,
    xarr_totprcp: DataArray,
    intp_dir: str,
    current_day: str,
    tgt_latt: DataArray,
    tgt_lont: DataArray,
    ebb_tot_reshaped: np.ndarray,
    fire_age: np.ndarray,
    cols: int,
    rows: int,
) -> str:
    """
    Produce the emissions file.

    Args:
        xarr_hwp: Data array containing HWP
        frp_avg_reshaped: Average FRP array
        totprcp_ave_arr: Average total precipitation array
        xarr_totprcp: Average total precipitation as a data array
        intp_dir: Directory containing interpolated RAVE data
        current_day: The current forecast day/hour
        tgt_latt: The target grid latitude
        tgt_lont: The target grid longitudes
        ebb_tot_reshaped: Total EBB array
        fire_age: Estimated fire age array
        cols: Number of columns
        rows: Number of rows

    Returns:
        A string indicating the file was written as expected
    """
    # Ensure arrays are not negative or NaN
    frp_avg_reshaped = np.clip(frp_avg_reshaped, 0, None)
    frp_avg_reshaped = np.nan_to_num(frp_avg_reshaped)

    ebb_tot_reshaped = np.clip(ebb_tot_reshaped, 0, None)
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

    # Produce emiss file
    file_path = os.path.join(intp_dir, f"SMOKE_RRFS_data_{current_day}00.nc")

    try:
        with Dataset(file_path, "w") as fout:
            i_tools.create_emiss_file(fout, cols, rows)
            i_tools.Store_latlon_by_Level(
                fout,
                "geolat",
                tgt_latt,
                "cell center latitude",
                "degrees_north",
                "-9999.f",
            )
            i_tools.Store_latlon_by_Level(
                fout,
                "geolon",
                tgt_lont,
                "cell center longitude",
                "degrees_east",
                "-9999.f",
            )

            print("Storing different variables")
            i_tools.Store_by_Level(
                fout, "frp_davg", "Daily mean Fire Radiative Power", "MW", "0.f"
            )
            fout.variables["frp_davg"][0, :, :] = frp_avg_reshaped
            i_tools.Store_by_Level(
                fout, "ebb_rate", "Total EBB emission", "ug m-2 s-1", "0.f"
            )
            fout.variables["ebb_rate"][0, :, :] = ebb_tot_reshaped
            i_tools.Store_by_Level(
                fout, "fire_end_hr", "Hours since fire was last detected", "hrs", "0.f"
            )
            fout.variables["fire_end_hr"][0, :, :] = fire_age
            i_tools.Store_by_Level(
                fout, "hwp_davg", "Daily mean Hourly Wildfire Potential", "none", "0.f"
            )
            fout.variables["hwp_davg"][0, :, :] = filtered_hwp
            i_tools.Store_by_Level(
                fout, "totprcp_24hrs", "Sum of precipitation", "m", "0.f"
            )
            fout.variables["totprcp_24hrs"][0, :, :] = filtered_prcp

        print("Emissions file created successfully")
        return "Emissions file created successfully"

    except (OSError, IOError) as e:
        print(f"Error creating or writing to NetCDF file {file_path}: {e}")
        return f"Error creating or writing to NetCDF file {file_path}: {e}"

    return "Emissions file created successfully"
