from typing import Tuple, Any

from pydantic import BaseModel, field_validator


class SmokeDustVariable(BaseModel):
    name: str
    long_name: str
    units: str
    fill_value_str: str
    fill_value_float: float


SmokeDustVariablesType = Tuple[SmokeDustVariable, ...]


class SmokeDustVariables(BaseModel):
    values: SmokeDustVariablesType

    def get(self, name: str) -> SmokeDustVariable:
        for value in self.values:
            if value.name == name:
                return value
        raise ValueError(name)

    @field_validator("values", mode="after")
    @classmethod
    def _validate_values_(
        cls, values: SmokeDustVariablesType
    ) -> SmokeDustVariablesType:
        names = [ii.name for ii in values]
        if len(names) != len(set(names)):
            raise ValueError("Variable names must be unique")
        return values


SD_VARS = SmokeDustVariables(
    values=(
        SmokeDustVariable(
            name="geolat",
            long_name="cell center latitude",
            units="degrees_north",
            fill_value_str="-9999.f",
            fill_value_float=-9999.0,
        ),
        SmokeDustVariable(
            name="geolon",
            long_name="cell center longitude",
            units="degrees_east",
            fill_value_str="-9999.f",
            fill_value_float=-9999.0,
        ),
        SmokeDustVariable(
            name="frp_avg_hr",
            long_name="mean Fire Radiative Power",
            units="MW",
            fill_value_str="0.f",
            fill_value_float=0.0,
        ),
        SmokeDustVariable(
            name="ebb_smoke_hr",
            long_name="EBB emissions",
            units="ug m-2 s-1",
            fill_value_str="0.f",
            fill_value_float=0.0,
        ),
        SmokeDustVariable(
            name="frp_davg",
            long_name="Daily mean Fire Radiative Power",
            units="MW",
            fill_value_str="0.f",
            fill_value_float=0.0,
        ),
        SmokeDustVariable(
            name="ebb_rate",
            long_name="Total EBB emission",
            units="ug m-2 s-1",
            fill_value_str="0.f",
            fill_value_float=0.0,
        ),
        SmokeDustVariable(
            name="fire_end_hr",
            long_name="Hours since fire was last detected",
            units="hrs",
            fill_value_str="0.f",
            fill_value_float=0.0,
        ),
        SmokeDustVariable(
            name="hwp_davg",
            long_name="Daily mean Hourly Wildfire Potential",
            units="none",
            fill_value_str="0.f",
            fill_value_float=0.0,
        ),
        SmokeDustVariable(
            name="totprcp_24hrs",
            long_name="Sum of precipitation",
            units="m",
            fill_value_str="0.f",
            fill_value_float=0.0,
        ),
    )
)
