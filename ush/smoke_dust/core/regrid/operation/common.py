import esmpy
import pandas as pd
from pydantic import BaseModel, ConfigDict

from smoke_dust.core.context import SmokeDustContext


class EsmpyContext(BaseModel):
    model_config = ConfigDict(frozen=True)

    regrid_method: int
    zero_region: int
    debug: bool = False
    ignore_degenerate: bool = False
    unmapped_action: int = esmpy.UnmappedAction.ERROR


class RegridOperationContext(SmokeDustContext):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    cycle_metadata: pd.DataFrame
    create_weight_file: bool = False


class RegridFieldName(BaseModel):
    src: str
    dst: str


