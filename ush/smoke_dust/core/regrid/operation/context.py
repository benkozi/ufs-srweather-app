from logging import Logger

import esmpy
from pydantic import BaseModel, ConfigDict


class EsmpyContext(BaseModel):
    model_config = ConfigDict(frozen=True)

    regrid_method: int
    zero_region: int
    debug: bool = False
    ignore_degenerate: bool = False
    unmapped_action: int = esmpy.UnmappedAction.ERROR


class RegridOperationContext(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    esmpy_context: EsmpyContext