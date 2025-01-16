import abc
import datetime
from typing import Dict

from smoke_dust_context import SmokeDustContext, EmissionVariable
import numpy as np


class AbstractSmokeDustCycle(abc.ABC):

    def __init__(self, context: SmokeDustContext):
        self._context = context

    @abc.abstractmethod
    def create_start_datetime(self) -> datetime.datetime:
        ...

    @abc.abstractmethod
    def process_emissions(self) -> Dict[EmissionVariable, np.ndarray]:
        ...