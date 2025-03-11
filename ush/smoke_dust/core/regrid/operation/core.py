import abc

import esmpy

from smoke_dust.core.regrid.operation.context import RegridOperationContext


class AbstractRegridOperation(abc.ABC):

    def __init__(self, context: RegridOperationContext) -> None:
        self.context = context
        self._esmpy_manager = esmpy.Manager(debug=self.context.debug)

    @abc.abstractmethod
    def run(self) -> None: ...

    def finalize(self) -> None:
        self.log.info(f"finalizing regrid operation: {self._spec.name}")
