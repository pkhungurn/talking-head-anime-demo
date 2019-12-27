import abc
from abc import ABC

from torch.nn import Module


class BatchInputModule(Module, ABC):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def forward_from_batch(self, batch):
        pass


class BatchInputModuleSpec(ABC):
    @abc.abstractmethod
    def get_module(self) -> BatchInputModule:
        pass
