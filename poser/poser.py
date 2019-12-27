import abc
from typing import List

from torch import Tensor


class PoseParameter:
    def __init__(self,
                 name: str,
                 display_name: str,
                 lower_bound: float,
                 upper_bound: float,
                 default_value: float):
        self._name = name
        self._display_name = display_name
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound
        self._default_value = default_value

    @property
    def name(self):
        return self._name

    @property
    def display_name(self):
        return self._display_name

    @property
    def lower_bound(self):
        return self._lower_bound

    @property
    def upper_bound(self):
        return self._upper_bound

    @property
    def default_value(self):
        return self._default_value


class Poser:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def image_size(self):
        pass

    @abc.abstractmethod
    def pose_parameters(self) -> List[PoseParameter]:
        pass

    @abc.abstractmethod
    def pose(self, source_image: Tensor, pose: Tensor):
        pass
