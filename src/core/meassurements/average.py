from typing import cast

import numpy as np
from typing_extensions import TypeVar

from .base import Measurement


N = TypeVar('N', bound=float | int)


class AveragedMeasurement[N](Measurement):
    def __init__(self, name: str, value: N = None, allowed_deviation: float = 0.1,
                 cast=None,
                 **kwargs):
        super().__init__(name, value, **kwargs)
        self.allowed_deviation = allowed_deviation
        self._deviation = None
        self._average = None
        self.cast = cast
        self.recovering = False

    def update(self, value: N):
        if value is None:
            return
        super().update(value)
        self._reset()

    def _reset(self):
        self._deviation = None
        self._average = None

    def _calculate_deviation(self):
        return np.std(list(self.history.marked), ddof=1)

    def _calculate_average(self):
        return np.mean(list(self.history.marked))

    @property
    def deviation(self):
        if self._deviation is None:
            self._deviation = self._calculate_deviation()
        return self._deviation

    @property
    def average(self):
        if self._average is None:
            self._average = self._calculate_average()
        return self._average

    def _is_valid(self, value: N):
        if len(self.history) < self.history.max_size:
            return True
        if not self.history.marks:
            self.recovering = True
            return True
        if self.recovering:
            if len(self.history.marks) == self.history.max_size:
                self.recovering = False
            return True
        return np.abs(value - self.average) < self.allowed_deviation * self.deviation

    def value(self) -> N:
        if np.isnan(self.average):
            return None
        if not self.cast:
            return self.average
        return self.cast(self.average)
