from abc import ABC, abstractmethod
from typing import TypeVar


class MarkedBuffer(list):
    def __init__(self, max_size: int = 10):
        self.max_size = max_size
        super().__init__()
        self.marks = set()

    def append(self, item, marked=False):
        if len(self) >= self.max_size:
            self.marks = {m - 1 for m in self.marks if m > 0}
            self.pop(0)
        if marked:
            self.marks.add(len(self))
        super().append(item)

    @property
    def marked(self):
        for i in self.marks:
            yield self[i]

    @property
    def unmarked(self):
        for i in range(len(self)):
            if i not in self.marks:
                yield self[i]


T = TypeVar('T')


class Measurement(ABC):
    def __init__(self, name: str, value: T = None, unit: str = "", buffer_size: int = 10):
        self.name = name
        self.unit = unit
        self._value = value
        self.history = MarkedBuffer(buffer_size)
        if self._value is not None:
            self.history.append(self._value)

    def update(self, value: T):
        self.history.append(value, marked=self._is_valid(value))

    @abstractmethod
    def _is_valid(self, value: T):
        """Determines if new value is valid for the measurement's history"""

    @property
    @abstractmethod
    def value(self) -> T:
        """Current value of the measurement given the measurement's history"""
