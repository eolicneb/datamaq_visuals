from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class Context:
    mid_hue: Any = None
    validation_mask: Any = None
    validation_mask_area: Optional[int] = None
    mask_area_ratio_validation_threshold: Optional[float] = None
    masked_area_ratio_validation_threshold: Optional[float] = None
    reset_steps: int = 0
    eta: Optional[float] = None
    camera_to_target_distance: Optional[int] = None

    def reset(self):
        self.mid_hue = None
        self.validation_mask = None
        self.validation_mask_area = None


@dataclass
class State:
    focused: bool = False
    steps: int = 0
    context: Context = field(default_factory=Context)

    @property
    def need_reset(self):
        if not self.focused:
            return False
        return self.context.reset_steps and self.context.reset_steps <= self.steps

    def reset(self):
        self.steps = 0
        self.focused = False
        self.context.reset()

    def set_focused(self):
        self.focused = True
        self.steps = 0

@dataclass
class ExceptionReport:
    exception: Exception = None
    traceback: Any = None


@dataclass
class Box:
    left: Optional[int] = 0
    top: Optional[int] = 0
    width: Optional[int] = None
    height: Optional[int] = None

    def __bool__(self):
        return True

    def __iter__(self):
        return iter((self.left, self.top, self.width, self.height))

    @property
    def right(self):
        return (self.left is not None and self.width is not None) and (self.left + self.width) or None

    @property
    def bottom(self):
        return (self.top is not None and self.height is not None) and (self.top + self.height) or None

    @property
    def slice(self):
        return slice(self.top or 0, self.bottom), slice(self.left or 0, self.right)

    @property
    def image_slice(self):
        return *self.slice, slice(None)


@dataclass
class Locus:
    x: Optional[int] = None
    y: Optional[int] = None
    _x: Optional[int] = None
    _y: Optional[int] = None

    def setdefault(self, x=None, y=None):
        self._x = x
        self._y = y

    def __iter__(self):
        for value in self._get_values():
            yield value

    def _get_values(self):
        return self._x if self.x is None else self.x, self._y if self.y is None else self.y

    @property
    def slice(self):
        x, y = self._get_values()
        return self and (y, x) or None

    @property
    def image_slice(self):
        return self.slice and tuple((*self.slice, slice(None)))

    def __bool__(self):
        x, y = self._get_values()
        return x is not None and y is not None


@dataclass
class Processing:
    input: Any = None
    mid_hue: Any = None
    context: Context = field(default_factory=Context)
    hue_input: Any = None
    hue_locus: Locus = None
    focus: Any = None
    focus_box: Box = None
    mask: Any = None
    mask_box: Box = None
    output: Any = None
    diameter: Optional[int] = None
    state: State = field(default_factory=State)
    exception: Optional[ExceptionReport] = None

    def __post_init__(self):
        self.state.context = self.context
        self._set_default_hue_locus()

    def _set_default_hue_locus(self):
        if self.input is not None:
            if self.hue_locus is not None:
                self.hue_locus.setdefault(self.input.shape[1]//2, self.input.shape[0]//2)
            else:
                self.hue_locus = Locus(self.input.shape[1]//2, self.input.shape[0]//2)

    @property
    def hue_locus_slice(self):
        self._set_default_hue_locus()
        if self.focus is None:
            if self.input is None:
                return
            shape = self.input.shape[:2]
            focus_box = Box() if self.focus_box is None else self.focus_box
        else:
            shape = self.focus.shape[:2]
            focus_box = Box()
        x, y = self.hue_locus
        hue_x, hue_y = x - focus_box.left, y - focus_box.top
        hue_x = min(hue_x, focus_box.width or shape[1])
        hue_y = min(hue_y, focus_box.height or shape[0])
        return Locus(hue_x, hue_y).slice

    @property
    def hue_locus_image_slice(self):
        return tuple((*self.hue_locus_slice, slice(None)))
