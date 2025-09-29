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

    @property
    def slice(self):
        return self and (self.y, self.x) or None

    @property
    def image_slice(self):
        return self.slice and tuple((*self.slice, slice(None)))

    def __bool__(self):
        return self.x is not None and self.y is not None


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

    @property
    def hue_locus_slice(self):
        if self.hue_locus is None and self.input is not None:
            return self.input.shape[0]//2, self.input.shape[1]//2
        if self.focus_box is None:
            return self.hue_locus.slice
        return Locus(self.hue_locus.y - self.focus_box.top, self.hue_locus.x - self.focus_box.left).slice

    @property
    def hue_locus_image_slice(self):
        return tuple((*self.hue_locus_slice, slice(None)))
