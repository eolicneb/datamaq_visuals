from dataclasses import dataclass, field
from traceback import format_exc
from typing import Optional, Any
import logging

import cv2
import numpy as np

from src.core.detectors.image_methods import segment_process, triplicate, set_process_eta, blur_unfocused, \
    put_icon_in_image
from src.core.detectors.models import Processing, ExceptionReport, Box
from src.core.meassurements.average import AveragedMeasurement
from src.resources.graphics.cross import cross

logger = logging.getLogger(__name__)

@dataclass
class Reel:
    diameter: AveragedMeasurement = AveragedMeasurement[int]("diameter", allowed_deviation=2, cast=int)
    width: AveragedMeasurement = AveragedMeasurement[int]("width", allowed_deviation=2, cast=int)
    box: Optional[Box] = None

    def dict(self):
        return {"diameter": self.diameter and self.diameter.value(),
                "width": self.width and self.width.value(),
                "box": self.box and [int(x) for x in self.box]}


@dataclass
class ButtlerProcessing(Processing):
    reel: Reel = field(default_factory=Reel)


@set_process_eta
def process_buttler(processing: ButtlerProcessing):
    try:
        if not segment_process(processing):
            raise RuntimeError("Segment process failed")

        reel_dimensions(processing)

        processing.output = blur_unfocused(processing.input.copy(), processing.focus_box)
        # processing.output = processing.input.copy()
        draw_reel_box(processing, image=processing.output)
        # mask = np.dstack((processing.mask, processing.mask, processing.mask))
        # print(processing.state)
        put_icon_in_image(processing.output, cross(), processing.hue_locus)
        processing.exception = None
        return processing.output
    except Exception as e:
        # print(processing.state)
        processing.state.focused = False
        processing.state.steps += 1
        processing.exception = ExceptionReport(e, format_exc())
        if processing.mask is None:
            op = processing.input.copy() // 4 + 127
            put_icon_in_image(op, cross(), processing.hue_locus)
            return op
        now_mask = triplicate(processing.mask) // 2
        # put_icon_in_image(now_mask, cross(), processing.hue_locus)
        return now_mask


def draw_reel_box(processing, image=None):
    output = image if image is not None else processing.input
    x, y, w, h = processing.reel.box
    cv2.rectangle(output, (x, y), (x+w, y+h), (255, 255, 0), thickness=3)


def reel_dimensions(processing: ButtlerProcessing):
    f_x, f_y, f_w, f_h = processing.focus_box if processing.focus_box else (0, 0, 0, 0)
    x, y, w, h = processing.mask_box
    processing.reel.diameter.update(h)  # horiz_line(processing.hue_input[y0:y0+h, x0:x0+w, 1])
    radius = processing.reel.diameter.value() // 2
    processing.reel.width.update(np.sum(processing.mask[radius+y, :] > 0))
    reel_x0 = np.sum(processing.mask[radius+y,x:x+processing.reel.width.value()] < 1)
    processing.reel.box = x + f_x + reel_x0, y + f_y, processing.reel.width.value(), processing.reel.diameter.value()
