from dataclasses import dataclass, field
from traceback import format_exc
from typing import Optional, Any
import logging

import cv2
import numpy as np

from src.core.detectors.image_methods import segment_process, triplicate, set_process_eta
from src.core.detectors.models import Processing, ExceptionReport, Box
from src.core.meassurements.average import AveragedMeasurement


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

        processing.output = processing.input.copy()
        draw_reel_box(processing, image=processing.output)
        # mask = np.dstack((processing.mask, processing.mask, processing.mask))
        # print(processing.state)
        processing.exception = None
        return processing.output
    except Exception as e:
        # print(processing.state)
        processing.state.focused = False
        processing.state.steps += 1
        processing.exception = ExceptionReport(e, format_exc())
        now_mask = triplicate(processing.mask) // 2
        return now_mask


def draw_reel_box(processing, image=None):
    output = image if image is not None else processing.input
    x, y, w, h = processing.reel.box
    cv2.rectangle(output, (x, y), (x+w, y+h), (255, 255, 0), thickness=3)


def reel_dimensions(processing: ButtlerProcessing):
    x, y, w, h = processing.mask_box
    processing.reel.diameter.update(h)  # horiz_line(processing.hue_input[y0:y0+h, x0:x0+w, 1])
    radius = processing.reel.diameter.value() // 2
    processing.reel.width.update(np.sum(processing.mask[radius+y, :] > 0))
    reel_x0 = np.sum(processing.mask[radius+y,x:x+processing.reel.width.value()] < 1)
    processing.reel.box = x + reel_x0, y, processing.reel.width.value(), processing.reel.diameter.value()
