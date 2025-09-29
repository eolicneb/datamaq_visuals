from dataclasses import dataclass
from traceback import format_exc

import cv2
import numpy as np

from src.core.detectors.image_methods import set_process_eta, segment_process, triplicate, blur_unfocused
from src.core.detectors.models import Processing, ExceptionReport
from src.core.meassurements.average import AveragedMeasurement


@dataclass
class EdgeProcessing(Processing):
    edge: AveragedMeasurement[int] = AveragedMeasurement[int]("edge", allowed_deviation=2, cast=int)


@set_process_eta
def process_band(processing: EdgeProcessing):
    try:
        if not segment_process(processing):
            raise RuntimeError("Segment process failed")
        processing.output = blur_unfocused(processing.input.copy(), processing.focus_box)
        # if processing.focus_box:
        #     processing.output[not processing.output[processing.focus_box.image_slice]] //= 2
        processing.edge.update(processing.mask_box.right + (processing.focus_box and processing.focus_box.left or 0))
        draw_band_edge(processing, processing.output)
        processing.exception = None
        return processing.output
    except Exception as e:
        processing.state.focused = False
        processing.state.steps += 1
        processing.exception = ExceptionReport(e, format_exc())
        if processing.mask is None:
            return np.zeros(processing.input.shape, dtype=np.uint8)
        if processing.focus_box:
            now_mask = np.zeros(processing.input.shape, dtype=np.uint8)
            now_mask[processing.focus_box.image_slice] = triplicate(processing.mask)
        else:
            now_mask = triplicate(processing.mask) // 2
        return now_mask


def draw_band_edge(processing: EdgeProcessing, image: np.ndarray):
    x, y, w, h = processing.mask_box
    if processing.focus_box:
        x += processing.focus_box.left or 0
        y += processing.focus_box.top or 0
    cv2.line(image, (x+w, y), (x+w, y+h), (0, 255, 0), thickness=2)
