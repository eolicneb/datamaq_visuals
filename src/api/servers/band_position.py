import asyncio

import cv2
import numpy as np

import settings
from src.api.servers.process import ProcessServer
from src.api.video_stream.mock_video_capture import MockVideoCapture
from src.core.detectors.band_position import EdgeProcessing, process_band
from src.core.detectors.image_methods import stamp
from src.core.detectors.models import Locus, Context, Box


class BandEdgeServer(ProcessServer):
    def __init__(self, *args, processing: EdgeProcessing, **kwargs):
        self.processing: EdgeProcessing = None
        super().__init__(*args, processing=processing, **kwargs)
        self.app.get("/edge")(self.get_band_edge)

    async def get_band_edge(self):
        return {"edge": self.processing.edge.value()}


input_cap_1 = MockVideoCapture(name="input", buffer_size=5, fps=60, no_signal_pattern="rand")
output_cap_1 = MockVideoCapture(name="output", buffer_size=5, fps=60, no_signal_pattern="rand")
mask_cap_1 = MockVideoCapture(name="mask", buffer_size=5, fps=60, no_signal_pattern="rand")
cap_1 = cv2.VideoCapture(2)
pr_1 = EdgeProcessing(focus_box=Box(*settings.BAND_FOCUS_BOX), # Box(50, 200, 300, 100),
                      hue_locus=Locus(*settings.BAND_HUE_FOCUS), # Locus(50, 250),
                      context=Context(mask_area_ratio_validation_threshold=.1,
                                      masked_area_ratio_validation_threshold=.15,
                                      reset_steps=10))
server = BandEdgeServer(name="Band position process",
                        cameras=[input_cap_1, output_cap_1, mask_cap_1],
                        frame_width=400, frame_height=300, fps=60, processing=pr_1)

def process_band_edge():
    while True:
        ret, frame = cap_1.read()
        if not ret:
            asyncio.sleep(.01)
            continue
        pr_1.input = frame
        input_cap_1.put(frame)
        output = process_band(pr_1)
        if pr_1.state.focused:
            stamp(output, f"band edge in {pr_1.edge.value()}", corner="top")
        else:
            stamp(output, f"ERROR: {pr_1.exception.exception}", corner="top")
        stamp(output, f"processed in {pr_1.context.eta:1.5f} ms", corner="bottom")
        output_cap_1.put(output)
        if pr_1.mask is not None:
            mask_cap_1.put(np.repeat(pr_1.mask[:,:,np.newaxis], axis=2, repeats=3))
