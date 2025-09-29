import asyncio

import cv2
import numpy as np

from src.api.servers.process import ProcessServer
from src.api.video_stream.mock_video_capture import MockVideoCapture
from src.core.detectors.buttler import ButtlerProcessing, process_buttler
from src.core.detectors.image_methods import stamp
from src.core.detectors.models import Context


class ButtlerServer(ProcessServer):
    def __init__(self, *args, processing: ButtlerProcessing, **kwargs):
        self.processing: ButtlerProcessing = None
        super().__init__(*args, processing=processing, **kwargs)
        self.app.get("/reel")(self.get_reel)

    async def get_reel(self):
        return self.processing.reel.dict()


input_cap_0 = MockVideoCapture(name="input", buffer_size=5, fps=60, no_signal_pattern="rand")
output_cap_0 = MockVideoCapture(name="output", buffer_size=5, fps=60, no_signal_pattern="rand")
mask_cap_0 = MockVideoCapture(name="mask", buffer_size=5, fps=60, no_signal_pattern="rand")
cap_0 = cv2.VideoCapture(0)
pr = ButtlerProcessing(context=Context(mask_area_ratio_validation_threshold=.1,
                                       masked_area_ratio_validation_threshold=.15,
                                       reset_steps=10))
server = ButtlerServer(name="Buttler process", cameras=[input_cap_0, output_cap_0, mask_cap_0],
                       frame_width=480, frame_height=320, fps=60, processing=pr)


def update_buttler():
    while True:
        ret, frame = cap_0.read()
        if not ret:
            asyncio.sleep(.01)
            continue
        pr.input = frame
        input_cap_0.put(frame)
        output = process_buttler(pr)
        stamp(output, f"mid_hue in {pr.context.mid_hue}", corner="top")
        stamp(output, f"processed in {pr.context.eta:1.5f} ms", corner="bottom")
        output_cap_0.put(output)
        mask_cap_0.put(np.repeat(pr.mask[:,:,np.newaxis], axis=2, repeats=3))
