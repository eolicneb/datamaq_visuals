from src.api.video_stream.multi_stream import MultiStreamServer
from src.core.detectors.models import Processing


class ProcessServer(MultiStreamServer):
    def __init__(self, *args, processing: Processing, **kwargs):
        super().__init__(*args, **kwargs)
        self.processing = processing

    async def health_check(self):
        check = await super().health_check()
        check["processing_exception"] = self.processing.exception and {"traceback": self.processing.exception.traceback}
        return check
