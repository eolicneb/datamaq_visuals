import asyncio

import uvicorn
from fastapi import FastAPI, Response
import logging
import cv2
import threading

from mpl_toolkits.axes_grid1.axes_size import from_any
from starlette.responses import StreamingResponse

# app = FastAPI()

logger = logging.getLogger(__name__)


class MultiCameraStream:
    def __init__(self, sources=None, cameras=None, frame_width=320, frame_height=240, fps=30):
        self.sources = sources or []
        self.cameras = cameras or[]
        self.names = []
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.fps = fps

        self.frames = [None] * len(sources or cameras)
        self.stopped = True

    def __len__(self):
        return len(self.cameras or self.sources or [])

    def init_cameras(self):
        logger.info("Initializing cameras")
        if not self.cameras and self.sources:
            for i, src in enumerate(self.sources):
                cap = cv2.VideoCapture(src)
                self.cameras.append(cap)
        if not self.cameras:
            raise ValueError(f"No cameras found for sources {self.sources}")
        for cap in self.cameras:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            cap.set(cv2.CAP_PROP_FPS, self.fps)
            self.names.append(cap.getBackendName())

    def start(self):
        self.init_cameras()
        self.stopped = False
        threading.Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while not self.stopped:
            for i, camera in enumerate(self.cameras):
                ret, frame = camera.read()
                if ret:
                    self.frames[i] = frame

    def get_frame(self, camera_index):
        try:
            return self.frames[camera_index]
        except:
            logger.error("Error getting frame for camera", camera_index)
            raise

    def stop(self):
        logger.info("Stopping cameras")
        self.stopped = True
        for camera in self.cameras:
            camera.release()


class MultiStreamServer:
    def __init__(self, name="Stream", sources=None, cameras=None, frame_width=320, frame_height=240, fps=30):
        self.name = name
        self.multi_stream = MultiCameraStream(sources=sources, cameras=cameras,
                                              frame_width=frame_width, frame_height=frame_height, fps=fps)
        self.app = FastAPI()
        self.app.get('/')(self.index)
        self.app.get('/camera/{camera_id}')(self.camera_feed)
        self.app.get('/health')(self.health_check)

        self.app.add_event_handler("startup", self.startup_event)
        self.app.add_event_handler("shutdown", self.shutdown_event)

        logger.info("MultiStreamServer initialized")

    async def startup_event(self):
        """Inicializar cámaras asíncronamente"""
        logger.info("Inicializando cámaras...")
        self.multi_stream.start()

    async def shutdown_event(self):
        """Liberar recursos asíncronamente"""
        if self.multi_stream and not self.multi_stream.stopped:
            self.multi_stream.stop()

    async def run_async(self, host="0.0.0.0", port=5000, reload=False):
        """Ejecutar servidor de forma asíncrona"""
        config = uvicorn.Config(
            app=self.app,
            host=host,
            port=port,
            reload=reload,
            log_level="info"
        )

        server = uvicorn.Server(config)
        await server.serve()

    def generate_multi_frames(self, camera_index):
        logger.info(f"Generating frames for camera {camera_index}")
        while True:
            frame = self.multi_stream.get_frame(camera_index)
            if frame is not None:
                ret, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


    async def camera_feed(self, camera_id: int):
        return StreamingResponse(self.generate_multi_frames(camera_id),
                                 media_type='multipart/x-mixed-replace; boundary=frame')

    async def index(self):
        """Página principal"""
        html_content = """
        <html>
        <head>
            <meta charset="UTF-8">
            <title>#name#</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }
                .container {
                    display: flex;
                    flex-wrap: wrap;
                    justify-content: center;
                    gap: 20px;
                    max-width: 1200px;
                    margin: 0 auto;
                }
                .camera {
                    background: white;
                    padding: 15px;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    text-align: center;
                }
                .camera img {
                    border-radius: 5px;
                    border: 1px solid #ddd;
                }
                h1 {
                    text-align: center;
                    color: #333;
                    margin-bottom: 30px;
                }
                .status {
                    text-align: center;
                    margin-top: 20px;
                }
            </style>
        </head>
        <body>
            <h1>#name# - Servidor de Cámaras</h1>
            <div class="container">
        """.replace("#name#", self.name)

        # Generar dinámicamente las cámaras disponibles
        camera_count = len(self.multi_stream.cameras) if self.multi_stream else 0
        for i in range(camera_count):
            html_content += f"""
                <div class="camera">
                    <img src="/camera/{i}" width="{self.multi_stream.frame_width}" height="{self.multi_stream.frame_height}" alt="Cámara {i+1}">
                    <p>Cámara {i+1} - {self.multi_stream.names[i]}</p>
                </div>
            """

        html_content += """
            </div>
            <div class="status">
                <p><a href="/health">Ver estado del sistema</a></p>
            </div>
        </body>
        </html>
        """

        return Response(html_content, media_type="text/html")

    async def health_check(self):
        """Endpoint para verificar estado de las cámaras"""
        if not self.multi_stream:
            return {"status": "error", "message": "Stream no inicializado"}

        if self.multi_stream.stopped:
            return {"status": "error", "message": "Stream detenido"}

        camera_status = {}
        for i in range(len(self.multi_stream.cameras)):
            camera_status[f"camera_{i}"] = {
                "available": i < len(self.multi_stream.frames) and self.multi_stream.frames[i] is not None,
                "source": self.multi_stream.names[i] if i < len(self.multi_stream) else "unknown"
            }

        return {
            "status": "ok" if any(camera_status.values()) else "error",
            "cameras": camera_status
        }


if __name__ == '__main__':
    from mock_video_capture import MockVideoCapture
    import numpy as np

    def rand_frame():
        return np.random.randint(0, 255, (100, 60, 3), dtype=np.uint8) // (1.5, 2, 4)

    mock_cap = MockVideoCapture(name="rand", callback=rand_frame, buffer_size=5, fps=60, no_signal_pattern="rand")
    cap2 = cv2.VideoCapture(0)

    logging.basicConfig(level=logging.INFO)
    server = MultiStreamServer(sources=[0, 2], cameras=[mock_cap, cap2], frame_width=640, frame_height=480, fps=60)
    asyncio.run(server.run_async())
