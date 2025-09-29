import cv2
import numpy as np
import threading
import time
from typing import Optional, Tuple, Callable
import logging

logger = logging.getLogger(__name__)


NO_SIGNAL_PATTERNS = ["black", "rand"]


class MockVideoCapture:
    """
    Clase que imita cv2.VideoCapture pero con buffer interno.
    Útil para testing, simulaciones o cuando necesitas alimentar frames manualmente.
    """

    def __init__(self, name: str = "", buffer_size: int = 10, fps: float = 30.0, no_signal_pattern: str = "black",
                 callback: Optional[Callable[[], np.ndarray]] = None):
        """
        Inicializa el mock de VideoCapture.

        Args:
            buffer_size: Tamaño máximo del buffer de frames
            fps: Frames por segundo simulados
        """
        self.name = name

        self.callback = callback

        self.buffer_size = buffer_size
        self.fps = fps
        self.frame_interval = 1.0 / fps if fps > 0 else 0.033

        # Buffer de frames
        self.buffer = []
        self.current_frame = None
        self.buffer_lock = threading.Lock()

        self.no_signal_pattern = no_signal_pattern

        # Estado de la cámara
        self.is_opened = False
        self.properties = {
            cv2.CAP_PROP_FRAME_WIDTH: 640,
            cv2.CAP_PROP_FRAME_HEIGHT: 480,
            cv2.CAP_PROP_FPS: fps,
            cv2.CAP_PROP_POS_FRAMES: 0,
            cv2.CAP_PROP_FRAME_COUNT: 0,
            cv2.CAP_PROP_BRIGHTNESS: 0.5,
            cv2.CAP_PROP_CONTRAST: 0.5,
            cv2.CAP_PROP_SATURATION: 0.5,
            cv2.CAP_PROP_HUE: 0.5,
            cv2.CAP_PROP_GAIN: 0.5,
            cv2.CAP_PROP_EXPOSURE: 0.5,
        }

        # Contadores
        self.frame_count = 0
        self.last_read_time = 0

        # Hilo para generación automática de frames (opcional)
        self.generation_thread = None
        self.generation_running = False

        logger.info(f"MockVideoCapture inicializado (buffer: {buffer_size}, FPS: {fps})")

    def getBackendName(self):
        return self.name

    def put(self, frame: np.ndarray) -> bool:
        """
        Agrega un frame al buffer.

        Args:
            frame: Frame a agregar (numpy array)

        Returns:
            bool: True si se agregó correctamente
        """
        if not isinstance(frame, np.ndarray):
            logger.error("El frame debe ser un numpy array")
            return False

        with self.buffer_lock:
            # Limitar el tamaño del buffer
            if len(self.buffer) >= self.buffer_size:
                self.buffer.pop(0)  # Remover el frame más antiguo

            self.buffer.append(frame.copy())
            self.properties[cv2.CAP_PROP_FRAME_COUNT] = len(self.buffer)
            logger.debug(f"Frame agregado al buffer. Tamaño actual: {len(self.buffer)}")

        return True

    def put_from_file(self, image_path: str) -> bool:
        """
        Carga una imagen desde archivo y la agrega al buffer.

        Args:
            image_path: Ruta a la imagen

        Returns:
            bool: True si se cargó correctamente
        """
        try:
            frame = cv2.imread(image_path)
            if frame is None:
                logger.error(f"No se pudo cargar la imagen: {image_path}")
                return False
            return self.put(frame)
        except Exception as e:
            logger.error(f"Error cargando imagen {image_path}: {e}")
            return False

    def generate_test_pattern(self, pattern: str = "color_bars") -> bool:
        """
        Genera un patrón de prueba y lo agrega al buffer.

        Args:
            pattern: Tipo de patrón ("color_bars", "gradient", "checkerboard")

        Returns:
            bool: True si se generó correctamente
        """
        width = int(self.properties[cv2.CAP_PROP_FRAME_WIDTH])
        height = int(self.properties[cv2.CAP_PROP_FRAME_HEIGHT])

        try:
            if pattern == "color_bars":
                frame = self._create_color_bars(width, height)
            elif pattern == "gradient":
                frame = self._create_gradient(width, height)
            elif pattern == "checkerboard":
                frame = self._create_checkerboard(width, height)
            else:
                logger.error(f"Patrón no reconocido: {pattern}")
                return False

            return self.put(frame)

        except Exception as e:
            logger.error(f"Error generando patrón {pattern}: {e}")
            return False

    def _create_color_bars(self, width: int, height: int) -> np.ndarray:
        """Crea un frame con barras de colores."""
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        colors = [
            (255, 255, 255),  # Blanco
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (255, 0, 0),  # Azul
            (0, 255, 255),  # Amarillo
            (0, 255, 0),  # Verde
            (0, 0, 255),  # Rojo
            (0, 0, 0),  # Negro
        ]

        bar_width = width // len(colors)
        for i, color in enumerate(colors):
            start_x = i * bar_width
            end_x = (i + 1) * bar_width if i < len(colors) - 1 else width
            frame[:, start_x:end_x] = color

        return frame

    def _create_gradient(self, width: int, height: int) -> np.ndarray:
        """Crea un frame con gradiente de colores."""
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        for y in range(height):
            for x in range(width):
                frame[y, x] = [
                    int(255 * x / width),  # Rojo
                    int(255 * y / height),  # Verde
                    int(255 * (x + y) / (width + height))  # Azul
                ]

        return frame

    def _create_checkerboard(self, width: int, height: int) -> np.ndarray:
        """Crea un frame con patrón de tablero de ajedrez."""
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        square_size = 40

        for y in range(height):
            for x in range(width):
                if (x // square_size + y // square_size) % 2 == 0:
                    frame[y, x] = (255, 255, 255)  # Blanco
                else:
                    frame[y, x] = (0, 0, 0)  # Negro

        return frame

    def _create_rand(self, width: int, height: int) -> np.ndarray:
        """Crea un frame con patrón de tablero de ajedrez."""
        frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        return frame

    def start_auto_generation(self, pattern: str = "color_bars", interval: float = 0.5):
        """
        Inicia la generación automática de frames.

        Args:
            pattern: Patrón a generar
            interval: Intervalo entre frames en segundos
        """
        if self.generation_running:
            logger.warning("La generación automática ya está en ejecución")
            return

        self.generation_running = True
        self.generation_thread = threading.Thread(
            target=self._auto_generation_worker,
            args=(pattern, interval),
            daemon=True
        )
        self.generation_thread.start()
        logger.info(f"Iniciada generación automática de frames ({pattern}, intervalo: {interval}s)")

    def stop_auto_generation(self):
        """Detiene la generación automática de frames."""
        self.generation_running = False
        if self.generation_thread:
            self.generation_thread.join(timeout=1.0)
        logger.info("Generación automática detenida")

    def _auto_generation_worker(self, pattern: str, interval: float):
        """Hilo worker para generación automática."""
        while self.generation_running:
            self.generate_test_pattern(pattern)
            time.sleep(interval)

    # Métodos que imitan cv2.VideoCapture

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Lee el siguiente frame del buffer (similar a cv2.VideoCapture.read()).

        Returns:
            Tuple[bool, Optional[np.ndarray]]: (éxito, frame)
        """
        current_time = time.time()

        # Simular FPS limitando la tasa de lectura
        if current_time - self.last_read_time < self.frame_interval:
            time.sleep(0.001)  # Pequeña pausa

        if self.callback:
            image = self.callback()
            if image is not None:
                width = int(self.properties[cv2.CAP_PROP_FRAME_WIDTH])
                height = int(self.properties[cv2.CAP_PROP_FRAME_HEIGHT])
                cv2.resize(image, (width, height))
                self.put(image)

        with self.buffer_lock:
            if self.buffer:
                self.current_frame = self.buffer[-1]  # Último frame
                self.frame_count += 1
                self.properties[cv2.CAP_PROP_POS_FRAMES] = self.frame_count
                self.last_read_time = current_time
                return True, self.current_frame.copy()
            else:
                if self.no_signal_pattern == "black":
                    # Si no hay frames, devolver un frame negro
                    width = int(self.properties[cv2.CAP_PROP_FRAME_WIDTH])
                    height = int(self.properties[cv2.CAP_PROP_FRAME_HEIGHT])
                    black_frame = np.zeros((height, width, 3), dtype=np.uint8)
                    self.last_read_time = current_time
                    return False, black_frame
                elif self.no_signal_pattern == "rand":
                    width = int(self.properties[cv2.CAP_PROP_FRAME_WIDTH])
                    height = int(self.properties[cv2.CAP_PROP_FRAME_HEIGHT])
                    return True, self._create_rand(width, height)

    def grab(self) -> bool:
        """
        Captura el siguiente frame (similar a cv2.VideoCapture.grab()).

        Returns:
            bool: True si se capturó correctamente
        """
        success, _ = self.read()
        return success

    def retrieve(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Decodifica y devuelve el frame capturado (similar a cv2.VideoCapture.retrieve()).

        Returns:
            Tuple[bool, Optional[np.ndarray]]: (éxito, frame)
        """
        return self.read()

    def set(self, prop_id: int, value: float) -> bool:
        """
        Establece una propiedad (similar a cv2.VideoCapture.set()).

        Args:
            prop_id: ID de la propiedad
            value: Valor a establecer

        Returns:
            bool: True si se estableció correctamente
        """
        if prop_id in self.properties:
            self.properties[prop_id] = value
            logger.debug(f"Propiedad {prop_id} establecida a {value}")
            return True
        else:
            logger.warning(f"Propiedad {prop_id} no reconocida")
            return False

    def get(self, prop_id: int) -> float:
        """
        Obtiene una propiedad (similar a cv2.VideoCapture.get()).

        Args:
            prop_id: ID de la propiedad

        Returns:
            float: Valor de la propiedad
        """
        return self.properties.get(prop_id, 0.0)

    def isOpened(self) -> bool:
        """
        Verifica si la cámara está abierta.

        Returns:
            bool: True si está abierta
        """
        return self.is_opened

    def open(self) -> bool:
        """
        Abre la cámara mock.

        Returns:
            bool: True si se abrió correctamente
        """
        self.is_opened = True
        logger.info("MockVideoCapture abierto")
        return True

    def release(self):
        """Libera los recursos (similar a cv2.VideoCapture.release())."""
        self.is_opened = False
        self.generation_running = False

        with self.buffer_lock:
            self.buffer.clear()

        if self.generation_thread:
            self.generation_thread.join(timeout=1.0)

        logger.info("MockVideoCapture liberado")

    def get_buffer_size(self) -> int:
        """Devuelve el tamaño actual del buffer."""
        with self.buffer_lock:
            return len(self.buffer)

    def clear_buffer(self):
        """Limpia el buffer de frames."""
        with self.buffer_lock:
            self.buffer.clear()
        logger.info("Buffer limpiado")


# Ejemplo de uso
if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(level=logging.INFO)

    # Crear instancia
    mock_cam = MockVideoCapture(buffer_size=5, fps=10)

    # Abrir la cámara
    mock_cam.open()

    # Agregar algunos frames de prueba
    mock_cam.generate_test_pattern("color_bars")
    mock_cam.generate_test_pattern("gradient")

    # Leer frames (simulando uso normal)
    for i in range(10):
        success, frame = mock_cam.read()
        if success:
            print(f"Frame {i}: {frame.shape}")
            # Mostrar el frame (opcional)
            cv2.imshow('Mock Camera', frame)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break

    # Liberar recursos
    mock_cam.release()
    cv2.destroyAllWindows()