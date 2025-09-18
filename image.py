import os
from random import randint
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg') # or 'TkAgg', 'MacOSX'
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

HUE_DELTA = np.array([7, 255, 100])
MID_HUE = np.array([105,  67, 172])

# os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = "/usr/lib/x86_64-linux-gnu/qt5/plugins" 
# os.environ["QT_QPA_PLATFORM"] = "xcb" 

# Deshabilitar funcionalidades Qt de OpenCV
cv2.setUseOptimized(True)

def setup_plot():
    fig, ax = plt.subplots()
    im = ax.imshow(np.zeros((480, 640, 3), dtype=np.uint8))
    ax.axis('off')
    return fig, ax, im

def update_frame(frame_num, image, im):
    processed = procesar_frame(image)
    im.set_array(processed)
    return im,

def procesar_frame(image):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # print(f"Max in image_hsv {np.max(image_hsv, axis=(0, 1))}")
    # mid_hue = np.array(image_hsv[image_hsv.shape[0] // 2, image_hsv.shape[1] // 2, :])
    mid_hue = MID_HUE
    print(f"mid hue: {mid_hue}")

    lower_paper = mid_hue - HUE_DELTA
    upper_paper = mid_hue + HUE_DELTA
    mask_ = cv2.inRange(image_hsv, lower_paper, upper_paper)
    return mark_bounds(image, mask_)
    mask = mask_ / 255
    # image_hsv[:,:,1] = 255
    # image_hsv[:,:,2] = 255
    # mask = np.ones(shape=image_hsv.shape[:2])
    # mask[image_hsv[:,:,0] > mid_hue[0] + HUE_DELTA] = 0
    # mask[image_hsv[:,:,0] < mid_hue[0] - HUE_DELTA] = 0
    # image_hsv[image_hsv[:,:,0] > mid_hue + HUE_DELTA] = 0
    # image_hsv[image_hsv[:,:,0] < mid_hue - HUE_DELTA] = 0
    # image_hsv[image_hsv[:,:,2] > 0] = [mid_hue, 255, 255]
    blured_mask = cv2.GaussianBlur(mask, (35, 35), 0)
    # image_h = cv2.cvtColor(blured_hsv, cv2.COLOR_HSV2RGB)
    # usar_marcadores_aruco(image_h)
    masked = np.array(image * blured_mask[:, :, np.newaxis], dtype=np.uint8)
    return masked


def pre_process(frame):
    # 1. Convertir a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 2. Mejorar contraste
    gray = cv2.equalizeHist(gray)
    
    # 3. Reducir ruido
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 4. Binarización adaptativa (mejor que threshold simple)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)

    return gray


def rand_col():
    return randint(100, 255), randint(100, 255), randint(100, 255)


def detectar_bordes_bobina(image):
    # Preprocesamiento
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    # return edges

    print(f"min edges {np.min(edges)} | max edges {np.max(edges)}")
    
    # Detectar líneas con Hough
    lines = cv2.HoughLinesP(edges, 1, np.pi/90, threshold=15, 
                            minLineLength=15, maxLineGap=10)
    
    edged = np.maximum(image, edges[:,:,np.newaxis])
    for line in lines:
        print(line)
        x0, y0, x1, y1 = line[0]
        cv2.line(edged, (x0, y0), (x1, y1), rand_col(), 1)
    
    return edged


def bound(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return []
    # Encontrar el contorno más grande (la bobina)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Calcular bounding rectangle
    x, y, w, h = cv2.boundingRect(largest_contour)
    print("largest contour:", x, y, w, h)

    return [cv2.boundingRect(largest_contour)]


def mark_bounds(image, mask):
    for rect in bound(mask):
        x, y, w, h = rect
        x0, y0, x1, y1 = x, y, x + w, y + h
        cv2.rectangle(image, (x0, y0), (x1, y1), rand_col(), 1)
    return image


def clean(image):
    image = cv2.resize(image, (300, 200))
    k_size = np.max(image.shape) // 200 * 2 + 1
    print(f"cleaning with kernel size {k_size}")
    kernel = np.ones((k_size, k_size), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return image


def usar_marcadores_aruco(frame):
    # Inicializar el detector ArUco
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    
    # Detectar marcadores
    corners, ids, rejected = detector.detectMarkers(frame)
    
    if ids is not None:
        print("APROVED")
        # Dibujar marcadores detectados
        frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        
        # Calcular poses si es necesario
        # camera_matrix y dist_coeffs deben ser calibrados
        # rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, dist_coeffs)

    elif rejected is not None and len(rejected) > 0:
        print("REJECTED")
        for rejected_corners in rejected:
            cv2.polylines(frame, [rejected_corners.astype(np.int32)], 
                          True, (0, 0, 255), 2)
        
    return corners, ids, frame

def main():
    # image = cv2.imread('buttler_marked.jpeg', cv2.IMREAD_COLOR_RGB)
    image = cv2.imread('banda.jpeg', cv2.IMREAD_COLOR_RGB)
    # image = cv2.imread('doblado.jpeg', cv2.IMREAD_COLOR_RGB)
    # image = pre_process(image)
    # aruco = usar_marcadores_aruco(image)
    image = clean(image)
    image = procesar_frame(image)
    # image = detectar_bordes_bobina(image)
    plt.imshow(image)
    plt.axis('off')
    plt.show()
    # fig, ax, im = setup_plot()
    
    # ani = FuncAnimation(fig, update_frame, fargs=(image, im),
    #                     interval=50, blit=True)
    
    plt.show()


main()
