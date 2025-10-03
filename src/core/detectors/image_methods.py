import timeit
from traceback import format_exc
from typing import Callable

import cv2
import numpy as np
import matplotlib

from src.core.detectors.models import Context, Processing, Box, ExceptionReport, Locus

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from settings import HUE_DELTA


# Deshabilitar funcionalidades Qt de OpenCV
cv2.setUseOptimized(True)


def segment_process(processing: Processing):
    if processing.state.need_reset:
        processing.context.mask = None
        processing.state.reset()
    elif not processing.state.focused:
        processing.context.reset()
    if processing.focus_box and processing.focus is None:
        processing.focus = processing.input[processing.focus_box.image_slice]
    input_image = processing.input if processing.focus is None else processing.focus
    processing.hue_input = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
    processing.focus = None
    update_mid_hue(processing)

    mask = segment_by_hue(processing.hue_input, processing.context.mid_hue)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False
    largest_contour = max(contours, key=cv2.contourArea)
    single_mask = np.zeros(mask.shape, dtype=np.uint8)
    cv2.drawContours(single_mask, [largest_contour], 0, 255, -1)
    processing.mask = single_mask
    processing.hue_input = cv2.bitwise_and(processing.hue_input, processing.hue_input, mask=single_mask)
    processing.mask_box = Box(*cv2.boundingRect(largest_contour))
    processing.context.mid_hue = segment_hue(processing)
    processing.mid_hue = processing.context.mid_hue
    if check_segment_validity(processing):
        processing.state.set_focused()
        return True


def check_segment_validity(processing: Processing):
    def new_validation(processing: Processing):
        found_mask_area = np.sum(processing.mask > 0)
        if not found_mask_area:
            raise RuntimeError("Found not first mask")
        mask_area = processing.mask.shape[0] * processing.mask.shape[1]
        is_valid = found_mask_area > processing.context.mask_area_ratio_validation_threshold * mask_area
        if not is_valid:
            raise RuntimeError(f"First mask area is below {processing.context.mask_area_ratio_validation_threshold} %")
        processing.context.validation_mask = processing.mask
        processing.context.validation_mask_area = found_mask_area
        processing.context.mid_hue = segment_hue(processing)
        # print(f"First mask area: {found_mask_area}")
        return True

    def validate_against_previous_mask(processing: Processing):
        if processing.mask is None or np.sum(processing.mask > 0) == 0:
            raise RuntimeError("Found not mask at all")

        mask_diff = np.abs(processing.mask - processing.context.validation_mask)
        mask_diff[mask_diff > 0] = 255
        mask_diff_area = np.sum(mask_diff > 0)
        masked_area = np.sum(processing.context.validation_mask > 0)
        max_diff_area_allowed = processing.context.masked_area_ratio_validation_threshold * masked_area
        is_valid = mask_diff_area < max_diff_area_allowed
        if not is_valid:
            if not masked_area:
                raise RuntimeError("Found not mask at all")
            else:
                raise RuntimeError(f"Mask diff is above {processing.context.masked_area_ratio_validation_threshold} %")

        processing.context.validation_mask = processing.mask
        processing.context.validation_mask_area = np.sum(processing.mask > 0)
        # set_mask_mid_hue(processing)
        return True

    if processing.context.validation_mask is None:
        return new_validation(processing)
    else:
        return validate_against_previous_mask(processing)


def set_mask_mid_hue(processing: Processing):
    mask_mid_hue = segment_hue(processing)
    processing.context.mid_hue = mask_mid_hue * 0.1 + processing.context.mid_hue * 0.9


def segment_hue(processing: Processing):
    return np.average(processing.hue_input[processing.hue_input[:,:,0] > 0], axis=0)


def horiz_edges(image_2d):
    kernel = np.array([[ 1,  2,  1],
                       [ 0,  0,  0],
                       [-1, -2, -1]])
    # kernel = np.array([[ 1,  1,  2,  1,  1],
    #                    [ 1,  1,  2,  1,  1],
    #                    [ 0,  0,  0,  0,  0],
    #                    [-1, -1, -2, -1, -1],
    #                    [-1, -1, -2, -1, -1]])
    return cv2.filter2D(image_2d, -1, kernel=kernel)


def horiz_line(image_2d):
    center_x = image_2d.shape[0] // 2
    edges = horiz_edges(image_2d[:, center_x - 10:center_x + 10])
    peak_columns = np.argmax(edges, axis=0)
    return int(np.average(peak_columns))


def update_mid_hue(processing: Processing):
    h, w, d = processing.hue_input.shape
    mid_hue = processing.hue_input[processing.hue_locus_image_slice]
    last_hue = mid_hue if processing.context.mid_hue is None else processing.context.mid_hue
    processing.context.mid_hue = mid_hue * 0.1 + last_hue * 0.9


def segment_by_hue(image_hsv, mid_hue):
    lower_paper = mid_hue - HUE_DELTA
    upper_paper = mid_hue + HUE_DELTA
    mask = (cv2.inRange(image_hsv, lower_paper, upper_paper) != 0).astype(np.uint8)
    mask[mask > 0] = 255
    return mask


def stamp(image, text, corner="bottom", org=None, margin=50):
    if image is None:
        return
    org_dispatch = {
        'bottom': lambda image: (margin, image.shape[0] - margin),
        'top': lambda image: (margin, margin),
        'mid': lambda image: (margin, image.shape[0]//2),
    }
    org = org or org_dispatch[corner](image)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = .5
    lineType = cv2.LINE_AA
    color = (0, 0, 0) # Black color
    thickness = 2
    cv2.putText(image, text, org, font, fontScale, color, thickness, lineType)
    color = (0, 255, 255) # Green color
    thickness = 1
    cv2.putText(image, text, org, font, fontScale, color, thickness, lineType)


def blur_unfocused(image, focus_box: Box):
    if not focus_box:
        return image
    blured = image.copy() // 2
    blured[focus_box.image_slice] = image[focus_box.image_slice]
    return blured


def put_icon_in_image(image:np.ndarray, icon:np.ndarray, locus:Locus):
    x, y = locus
    rad_x, rad_y = icon.shape[1] // 2, icon.shape[0] // 2
    icon_x, icon_y = rad_x, rad_y
    icon_slice = (slice(icon_y - rad_y, icon_y + rad_y + 1),
                  slice(icon_x - rad_x, icon_x + rad_x + 1),
                  slice(None, None))
    slice_x ,slice_y = slice(x - rad_x, x + rad_x + 1), slice(y - rad_y, y + rad_y + 1)
    # slice_x = slice(max(slice_x[0], 0), min(slice_x[1], image.shape[1]))
    # slice_y = slice(max(slice_y[0], 0), min(slice_y[1], image.shape[0]))
    place_icon = icon[*icon_slice]
    place_icon[np.all(place_icon==0, axis=2)] = image[slice_y,slice_x,:][np.all(place_icon==0, axis=2)]
    image[slice_y,slice_x,:] = place_icon
    return image


def triplicate(mask):
    return np.dstack((mask, mask, mask))


def set_process_eta(arg: Processing | Callable):
    processing = None
    def decorator(func):
        def _inner(*args, **kwargs):
            nonlocal processing
            if processing is None:
                if 'processing' in kwargs:
                    processing = kwargs['processing']
                else:
                    for arg in args:
                        if isinstance(arg, Processing):
                            processing = arg
                            break
            if processing:
                t0 = timeit.default_timer()
            result = func(*args, **kwargs)
            if processing:
                t1 = timeit.default_timer()
                processing.context.eta = (t1 - t0) * 1000
            return result
        return _inner
    if isinstance(arg, Processing):
        processing = arg
        return decorator
    elif isinstance(arg, Callable):
        return decorator(arg)


def process_image(processing: Processing):
    t0 = timeit.default_timer()
    try:
        if not segment_process(processing):
            raise RuntimeError("Segment process failed")
        processing.output = processing.input.copy()[(processing.mask_box and processing.mask_box.image_slice)
                                                    or (slice(None), slice(None), slice(None))]
        # mask = np.dstack((processing.mask, processing.mask, processing.mask))
        # print(processing.state)
        return processing.output
    except Exception as e:
        # print(processing.state)
        processing.state.focused = False
        processing.state.steps += 1
        # print(type(e), e, f"{processing.state.steps} steps")
        now_mask = np.dstack((processing.mask, processing.mask, processing.mask)) // 2
        processing.exception = ExceptionReport(e, format_exc())
        return np.vstack((processing.focus, now_mask))
    finally:
        t1 = timeit.default_timer()
        processing.context.eta = (t1 - t0) * 1000


if __name__ == "__main__":


        
    def setup_plot():
        fig, ax = plt.subplots()
        im = ax.imshow(np.zeros((480, 640, 3), dtype=np.uint8))
        ax.axis('off')
        return fig, ax, im


    def update_processing(frame_num, processing: Processing, cap: cv2.VideoCapture, im):
        _, processing.input = cap.read()
        output = process_image(processing)
        stamp(output, f"mid_hue in {processing.context.mid_hue}", corner="top", margin=5)
        if processing.state.focused:
            stamp(output, f"mask_box in {processing.mask_box}", corner="mid", margin=5)
        else:
            stamp(output, f"ERROR: {processing.exception.exception}", corner="mid", margin=5)
        stamp(output, f"processed in {processing.context.eta:1.5f} ms", corner="bottom", margin=5)
        im.set_array(output)
        # plt.pause(.3)
        return im,


    def video_main():
        fig, ax, im = setup_plot()
        image = np.random.randint(0, 255, size=400*300*3, dtype=np.uint8).reshape((400, 300, -1))
        pr = Processing(focus_box=Box(0, 100, None, 100),
                        context=Context(mask_area_ratio_validation_threshold=.1,
                                        masked_area_ratio_validation_threshold=.15,
                                        reset_steps=10))

        available_cameras = []
        for i in range(20):  # Check up to 10 potential camera indices
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    # print(f"Camera {i} - {cap.getBackendName()} is available")
                    available_cameras.append(i)
                    cap.release()
            except:
                pass
        cap = cv2.VideoCapture(available_cameras[0])

        ani = FuncAnimation(fig, update_processing, fargs=(pr, cap, im),
                            interval=50, blit=True)
        
        plt.show()


    video_main()
