from unittest.mock import DEFAULT

import numpy as np
from decouple import config

HUE_DELTA = config("HUE_DELTA", default="10,255,100", cast=lambda v: np.array(v.split(","), dtype=int))
DEFAULT_VIDEO_BUFFER_SIZE = config("DEFAULT_VIDEO_BUFFER_SIZE", default=5, cast=int)
DEFAULT_FPS = config("DEFAULT_FPS", default=5, cast=int)
DEFAULT_FRAME_WIDTH = config("DEFAULT_FRAME_WIDTH", default=400, cast=int)
DEFAULT_FRAME_HEIGHT = config("DEFAULT_FRAME_HEIGHT", default=300, cast=int)

def int_or_none(x: str):
    return int(x) if x != "" else None

def split_to_ints(v: str):
    return map(int_or_none, v.split(","))

# BUTTLER SETTINGS
BUTTLER_FOCUS_BOX = config("BUTTLER_FOCUS_BOX", default="0,0,,", cast=split_to_ints)
BUTTLER_HUE_FOCUS = config("BUTTLER_HUE_FOCUS", default=",", cast=split_to_ints)

# BAND SETTINGS
BAND_FOCUS_BOX = config("BAND_FOCUS_BOX", default="0,0,,", cast=split_to_ints)
BAND_HUE_FOCUS = config("BAND_HUE_FOCUS", default=",", cast=split_to_ints)
