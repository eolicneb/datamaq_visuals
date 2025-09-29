import numpy as np
from decouple import config

HUE_DELTA = config("HUE_DELTA", default="10,255,100", cast=lambda v: np.array(v.split(","), dtype=int))
