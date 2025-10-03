import numpy as np


def cross(col=(0, 255, 255), bg=(63, 63, 63)):
    icon = np.zeros((13, 13, 3), dtype=np.uint8)
    icon[5:8,:,None] = bg
    icon[:,5:8,None] = bg
    icon[6,1:-1,None] = col
    icon[1:-1,6,None] = col
    icon[5:8,5:8,None] = (0, 0, 0)
    # icon[6,6,None] = col
    return icon
