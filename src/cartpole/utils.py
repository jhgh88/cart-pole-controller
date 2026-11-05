import numpy as np
from numpy import cos, sin

def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi