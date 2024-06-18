import os

import numpy as np


def load_sample_height_data():
    path = os.path.join(os.path.dirname(__file__), "height.txt")
    return np.loadtxt(path)
