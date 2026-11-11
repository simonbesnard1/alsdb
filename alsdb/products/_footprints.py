"""Point arrays for footprint products."""

import numpy as np


def point_array(data):
    """Accept query dictionaries and structured point arrays without repeated copies."""
    if isinstance(data, np.ndarray):
        return data
    names = [name for name in data if name != "Year"]
    arr = np.empty(len(data[names[0]]), dtype=[(name, data[name].dtype) for name in names])
    for name in names:
        arr[name] = data[name]
    return arr
