import numpy as np


def to_array(x):
    if hasattr(x, "numpy"):
        return x.numpy()
    if hasattr(x, "tolist"):
        return np.array(x.tolist())
    return np.array(x)


def zeros(shape, dtype=float):
    return np.zeros(shape, dtype=dtype)
