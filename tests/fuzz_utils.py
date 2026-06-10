import numpy as np


def generate_random_arrays(key, shapes, dtypes):
    """Utility for fuzz testing: generate random arrays of shapes and dtypes."""
    np.random.seed(key)
    arrays = []
    for shape, dtype in zip(shapes, dtypes):
        if dtype in (np.float32, np.float64):
            arrays.append(np.random.randn(*shape).astype(dtype))
        elif dtype in (np.int32, np.int64):
            arrays.append(np.random.randint(-100, 100, size=shape).astype(dtype))
        elif dtype is bool:
            arrays.append(np.random.choice([True, False], size=shape))
        else:
            raise ValueError(f"Unsupported dtype for fuzzing: {dtype}")
    return arrays
