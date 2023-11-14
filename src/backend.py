import numpy as np
from functools import wraps

class Backend:
    pi = np.pi

    def __repr__(self):
        return self.__class__.__name__

numpy_float_dtypes = {
    getattr(np, "float_", np.float64),
    getattr(np, "float16", np.float64),
    getattr(np, "float32", np.float64),
    getattr(np, "float64", np.float64),
    getattr(np, "float128", np.float64),
}

def _replace_float(func):

    @wraps(func)
    def new_func(self, *args, **kwargs):
        result = func(*args, **kwargs)
        if result.dtype in numpy_float_dtypes:
            result = np.asarray(result, dtype=self.float)
        return result
    
    return new_func

class NumpyBackend(Backend):
    int = np.int64

    float = np.float64
    
    complex = np.complex128

    asarray = _replace_float(np.asarray)

    exp = staticmethod(np.exp)

    sin = staticmethod(np.sin)

    cos = staticmethod(np.cos)

    sum = staticmethod(np.sum)

    max = staticmethod(np.max)

    stack = staticmethod(np.stack)

    transpose = staticmethod(np.transpose)

    reshape = staticmethod(np.reshape)

    squeeze = staticmethod(np.squeeze)

    broadcast_arrays = staticmethod(np.broadcast_arrays)

    broadcast_to = staticmethod(np.broadcast_to)

    @staticmethod
    def bmm(arr1, arr2):
        return np.einsum("ijk,ikl->ijl", arr1, arr2)

    @staticmethod
    def is_array(arr):
        return isinstance(arr, np.ndarray)

    array = _replace_float(np.array)

    ones = _replace_float(np.ones)

    zeros = _replace_float(np.zeros)

    zeros_like = staticmethod(np.zeros_like)

    linspace = _replace_float(np.linspace)

    arange = _replace_float(np.arange)

    pad = staticmethod(np.pad)

    fftfreq = staticmethod(np.fft.fftfreq)

    fft = staticmethod(np.fft.fft)

    exp = staticmethod(np.exp)

    divide = staticmethod(np.divide)

    np = _replace_float(np.asarray)

backend = NumpyBackend()