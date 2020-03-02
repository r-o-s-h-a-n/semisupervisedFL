import math
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import random_ops


def _assert_float_dtype(dtype):
  """Validate and return floating point type based on `dtype`.
  `dtype` must be a floating point type.
  Args:
    dtype: The data type to validate.
  Returns:
    Validated type.
  Raises:
    ValueError: if `dtype` is not a floating point type.
  """
  if not dtype.is_floating:
    raise ValueError("Expected floating point type, got %s." % dtype)
  return dtype


class ConvInitializer(tf.keras.initializers.Initializer):
    '''
    Conv weight initializer used by Rotation Net paper.
    '''
    def __init__(self, 
                kernel_size,
                out_channels,
                seed=None, 
                dtype=dtypes.float32):
        assert (isinstance(kernel_size, int) or (isinstance(kernel_size, tuple) and len(kernel_size)==2)), 'kernel size must be int or tuple'

        super(ConvInitializer, self).__init__()
        self.dtype = _assert_float_dtype(dtypes.as_dtype(dtype))
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.seed = seed

    def __call__(self, shape, dtype=None):
        if dtype is None:
            dtype = self.dtype
        
        if isinstance(self.kernel_size, int):
            n = self.kernel_size * self.kernel_size * self.out_channels
        else:
            n = self.kernel_size[0] * self.kernel_size[1] * self.out_channels
        stddev = math.sqrt(2/n)
        return random_ops.random_normal(shape, 0.0, stddev, dtype, seed=self.seed)

    def get_config(self):
        return {
            "kernel_size": self.kernel_size,
            "out_channels": self.out_channels,
            "seed": self.seed,
            "dtype": self.dtype.name
        }


class DenseInitializer(tf.keras.initializers.Initializer):
    '''
    Dense weight initializer used by Rotation Net paper.
    '''
    def __init__(self, 
                out_channels,
                seed=None, 
                dtype=dtypes.float32):

        super(DenseInitializer, self).__init__()
        self.dtype = _assert_float_dtype(dtypes.as_dtype(dtype))
        self.out_channels = out_channels
        self.seed = seed

    def __call__(self, shape, dtype=None):
        if dtype is None:
            dtype = self.dtype
        
        stddev = math.sqrt(2.0 / self.out_channels)
        return random_ops.random_normal(shape, 0.0, stddev, dtype, seed=self.seed)

    def get_config(self):
        return {
            "out_channels": self.out_channels,
            "seed": self.seed,
            "dtype": self.dtype.name
        }
