from keras import backend as K
from keras.layers import Layer
import tensorflow as tf

class ConvKernel(Layer):

    def __init__(self, kernel_fn, **kwargs):
        self.kernel_fn = kernel_fn
        self.kernel_shape = kernel_fn.get_shape()
        super(ConvKernel, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        super(ConvKernel, self).build(input_shape)

    def call(self, x):
        assert isinstance(x, list)
        # conv_kernel = self.kernel_fn.compute(x[0], x[1])
        conv_kernel = self.kernel_fn.compute(*x)
        # conv_kernel /= float(x[0].get_shape()[2].value)
        return conv_kernel


    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        nb = input_shape[0][0]
        nv = input_shape[0][1]
        np = input_shape[0][2]

        if isinstance(self.kernel_shape, list):
            return [(nb, nv, np,) + self.kernel_shape[i] for i in range(len(self.kernel_shape))]
        else:
            return (nb, nv, np,) + self.kernel_shape