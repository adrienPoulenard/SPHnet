from keras import backend as K
from keras.layers import Layer
import tensorflow as tf
import numpy as np
from utils.pointclouds_utils import is_power2

def kd_tree_upsample(x, k):
    nb = K.int_shape(x)[0]
    nv = K.int_shape(x)[1]
    nc = K.int_shape(x)[-1]

    n = K.ndim(x)
    shape = list(K.int_shape(x))
    x = tf.expand_dims(x, axis=2)
    tile = (1, 1, 2**k) + tuple([1]*(n-2))
    x = tf.tile(x, tile)

    new_shape = tuple([shape[0], (2**k)*shape[1]] + shape[2:])

    """
    x = K.expand_dims(x, axis=2)
    x = K.repeat_elements(x, rep=2**k, axis=2)
    x = K.permute_dimensions(x, pattern=(0, 2, 1, 3))
    """

    # x = tf.transpose(x, perm=(0, 2, 1, 3))
    return tf.reshape(x, shape=new_shape)


class KdTreeDeconv(Layer):

    def __init__(self, nv1=None, nv2=None, pc1=None, pc2=None, k=0, **kwargs):
        if (nv1 is not None) and (nv2 is not None):
            assert (is_power2(nv1))
            assert (is_power2(nv2))
            self.k = int(np.ceil(np.log(nv2 / nv1) / np.log(2.)))
        elif (pc1 is not None) and (pc2 is not None):
            nv1 = pc1.get_shape()[1].value
            nv2 = pc2.get_shape()[1].value
            assert (is_power2(nv1))
            assert (is_power2(nv2))
            self.k = int(np.ceil(np.log(float(nv2)/float(nv1)) / np.log(2.)))
        else:
            self.k = k

        super(KdTreeDeconv, self).__init__(**kwargs)

    def build(self, input_shape):
        # assert isinstance(input_shape, list)
        super(KdTreeDeconv, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        # assert isinstance(x, list)
        if isinstance(x, list):
            y = []
            for i in range(len(x)):
                yi = kd_tree_upsample(x[i], self.k)

                y.append(yi)
        else:
            y = kd_tree_upsample(x, self.k)
        return y

    def compute_output_shape(self, input_shape):
        # assert isinstance(input_shape, list)

        if isinstance(input_shape, list):
            output_shapes = []
            for i in range(len(input_shape)):
                shape = list(input_shape[i])
                new_shape = [shape[0], shape[1]*(2**self.k)] + shape[2:]

                output_shapes.append(tuple(new_shape))
            return output_shapes
        else:
            shape = list(input_shape)
            new_shape = [shape[0], shape[1] * (2 ** self.k)] + shape[2:]
        return tuple(new_shape)
