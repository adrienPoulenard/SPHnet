from keras import backend as K
from keras.layers import Layer
import tensorflow as tf
from layers.kd_tree_deconv import kd_tree_upsample
import numpy as np

def pool1d(x, pool_size, strides=(1,), padding='valid', data_format='channels_last', pool_mode='max'):

    x = K.expand_dims(x, axis=2)  # add dummy last dimension
    x = K.pool2d(x=x,
                 pool_size=pool_size + (1,),
                 strides=strides + (1,),
                 padding=padding,
                 data_format=data_format,
                 pool_mode=pool_mode)
    return K.squeeze(x, 2)  # remove dummy last dimension





def pool_input_kd(inputs, k, pool_mode='max'):

    rank = len(list(K.int_shape(inputs)))

    k = 2**k

    assert 3 <= rank <= 5

    if rank == 3:
        pool_func = pool1d
        pool_size = (k,)
    if rank == 4:
        pool_func = K.pool2d
        pool_size = (k, 1)
    if rank == 5:
        pool_func = K.pool3d
        pool_size = (k, 1, 1)

    pooled_input = pool_func(x=inputs,
                             pool_size=pool_size,
                             strides=pool_size,
                             padding='valid',
                             data_format='channels_last',
                             pool_mode=pool_mode)
    return pooled_input


class KdTreePooling(Layer):

    def __init__(self, k=None, ratio=None, pool_mode=None, same=False, **kwargs):
        self.same = same

        self.k = 1


        if k is not None:
            self.k = k

        if ratio is not None:
            self.k = np.rint(np.log(ratio)/np.log(2.))

        if pool_mode is None:
            self.pool_mode = ['max']
        else:
            assert isinstance(pool_mode, list)
            self.pool_mode = pool_mode
        super(KdTreePooling, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        super(KdTreePooling, self).build(input_shape)

    def call(self, x):
        assert isinstance(x, list)
        assert(len(x) <= len(self.pool_mode)+1)
        y = []
        for i in range(len(self.pool_mode)):
            pooled = pool_input_kd(x[i], self.k, self.pool_mode[i])
            if self.same:
                pooled = kd_tree_upsample(pooled, self.k)
            y.append(pooled)

        if len(x) > len(self.pool_mode):
            if self.same:
                pooled_points_pl = x[-1]
            else:
                pooled_points_pl = pool1d(x=x[-1],
                                        pool_size=(2 ** self.k,),
                                        strides=(2 ** self.k,),
                                        padding='valid',
                                        data_format='channels_last',
                                        pool_mode='avg')

            y.append(pooled_points_pl)
        return y

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        output_shape = []
        for i in range(len(input_shape)):
            shape = list(input_shape[i])
            new_nv = int(input_shape[i][1] / (2**self.k))
            new_shape = [input_shape[i][0], new_nv] + shape[2:]
            output_shape.append(tuple(new_shape))
        return output_shape






