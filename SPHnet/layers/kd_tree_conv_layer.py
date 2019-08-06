import tensorflow as tf
import numpy as np
from keras.layers import Layer, BatchNormalization, Activation, AveragePooling1D, Dropout
from utils.patches_builder import BuildPatches
from utils.conv_kernel import ConvKernel
import keras.backend as K
from keras import regularizers, initializers, activations
from utils.sh_conv import ShKernel, tf_gaussian, tf_hat, sh_invar_conv, sh_norm, sh_eqvar_conv_1
from layers.kd_tree_pooling import pool_input_kd
from layers.kd_tree_deconv import kd_tree_upsample

def get_b_tree_conv_elems_params(conf, stack_idx):
    params = dict()
    i = stack_idx
    params['conv_max_pool'] = conf['conv_max_pool'][i]
    params['kernel_radius'] = conf['kernel_radius'][i]
    params['patch_size'] = conf['patch_size'][i]
    params['l_max'] = conf['l_max'][i]
    params['nr'] = conf['nr'][i]
    params['radial_fn'] = conf['radial_fn']
    if conf['radial_fn'] == 'hat':
        params['radial_fn'] = tf_hat
    else:
        params['radial_fn'] = tf_gaussian
    params['normalize_patches'] = conf['normalize_patches']
    params['radial_spacing'] = conf['radial_spacing'][i]
    params['tree_spacing'] = conf['tree_spacing'][i]
    params['strides'] = conf['strides'][i]
    params['keep_num_points'] = conf['keep_num_points'][i]

    return params

class BinaryTreeConvElements:
    def __init__(self, points_pl, params, roots=None):
        if 'conv_max_pool' in params:
            self.conv_max_pool = params['conv_max_pool']
        else:
            self.conv_max_pool = 0

        if 'normalize_patches' in params:
            self.normalize_patches = params['normalize_patches']
        else:
            self.normalize_patches = False

        if 'kernel_radius' in params:
            self.radius = params['kernel_radius']
        else:
            self.radius = 1.
            self.normalize_patches = True


        self.patch_size = params['patch_size']
        self.l_max = params['l_max']
        self.nr = params['nr']

        if 'radial_fn' in params:
            self.radial_fn = params['radial_fn']
        else:
            self.radial_fn = tf_gaussian

        """  
        self.radial_fn = params['radial_fn']
        if params['radial_fn'] == 'hat':
            self.radial_fn = tf_hat
        else:
            self.radial_fn = tf_gaussian
        """

        self.points_pl = points_pl
        self.batch_size = K.int_shape(self.points_pl)[0]
        self.num_of_points = K.int_shape(self.points_pl)[1]

        if 'radial_spacing' in params:
            self.radial_spacing = params['radial_spacing']
        else:
            self.radial_spacing = 0
        if 'tree_spacing' in params:
            self.tree_spacing = params['tree_spacing']
        else:
            self.tree_spacing = 0
        if 'strides' in params:
            self.strides = params['strides']
        else:
            self.strides = 0
        if 'keep_num_points' in params:
            self.keep_num_points = params['keep_num_points']
        else:
            self.keep_num_points = True

        if roots is not None:
            self.roots = roots
        else:
            self.roots = points_pl

        if self.strides > 0:
            self.roots = AveragePooling1D(pool_size=2**self.strides)(self.roots)
        self.points = points_pl
        if self.tree_spacing > 0:
            self.points = AveragePooling1D(pool_size=2**self.tree_spacing)(self.points)

        # assert (self.num_of_points >= self.patch_size)

        self.patch_size = min(self.num_of_points, self.patch_size)
        P = BuildPatches(self.patch_size)([self.points, self.roots])
        self.patches = P[0]
        self.patches_idx = P[1]
        self.patches_sq_dist = P[2]
        self.sq_dist_mat = P[3]

        self.kernel_fn = ShKernel(self.nr, self.l_max, self.radius, self.radial_fn, radial_first=True,
                                  normalize_patch=self.normalize_patches, return_sh=True)

        self.conv_kernel, self.sh = ConvKernel(kernel_fn=self.kernel_fn)([self.patches, self.patches_sq_dist])



class BinaryTreeShInvariantConv(Layer):

    def __init__(self, out_channels, l_max, nr,
                 strides=0,
                 tree_spacing=0,
                 keep_num_points=True,
                 max_pool = 0,
                 initializer='glorot_uniform',
                 l2_regularizer=1.0e-3,
                 with_relu=True, **kwargs):
        self.out_channels = out_channels
        self.l_max = l_max
        self.nr = nr
        self.strides = strides
        self.tree_spacing = tree_spacing
        self.keep_num_points = keep_num_points
        self.initializer = initializer
        self.l2_regularizer = l2_regularizer
        self.with_relu = with_relu
        self.max_pool = max_pool

        super(BinaryTreeShInvariantConv, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        # Create a trainable weight variable for this layer.
        # shape = (self.out_channels, self.l_max+1, self.nr, input_shape[0][-1])
        shape = (self.out_channels, input_shape[0][-1], self.nr, self.l_max + 1)

        self.kernel_weights = self.add_weight(name='kernel',
                                              shape=shape,
                                              initializer=self.initializer,
                                              regularizer=regularizers.l2(self.l2_regularizer),
                                              trainable=self.trainable)

        self.biases = self.add_weight(name='bias',
                                      shape=(self.out_channels,),
                                      initializer=initializers.get('zeros'),
                                      regularizer=regularizers.get(None),
                                      trainable=self.trainable)

        super(BinaryTreeShInvariantConv, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        assert isinstance(x, list)

        signal = x[0]
        if self.tree_spacing > 0:
            signal = pool_input_kd(x[0], self.tree_spacing, pool_mode='max')

        patches_idx = x[1]
        conv_kernel = x[2]

        # y = sh_invar_conv(signal, x[1], x[2], self.kernel_weights, self.l_max)

        patches = tf.gather_nd(signal, patches_idx)

        y = tf.einsum('bvprn,bvpc->bvcrn', conv_kernel, patches)

        y = tf.multiply(y, y)
        L = []
        p = 0
        for l in range(0, self.l_max + 1):
            x = y[:, :, :, :, p:(p + 2 * l + 1)]
            x = tf.reduce_sum(x, axis=-1, keepdims=False)
            L.append(x)
            p += 2 * l + 1

        y = tf.stack(L, axis=-1)
        y = tf.sqrt(tf.maximum(y, 0.0001))
        y = tf.einsum('ijrn,bvjrn->bvi', self.kernel_weights, y)

        # K.bias_add(y, self.biases)
        y = tf.nn.bias_add(y, self.biases)

        if self.with_relu:
            y = activations.relu(y)

        if self.max_pool > 0:
            y = pool_input_kd(y, self.max_pool, pool_mode='max')

        if self.keep_num_points and self.strides > 0:
            y = kd_tree_upsample(y, self.strides + self.max_pool)

        return y

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        if self.keep_num_points:
            return (input_shape[0][0], input_shape[0][1], self.out_channels)
        else:
            new_nv = int(input_shape[0][1] / 2**(self.strides + self.max_pool))
            return (input_shape[0][0], new_nv, self.out_channels)

def b_tree_sh_inv_conv_params(config, stack_idx, layer_idx):
    params = dict()

    i = stack_idx
    params['out_channels'] = config['out_channels'][i][layer_idx]
    params['initializer'] = config['initializer']
    params['l2_regularizer'] = config['l2_regularizer']
    params['l_max'] = config['l_max'][i]
    params['nr'] = config['nr'][i]
    params['strides'] = config['strides'][i]
    params['tree_spacing'] = config['tree_spacing'][i]
    params['keep_num_points'] = config['keep_num_points'][i]
    params['conv_max_pool'] = config['conv_max_pool'][i]

    return params


class BinaryTreeShInvariantConvLayer:
    def __init__(self, params, conv_elms, out_channels=None):

        if out_channels is not None:
            self.out_channels = out_channels
        else:
            self.out_channels = params['out_channels']

        if 'initializer' in params:
            self.initializer = params['initializer']
        else:
            self.initializer = 'glorot_uniform'

        if 'l2_regularizer' in params:
            self.l2_regularizer = params['l2_regularizer']
        else:
            self.l2_regularizer = 0.0



        """
        self.l_max = params['l_max']
        self.nr = params['nr']
        self.strides = params['strides']
        self.tree_spacing = params['tree_spacing']
        self.keep_num_points = params['keep_num_points']
        self.conv_max_pool = params['conv_max_pool']
        """

        self.l_max = conv_elms.l_max
        self.nr = conv_elms.nr
        self.strides = conv_elms.strides
        self.tree_spacing = conv_elms.tree_spacing
        self.keep_num_points = conv_elms.keep_num_points
        self.conv_max_pool = conv_elms.conv_max_pool

        self.patches_idx = conv_elms.patches_idx
        self.conv_kernel = conv_elms.conv_kernel

    def get_layer(self, x, with_relu=True, with_bn=True, bn_decay=0.9):

        if isinstance(x, list):
            assert(len(x) == 1)
            x = x[0]

        y = BinaryTreeShInvariantConv(out_channels=self.out_channels,
                            l_max=self.l_max,
                            nr=self.nr,
                            strides=self.strides,
                            tree_spacing=self.tree_spacing,
                            keep_num_points=self.keep_num_points,
                            max_pool=self.conv_max_pool,
                            initializer=self.initializer,
                            l2_regularizer=self.l2_regularizer,
                            with_relu=False)([x, self.patches_idx, self.conv_kernel])

        if with_bn:
            y = BatchNormalization(momentum=bn_decay)(y)

        if with_relu:
            y = Activation('relu')(y)

        return y


class BinaryTreeShConv(Layer):

    def __init__(self, out_channels, l_max, nr,
                 strides=0,
                 tree_spacing=0,
                 keep_num_points=True,
                 max_pool = 0,
                 initializer='glorot_uniform',
                 l2_regularizer=1.0e-3,
                 with_relu=True, **kwargs):
        self.out_channels = out_channels
        self.l_max = l_max
        self.nr = nr
        self.strides = strides
        self.tree_spacing = tree_spacing
        self.keep_num_points = keep_num_points
        self.initializer = initializer
        self.l2_regularizer = l2_regularizer
        self.with_relu = with_relu
        self.max_pool = max_pool

        super(BinaryTreeShConv, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        # Create a trainable weight variable for this layer.
        # shape = (self.out_channels, self.l_max+1, self.nr, input_shape[0][-1])
        shape = (self.out_channels, input_shape[0][-1], self.nr, (self.l_max + 1)**2)

        self.kernel_weights = self.add_weight(name='kernel',
                                              shape=shape,
                                              initializer=self.initializer,
                                              regularizer=regularizers.l2(self.l2_regularizer),
                                              trainable=self.trainable)

        self.biases = self.add_weight(name='bias',
                                      shape=(self.out_channels,),
                                      initializer=initializers.get('zeros'),
                                      regularizer=regularizers.get(None),
                                      trainable=self.trainable)

        super(BinaryTreeShConv, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        assert isinstance(x, list)

        signal = x[0]
        if self.tree_spacing > 0:
            signal = pool_input_kd(x[0], self.tree_spacing, pool_mode='max')

        patches_idx = x[1]
        conv_kernel = x[2]

        # y = sh_invar_conv(signal, x[1], x[2], self.kernel_weights, self.l_max)

        patches = tf.gather_nd(signal, patches_idx)

        y = tf.einsum('bvprn,bvpc->bvcrn', conv_kernel, patches)



        y = tf.einsum('ijrn,bvjrn->bvi', self.kernel_weights, y)

        # K.bias_add(y, self.biases)
        y = tf.nn.bias_add(y, self.biases)

        if self.with_relu:
            y = activations.relu(y)

        if self.max_pool > 0:
            y = pool_input_kd(y, self.max_pool, pool_mode='max')

        if self.keep_num_points and self.strides > 0:
            y = kd_tree_upsample(y, self.strides + self.max_pool)

        return y

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        if self.keep_num_points:
            return (input_shape[0][0], input_shape[0][1], self.out_channels)
        else:
            new_nv = int(input_shape[0][1] / 2**self.strides)
            return (input_shape[0][0], new_nv, self.out_channels)


class BinaryTreeShConvLayer:
    def __init__(self, params, conv_elms, out_channels=None):
        if out_channels is not None:
            self.out_channels = out_channels
        else:
            self.out_channels = params['out_channels']

        if 'initializer' in params:
            self.initializer = params['initializer']
        else:
            self.initializer = 'glorot_uniform'

        if 'l2_regularizer' in params:
            self.l2_regularizer = params['l2_regularizer']
        else:
            self.l2_regularizer = 0.0



        """
        self.l_max = params['l_max']
        self.nr = params['nr']
        self.strides = params['strides']
        self.tree_spacing = params['tree_spacing']
        self.keep_num_points = params['keep_num_points']
        self.conv_max_pool = params['conv_max_pool']
        """

        self.l_max = conv_elms.l_max
        self.nr = conv_elms.nr
        self.strides = conv_elms.strides
        self.tree_spacing = conv_elms.tree_spacing
        self.keep_num_points = conv_elms.keep_num_points
        self.conv_max_pool = conv_elms.conv_max_pool

        self.patches_idx = conv_elms.patches_idx
        self.conv_kernel = conv_elms.conv_kernel

    def get_layer(self, x, with_relu=True, with_bn=True, bn_decay=0.9):

        if isinstance(x, list):
            assert(len(x) == 1)
            x = x[0]

        y = BinaryTreeShConv(out_channels=self.out_channels,
                            l_max=self.l_max,
                            nr=self.nr,
                            strides=self.strides,
                            tree_spacing=self.tree_spacing,
                            keep_num_points=self.keep_num_points,
                            max_pool=self.conv_max_pool,
                            initializer=self.initializer,
                            l2_regularizer=self.l2_regularizer,
                            with_relu=False)([x, self.patches_idx, self.conv_kernel])

        if with_bn:
            y = BatchNormalization(momentum=bn_decay)(y)

        if with_relu:
            y = Activation('relu')(y)

        return y


