import numpy as np
from layers.kd_tree_pooling import KdTreePooling
from layers.kd_tree_conv_layer import BinaryTreeShInvariantConvLayer, BinaryTreeConvElements
from networks.classification_network import ClassNetwork



kernel_radius = 1.0*np.array([0.4*0.25, 0.4*0.5, 0.4*1., 1.])

nr = 2
l_max = 3
normalize_patches = False
patch_size = 64
radial_spacing = 0
tree_spacing = 0
strides = [0, 0, 0]
out_channels = [64, 256, 1024]
num_stacks = len(strides)

# kernel_radius = [2./np.sqrt(1024), 2./np.sqrt(256), 2./np.sqrt(64)]
conv_elms_params = []
conv_params = []
for i in range(num_stacks):
    conv_elms_params.append({'kernel_radius': kernel_radius[i],
                     'patch_size': patch_size,
                     'l_max': l_max,
                     'nr': nr,
                     'normalize_patches': normalize_patches,
                     'radial_spacing': radial_spacing,
                     'tree_spacing': tree_spacing,
                     'strides': strides[i]})
    conv_params.append({'out_channels': out_channels[i]})

conv_layer = BinaryTreeShInvariantConvLayer
conv_elms = BinaryTreeConvElements
pooling_layer = KdTreePooling

btree_conv_inv_config = {'num_conv_stacks': num_stacks,
                        'num_blocks_per_conv_stack': [1, 1, 1],
                        'with_bn': True,
                        'bn_momentum': 0.5,
                        'conv_elms': conv_elms,
                        'conv_elms_params': conv_elms_params,
                        'conv_layer': conv_layer,
                        'conv_params': conv_params,
                        'pool_layer': pooling_layer,
                        'pool_ratio': [4, 4],
                        'pool_mode': [['max'], ['max']],
                        'fc1_size': 512,
                        'fc2_size': 256,
                        'dropout_keep_prob': 0.5,
                        'btree_idx_depth': 10}

name = 'btree_inv_conv'


btree_inv_conv_method = {'name': name,
                         'arch': ClassNetwork,
                         'config': btree_conv_inv_config,
                         'pooling_layer': KdTreePooling}

methods_list = [btree_inv_conv_method]
