import keras.backend as K
from keras.layers import Input, Dense, Dropout, Concatenate, BatchNormalization, GlobalMaxPooling1D, Lambda, RepeatVector
from keras.layers import Activation
from keras.engine import Model
import tensorflow as tf
import numpy as np


class ConvNet(object):
    def __init__(self, x, pointclouds, config):
        self.ConvLayer = config['conv_layer']
        self.ConvElements = config['conv_elms']
        self.conv_elms_params = config['conv_elms_params']
        self.PoolingLayer = config['pool_layer']
        self.with_bn = config['with_bn']
        self.bn_momentum = config['bn_momentum']
        self.conv_params = config['conv_params']

        if 'pool_ratio' in config:
            self.pool_size = config['pool_ratio']
        elif 'pool_output_size' in config:
            self.pool_size = config['pool_size']

        self.num_stacks = config['num_conv_stacks']
        self.num_blocks_per_stack = config['num_blocks_per_conv_stack']
        self.pool_mode = config['pool_mode']

    # def __call__(self, x, pointcloud, *args, **kwargs):

        input_shape = K.int_shape(x)
        pc_shape = K.int_shape(pointclouds)
        num_batches = input_shape[0]
        num_points = input_shape[1]

        network = x

        self.conv_elms = []
        self.conv_stacks = []
        self.pc_hierarchy = []



        for stack_index in range(self.num_stacks):
            conv_elms = self.ConvElements(pointclouds,  self.conv_elms_params[stack_index])
            self.conv_elms.append(conv_elms)
            l = []
            conv_params = self.conv_params[stack_index]
            for layer_index in range(self.num_blocks_per_stack[stack_index]):
                conv_layer = self.ConvLayer(conv_params, conv_elms)
                network = conv_layer.get_layer(network, with_bn=self.with_bn, bn_decay=self.bn_momentum)

            self.conv_stacks.append(network)
            self.pc_hierarchy.append(pointclouds)
            if stack_index < len(self.pool_size):
                if not isinstance(network, list):
                    network = [network]
                to_pool = network + [pointclouds]

                if 'pool_ratio' in config:
                    pooled = self.PoolingLayer(ratio=self.pool_size[stack_index],
                                        pool_mode=self.pool_mode[stack_index])(to_pool)
                else:
                    pooled = self.PoolingLayer(output_size=self.pool_size[stack_index],
                                               pool_mode=self.pool_mode[stack_index])(to_pool)

                network = pooled[:-1]
                pointclouds = pooled[-1]

        # self.pc_hierarchy.append(pointclouds)

        self.output = [network, pointclouds]

    def get_output(self):
        return self.output

    def get_conv_stacks(self):
        return self.conv_stacks

    def get_pc_hierarchy(self):
        return self.pc_hierarchy

    def get_conv_elms(self):
        return self.conv_elms


class DeconvNet(object):
    def __init__(self, conv_net, deconv_layer,
                 num_deconv_stacks=None,
                 num_blocks_per_stack=None,
                 out_channels=None):
        self.DeconvLayer = deconv_layer
        self.ConvLayer = conv_net.ConvLayer
        self.pointclouds_down = conv_net.pc_hierarchy
        self.conv_stacks = conv_net.conv_stacks
        self.conv_elms = conv_net.conv_elms
        self.conv_params = conv_net.conv_params
        self.with_bn = conv_net.with_bn
        self.bn_momentum = conv_net.bn_momentum


        self.num_conv_stacks = len(self.conv_stacks)
        if num_deconv_stacks is None:
            self.num_deconv_stacks = self.num_conv_stacks - 1
        else:
            assert(num_deconv_stacks <= self.num_conv_stacks - 1)
            self.num_deconv_stacks = num_deconv_stacks


        if num_blocks_per_stack is None:
            self.num_blocks_per_stack = conv_net.num_blocks_per_stack
        else:
            self.num_blocks_per_stack = num_blocks_per_stack


        self.deconv_stacks = []

        network = conv_net.get_output()[0]


        self.j = -1
        for i in range(self.num_deconv_stacks):
            j = self.num_conv_stacks - i - 2
            self.j = j
            self.deconv_stacks.append(network)

            if not isinstance(network, list):
                network = [network]
            deconvnet = self.DeconvLayer(pc1=self.pointclouds_down[j+1], pc2=self.pointclouds_down[j])(network)

            conv_elms = self.conv_elms[j]

            conv_layer = self.ConvLayer(self.conv_params[j], conv_elms)
            convnet = conv_layer.get_layer(deconvnet, with_bn=self.with_bn, bn_decay=self.bn_momentum)

            if isinstance(convnet, list):
                assert(isinstance(self.conv_stacks[j], list))
                assert(len(self.conv_stacks[j]) == len(convnet))
                network = []
                for i in range(len(convnet)):
                    network.append(Concatenate(axis=-1)([self.conv_stacks[j][i], convnet[i]]))
            else:
                network = Concatenate(axis=-1)([self.conv_stacks[j], convnet])

        self.deconv_stacks.append(network)

        self.output = network



        """
            if (drop_out > 0. and i <= 2):
                network = Dropout(rate=drop_out)(network)

        block_elm = self.ConvElements(pointclouds_pl_down[0][0], self.conf)

        network = self.ConvLayer(part_num, block_elm).get_layer(network, with_bn=False, with_relu=False)

        network = Activation('softmax')(network)
        """

    def get_output(self):
        return self.output

    def get_deconv_stacks(self):
        return self.deconv_stacks

    def get_conv_elms(self):
        return self.conv_elms

    def get_conv_params(self):
        return self.conv_params

    def get_last_idx(self):
        return self.j


class DeconvNetBis(object):
    def __init__(self, conv_net, deconv_layer,
                 num_deconv_stacks=None,
                 num_blocks_per_stack=None,
                 out_channels=None):
        self.DeconvLayer = deconv_layer
        self.PoolingLayer = conv_net.PoolingLayer
        self.ConvLayer = conv_net.ConvLayer
        self.pointclouds_down = conv_net.pc_hierarchy
        self.conv_stacks = conv_net.conv_stacks
        self.conv_elms = conv_net.conv_elms
        self.conv_params = conv_net.conv_params
        self.with_bn = conv_net.with_bn
        self.bn_momentum = conv_net.bn_momentum


        self.num_conv_stacks = len(self.conv_stacks)
        if num_deconv_stacks is None:
            self.num_deconv_stacks = self.num_conv_stacks - 1
        else:
            assert(num_deconv_stacks <= self.num_conv_stacks - 1)
            self.num_deconv_stacks = num_deconv_stacks


        if num_blocks_per_stack is None:
            self.num_blocks_per_stack = conv_net.num_blocks_per_stack
        else:
            self.num_blocks_per_stack = num_blocks_per_stack


        self.deconv_stacks = []

        network = conv_net.get_output()[0]


        if isinstance(network, list):
            network = network[0]
        pooled_network = GlobalMaxPooling1D()(network)
        pooled_network = RepeatVector(n=K.int_shape(network)[1])(pooled_network)
        network = Concatenate(axis=-1)([pooled_network, network])

        network = Dense(units=K.int_shape(network)[-1]//2, activation=None)(network)
        if self.with_bn:
            network = BatchNormalization(momentum=self.bn_momentum)(network)
        network = Activation('relu')(network)

        self.deconv_stacks.append(network)




        self.PoolingLayer()
        self.j = -1
        for i in range(self.num_deconv_stacks):
            j = self.num_conv_stacks - i - 1

            # self.deconv_stacks.append(network)

            conv_elms = self.conv_elms[j]

            out_channels = self.conv_params[j-1]['out_channels']
            conv_layer = self.ConvLayer(self.conv_params[j], conv_elms, out_channels=out_channels)

            network = conv_layer.get_layer(network, with_bn=self.with_bn, bn_decay=self.bn_momentum)

            if not isinstance(network, list):
                network = [network]
            network = self.DeconvLayer(pc1=self.pointclouds_down[j], pc2=self.pointclouds_down[j-1])(network)

            j -= 1
            self.j = j

            if isinstance(network, list):
                assert(isinstance(self.conv_stacks[j], list))
                assert(len(self.conv_stacks[j]) == len(network))
                L = []
                for i in range(len(network)):
                    c = Concatenate(axis=-1)([self.conv_stacks[j][i], network[i]])
                    c = Dense(units=K.int_shape(network[i])[-1], activation=None)(c)
                    if self.with_bn:
                        c = BatchNormalization(momentum=self.bn_momentum)(c)
                    c = Activation('relu')(c)
                    L.append(c)
                network = L
            else:
                nc = K.int_shape(network)[-1]

                network = Concatenate(axis=-1)([self.conv_stacks[j], network])
                network = Dense(units=nc, activation=None)(network)
                if self.with_bn:
                    network = BatchNormalization(momentum=self.bn_momentum)(network)
                network = Activation('relu')(network)
            self.deconv_stacks.append(network)



        self.output = network


    def get_output(self):
        return self.output

    def get_deconv_stacks(self):
        return self.deconv_stacks

    def get_conv_elms(self):
        return self.conv_elms

    def get_conv_params(self):
        return self.conv_params

    def get_last_idx(self):
        return self.j