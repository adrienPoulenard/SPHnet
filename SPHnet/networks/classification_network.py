import keras.backend as K
from keras.layers import Input, Dense, Dropout, Reshape, BatchNormalization, Activation, Flatten, Concatenate
from keras.engine import Model
import numpy as np
from networks.conv_net import ConvNet
from keras.layers import GlobalMaxPooling1D, GlobalMaxPooling2D
import sys

sys.path.append('../layers')

class ClassNetwork:
    def __init__(self, method):
        self.config = method['config']

        self.InputLayer = None
        if 'input_layer' in method:
            self.InputLayer = method['input_layer']

        self.PoolingLayer = method['pooling_layer']

        self.PostProcessLayer = None
        if 'postprocess_layer' in method:
            self.PostProcessLayer = method['postprocess_layer']

        self.with_bn = self.config['with_bn']
        self.bn_momentum = self.config['bn_momentum']

    def get_network_model(self, num_classes, batch_size, num_points, bn_decay=0.5):
        pc_input = Input(shape=(num_points, 3), batch_shape=(batch_size, num_points, 3))
        pointclouds_pl = pc_input

        additional_input = None
        if self.InputLayer is '3d':
            additional_input = Input(tensor=K.constant(1., 'float32', (batch_size, num_points, 1)))
            ps_function_pl = Concatenate(axis=-1)([additional_input, pointclouds_pl])

        elif self.InputLayer is not None:
            # ps_function_pl = self.InputLayer(self.config)(pointclouds_pl)
            ps_function_pl = self.InputLayer()(pointclouds_pl)
        else:
            additional_input = Input(tensor=K.constant(1., 'float32', (batch_size, num_points, 1)))
            ps_function_pl = additional_input



        # ps_function_pl = pc_input

        network = ps_function_pl

        network = ConvNet(network, pointclouds_pl, self.config).get_output()

        if self.PostProcessLayer is not None:
            network = self.PostProcessLayer(self.config)(network)
            network = [network]



        if isinstance(network, list):
            network = network[0]


        if K.ndim(network) == 3:
            network = GlobalMaxPooling1D()(network)
        elif K.ndim(network) == 4:
            network = GlobalMaxPooling2D()(network)
        else:
            raise ValueError('output dim must be 3 or 4')
        # network = Reshape((-1,))(network)
        # network = Flatten()(network)


        network = Dense(units=self.config['fc1_size'], activation=None)(network)
        # if with_bn:
        network = BatchNormalization(momentum=self.bn_momentum)(network)
        network = Activation('relu')(network)

        network = Dropout(rate=self.config['dropout_keep_prob'])(network)

        network = Dense(units=self.config['fc2_size'], activation=None)(network)
        # if with_bn:
        network = BatchNormalization(momentum=self.bn_momentum)(network)
        network = Activation('relu')(network)
        network = Dropout(rate=self.config['dropout_keep_prob'])(network)

        network = Dense(units=num_classes, activation='softmax')(network)



        if additional_input is not None:
            return Model(inputs=[pc_input, additional_input], outputs=network)
        else:
            return Model(inputs=[pc_input], outputs=network)

