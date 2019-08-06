import keras.backend as K
from keras.layers import Input,Dropout, Concatenate
from keras.engine import Model
from keras.layers import Activation




class SegmentationNetwork:
    def __init__(self, method):
        self.config = method['config']

        if 'with_bn' in self.config:
            self.with_bn = self.config['with_bn']
            if self.with_bn:
                self.bn_momentum = self.config['bn_momentum']
        else:
            self.with_bn = False

        self.InputLayer = None
        if 'input_layer' in method:
            self.InputLayer = method['input_layer']

        # self.ConvLayer = method['conv_layer']
        # self.ConvElements = method['conv_elems']
        # self.PoolingLayer = method['pooling_layer']
        self.DeconvLayer = method['deconv_layer']
        self.PostProcessLayer = None
        if 'postprocess_layer' in method:
            self.PostProcessLayer = method['postprocess_layer']

        self.ConvLayer = self.config['conv_layer']
        self.ConvElements = self.config['conv_elms']
        self.conv_elms_params = self.config['conv_elms_params']
        self.PoolingLayer = self.config['pool_layer']
        self.with_bn = self.config['with_bn']
        self.bn_momentum = self.config['bn_momentum']
        self.conv_params = self.config['conv_params']

        if 'pool_ratio' in self.config:
            self.pool_size = self.config['pool_ratio']
        elif 'pool_output_size' in self.config:
            self.pool_size = self.config['pool_output_size']

        self.num_stacks = self.config['num_conv_stacks']
        self.num_blocks_per_stack = self.config['num_blocks_per_conv_stack']
        self.pool_mode = self.config['pool_mode']

        # def __call__(self, x, pointcloud, *args, **kwargs):

        self.conv_elms = []
        self.conv_stacks = []
        self.pc_hierarchy = []

    def get_network_model(self, part_num, batch_size, num_points, num_categories=1):

        # self.batch_size = batch_size
        # self.num_points = num_points
        # self.num_points = self.reference_shape.shape[0]

        # pointclouds

        pointcloud = Input(shape=(num_points, 3), batch_shape=(batch_size, num_points, 3))

        self.pointcloud = pointcloud

        if self.InputLayer is '3d':
            input_signal = Input(tensor=K.constant(1., 'float32', (batch_size, num_points, 1)))
            input_signal_ = Concatenate(axis=-1)([input_signal, pointcloud])
        elif self.InputLayer is not None:
            input_signal = self.InputLayer()(pointcloud)
            input_signal_ = input_signal
        else:
            # input_signal = Input(shape=(num_points, 2), batch_shape=(batch_size, num_points, 2))
            input_signal = Input(tensor=K.constant(1., 'float32', (batch_size, num_points, 1)))
            input_signal_ = input_signal

        one_hot_label = Input(shape=(num_categories,), batch_shape=(batch_size, num_categories))

        network = input_signal_

        self.pc_hierarchy.append(pointcloud)
        for i in range(self.num_stacks):
            conv_elm = self.ConvElements(pointcloud,  self.conv_elms_params[i])
            self.conv_elms.append(conv_elm)
            conv_layer = self.ConvLayer(self.conv_params[i], conv_elm)
            conv_layer = conv_layer.get_layer(network,
                                              with_bn=self.with_bn,
                                              bn_decay=self.bn_momentum)
            self.conv_stacks.append(conv_layer)
            if isinstance(conv_layer, list):
                to_pool = conv_layer + [pointcloud]
            else:
                to_pool = [conv_layer, pointcloud]
            pooled = self.PoolingLayer(ratio=self.pool_size[i],
                                       pool_mode=self.pool_mode[i])(to_pool)

            if isinstance(conv_layer, list):
                network = pooled[:-1]
            else:
                network = pooled[0]
            pointcloud = pooled[-1]
            self.pc_hierarchy.append(pointcloud)

        """
        one_hot_label_expand = RepeatVector(n=K.int_shape(network)[1])(one_hot_label)
        network = Concatenate(axis=-1)([network, one_hot_label_expand])
        """

        for i in range(len(self.pc_hierarchy)-1):
            j = len(self.pc_hierarchy)-2-i
            deconvnet = self.DeconvLayer(pc1=self.pc_hierarchy[j + 1], pc2=self.pc_hierarchy[j])(network)
            conv_elm = self.conv_elms[j]
            conv_layer = self.ConvLayer(self.conv_params[j], conv_elm)
            convnet = conv_layer.get_layer(deconvnet,
                                              with_bn=self.with_bn,
                                              bn_decay=self.bn_momentum)
            if isinstance(convnet, list):
                network = []
                for i in range(len(convnet)):
                    network.append(Concatenate(axis=-1)([self.conv_stacks[j][i], convnet[i]]))
            else:
                network = Concatenate(axis=-1)([self.conv_stacks[j], convnet])

            if j + 1 <= 2:
                if isinstance(network, list):
                    network_tmp = []
                    network_tmp.append(Dropout(rate=0.5)(network[0]))
                    network_tmp += network[1:]
                    network = network_tmp
                else:
                    network = Dropout(rate=0.5)(network)

        conv_elm = self.ConvElements(self.pointcloud,  self.config['classif_layer_elms_params'])
        # conv_elm = self.conv_elms[0]
        conv_layer = self.ConvLayer({'out_channels': part_num}, conv_elm)
        network = conv_layer.get_layer(network, with_bn=False, with_relu=False, bn_decay=self.bn_momentum)
        
        output = Activation('softmax')(network)


        # output = Dense(units=part_num, activation='softmax')(network)

        if self.InputLayer == '3d':
            inputs = [self.pointcloud, one_hot_label, input_signal]
        elif self.InputLayer is not None:
            inputs = [self.pointcloud, one_hot_label]
        else:
            inputs = [self.pointcloud, one_hot_label, input_signal]

        return Model(inputs=inputs, outputs=output)


