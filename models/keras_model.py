from tensorflow import keras
from tensorflow.keras import layers, backend

import tensorflow as tf

const = {
    'transpose_perm': [0, 1, 3, 2],
    'conv_block0_0_reshape0': [784, 2],
    'conv_block0_0_reshape1': [28, 28],
    'slice-Slice0/begin': (0, 0, 0),
    'conv_block0_1_slice-Slice0/size': (28, 28, 58),
    'conv_block0_1_slice-Slice1/begin': (0, 0, 58),
    'conv_block1_0_reshape0/shape': [196, 2],
    'conv_block1_0_reshape1/shape': [14, 14],
    'conv_block1_1_slice-Slice0/size': (14, 14, 116),
    'conv_block1_1_slice-Slice1/begin': (0, 0, 116),
    'conv_block2_0_reshape0/shape': [49, 2],
    'conv_block2_0_reshape1/shape': [7, 7],
    'conv_block2_1_slice-Slice0/size': (7, 7, 232),
    'conv_block2_1_slice-Slice1/begin': (0, 0, 232)
}

def setup_model(input_shape):


    model = keras.Sequential(
        [
            layers.Input(input_shape),
            linear1(),
            layers.ZeroPadding2D(padding=((1, 1), (1, 1))),
            layers.Conv2D(24, (3, 3), strides=(2, 2), padding='VALID', activation='relu'),
            layers.ZeroPadding2D(padding=((1, 1), (1, 1))),
            layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
            conv_block1(58, const['conv_block0_0_reshape0'], const['conv_block0_0_reshape1']),
            conv_block2(58, const['conv_block0_1_slice-Slice0/size'], const['slice-Slice0/begin'], const['conv_block0_1_slice-Slice1/begin'], const['conv_block0_0_reshape0'], const['conv_block0_0_reshape1']),
            conv_block2(58, const['conv_block0_1_slice-Slice0/size'], const['slice-Slice0/begin'], const['conv_block0_1_slice-Slice1/begin'], const['conv_block0_0_reshape0'], const['conv_block0_0_reshape1']),
            conv_block2(58, const['conv_block0_1_slice-Slice0/size'], const['slice-Slice0/begin'], const['conv_block0_1_slice-Slice1/begin'], const['conv_block0_0_reshape0'], const['conv_block0_0_reshape1']),
            conv_block1(116, const['conv_block1_0_reshape0/shape'], const['conv_block1_0_reshape1/shape']),
            conv_block2(116, const['conv_block1_1_slice-Slice0/size'], const['slice-Slice0/begin'], const['conv_block1_1_slice-Slice1/begin'], const['conv_block1_0_reshape0/shape'], const['conv_block1_0_reshape1/shape']),
            conv_block2(116, const['conv_block1_1_slice-Slice0/size'], const['slice-Slice0/begin'], const['conv_block1_1_slice-Slice1/begin'], const['conv_block1_0_reshape0/shape'], const['conv_block1_0_reshape1/shape']),
            conv_block2(116, const['conv_block1_1_slice-Slice0/size'], const['slice-Slice0/begin'], const['conv_block1_1_slice-Slice1/begin'], const['conv_block1_0_reshape0/shape'], const['conv_block1_0_reshape1/shape']),
            conv_block2(116, const['conv_block1_1_slice-Slice0/size'], const['slice-Slice0/begin'], const['conv_block1_1_slice-Slice1/begin'], const['conv_block1_0_reshape0/shape'], const['conv_block1_0_reshape1/shape']),
            conv_block2(116, const['conv_block1_1_slice-Slice0/size'], const['slice-Slice0/begin'], const['conv_block1_1_slice-Slice1/begin'], const['conv_block1_0_reshape0/shape'], const['conv_block1_0_reshape1/shape']),
            conv_block2(116, const['conv_block1_1_slice-Slice0/size'], const['slice-Slice0/begin'], const['conv_block1_1_slice-Slice1/begin'], const['conv_block1_0_reshape0/shape'], const['conv_block1_0_reshape1/shape']),
            conv_block2(116, const['conv_block1_1_slice-Slice0/size'], const['slice-Slice0/begin'], const['conv_block1_1_slice-Slice1/begin'], const['conv_block1_0_reshape0/shape'], const['conv_block1_0_reshape1/shape']),
            conv_block1(232, const['conv_block2_0_reshape0/shape'], const['conv_block2_0_reshape1/shape']),
            conv_block2(232, const['conv_block2_1_slice-Slice0/size'], const['slice-Slice0/begin'], const['conv_block2_1_slice-Slice1/begin'], const['conv_block2_0_reshape0/shape'], const['conv_block2_0_reshape1/shape']),
            conv_block2(232, const['conv_block2_1_slice-Slice0/size'], const['slice-Slice0/begin'], const['conv_block2_1_slice-Slice1/begin'], const['conv_block2_0_reshape0/shape'], const['conv_block2_0_reshape1/shape']),
            conv_block2(232, const['conv_block2_1_slice-Slice0/size'], const['slice-Slice0/begin'], const['conv_block2_1_slice-Slice1/begin'], const['conv_block2_0_reshape0/shape'], const['conv_block2_0_reshape1/shape']),
            layers.Conv2D(1024, (1, 1), padding='SAME', activation='relu'),
            layers.AveragePooling2D(pool_size=(7, 7), strides=(1, 1))
        ])

    return model

class linear1(layers.Layer):
    def __init__(self):
        super(linear1, self).__init__()
        self.w = tf.Variable([0.9999949932098389, 0.9999949932098389, 0.9999949932098389], dtype=tf.float32, name='weights')
        self.b = tf.Variable([-103.99948120117188, -116.9994125366211, -122.99938201904297], dtype=tf.float32, name='biases')

    def call(self, inputs):
        mul = tf.math.multiply(inputs, self.w)
        add_1 = tf.math.add(mul, self.b)

        return add_1

    def get_config(self):
        config = super(linear1, self).get_config()
        return config

class linear2(layers.Layer):
    def __init__(self, units):
        super(linear2, self).__init__()
        self.units = units

    def build(self, input_shape):
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(initial_value=w_init(shape=(input_shape[-1], self.units), dtype='float32'), trainable=True, name='weights')
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(initial_value=b_init(shape=(self.units,), dtype='float32'), trainable=True, name='biases')

    def call(self, inputs):
        matmul = tf.linalg.matmul(inputs, self.w)
        biasadd = tf.nn.bias_add(matmul, self.b)

        return biasadd

    def get_config(self):
        config = super(linear2, self).get_config()
        config.update({'units': self.units})
        return config

class slice(layers.Layer):
    def __init__(self, begin, size):
        super(slice, self).__init__()
        self.begin = begin
        self.size = size

    def call(self, inputs):
        return tf.slice(inputs, (0,) + self.begin, (tf.shape(inputs)[0],) + self.size)

    def get_config(self):
        config = {'begin': self.begin, 'size': self.size}
        return dict(list(super(slice, self).get_config().items()) + list(config.items()))

class conv_block0(layers.Layer):
    def __init__(self, strides):
        super(conv_block0, self).__init__()
        self.strides = strides
        self.pad_size = layers.ZeroPadding2D(padding=((1, 1), (1, 1)))
        self.DepthwiseConv2D = layers.DepthwiseConv2D((3,3), strides=self.strides)

    def call(self, inputs):
        return self.DepthwiseConv2D(self.pad_size(inputs))

    def get_config(self):
        config = {'strides': self.strides}
        return dict(list(super(conv_block0, self).get_config().items()) + list(config.items()))

class conv_block1(layers.Layer):
    def __init__(self, filters, shape_0, shape_1):
        super(conv_block1, self).__init__()
        self.filters = filters
        self.shape_0 = shape_0
        self.shape_1 = shape_1
        self.conv0 = layers.Conv2D(filters, (1, 1), padding='SAME', activation='relu')
        self.conv1 = conv_block0((2, 2))
        self.conv2 = layers.Conv2D(filters, (1, 1), padding='SAME', activation='relu')

        self.conv3 = conv_block0((2, 2))
        self.conv4 = layers.Conv2D(filters, (1, 1), padding='SAME', activation='relu')

        self.reshap0 = layers.Reshape((shape_0[0], shape_0[1], filters))
        self.reshap1 = layers.Reshape((shape_1[0], shape_1[1], filters * 2))

    def call(self, inputs):
        conv_0 = self.conv0(inputs)
        conv_1 = self.conv1(conv_0)
        conv_2 = self.conv2(conv_1)

        conv_3 = self.conv3(inputs)
        conv_4 = self.conv4(conv_3)

        concat = layers.concatenate([conv_2, conv_4], axis=3)
        reshape = self.reshap0(concat)
        permute_dimensions = backend.permute_dimensions(reshape, pattern=const['transpose_perm'])
        reshape = self.reshap1(permute_dimensions)

        return reshape

    def get_config(self):
        config = {'filters': self.filters, 'shape_0': self.shape_0, 'shape_1': self.shape_1}
        return dict(list(super(conv_block1, self).get_config().items()) + list(config.items()))

class conv_block2(layers.Layer):
    def __init__(self, filters, slice_size, slice0_begin, slice1_begin, shape_0, shape_1):
        super(conv_block2, self).__init__()
        self.filters = filters
        self.slice_size = slice_size
        self.slice0_begin = slice0_begin
        self.slice1_begin = slice1_begin
        self.shape_0 = shape_0
        self.shape_1 = shape_1
        self.slice0 = slice(slice0_begin, slice_size)

        self.slice1 = slice(slice1_begin, slice_size)
        self.conv0 = layers.Conv2D(filters, (1, 1), padding='SAME', activation='relu')
        self.conv1 = conv_block0((1, 1))
        self.conv2 = layers.Conv2D(filters, (1, 1), padding='SAME', activation='relu')

        self.reshap0 = layers.Reshape((shape_0[0], shape_0[1], filters))
        self.reshap1 = layers.Reshape((shape_1[0], shape_1[1], filters * 2))

    def call(self, inputs):
        slice0 = self.slice0(inputs)

        slice1 = self.slice1(inputs)
        conv0 = self.conv0(slice1)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)

        concat = layers.concatenate([slice0, conv2], axis=3)
        reshape = self.reshap0(concat)
        permute_dimensions = backend.permute_dimensions(reshape, pattern=const['transpose_perm'])
        reshape = self.reshap1(permute_dimensions)

        return reshape

    def get_config(self):
        config = {'filters': self.filters, 'slice_size': self.slice_size, 'slice0_begin': self.slice0_begin, 'slice1_begin': self.slice1_begin, 'shape_0': self.shape_0, 'shape_1': self.shape_1}
        return dict(list(super(conv_block2, self).get_config().items()) + list(config.items()))
