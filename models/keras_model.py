from tensorflow import keras
from tensorflow.keras import layers, backend

import tensorflow as tf

const = {
    'transpose_perm': [0, 1, 3, 2],
    'block0_0_reshape0': [784, 2],
    'block0_0_reshape1': [28, 28],
    'slice-Slice0/begin': (0, 0, 0),
    'block0_1_slice-Slice0/size': (28, 28, 58),
    'block0_1_slice-Slice1/begin': (0, 0, 58),
    'block1_0_reshape0/shape': [196, 2],
    'block1_0_reshape1/shape': [14, 14],
    'block1_1_slice-Slice0/size': (14, 14, 116),
    'block1_1_slice-Slice1/begin': (0, 0, 116),
    'block2_0_reshape0/shape': [49, 2],
    'block2_0_reshape1/shape': [7, 7],
    'block2_1_slice-Slice0/size': (7, 7, 232),
    'block2_1_slice-Slice1/begin': (0, 0, 232)
}

def setup_model(input_shape, label_count):


    model = keras.Sequential(
        [
            layers.Input(input_shape),
            Linear1(),
            layers.ZeroPadding2D(padding=((1, 1), (1, 1))),
            layers.Conv2D(24, (3, 3), strides=(2, 2), padding='VALID', activation='relu'),
            layers.ZeroPadding2D(padding=((1, 1), (1, 1))),
            layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
            block_1(58, const['block0_0_reshape0'], const['block0_0_reshape1']),
            block_2(58, const['block0_1_slice-Slice0/size'], const['slice-Slice0/begin'], const['block0_1_slice-Slice1/begin'], const['block0_0_reshape0'], const['block0_0_reshape1']),
            block_2(58, const['block0_1_slice-Slice0/size'], const['slice-Slice0/begin'], const['block0_1_slice-Slice1/begin'], const['block0_0_reshape0'], const['block0_0_reshape1']),
            block_2(58, const['block0_1_slice-Slice0/size'], const['slice-Slice0/begin'], const['block0_1_slice-Slice1/begin'], const['block0_0_reshape0'], const['block0_0_reshape1']),
            # block_1(116, const['block1_0_reshape0/shape'], const['block1_0_reshape1/shape']),
            # block_2(116, const['block1_1_slice-Slice0/size'], const['slice-Slice0/begin'], const['block1_1_slice-Slice1/begin'], const['block1_0_reshape0/shape'], const['block1_0_reshape1/shape']),
            # block_2(116, const['block1_1_slice-Slice0/size'], const['slice-Slice0/begin'], const['block1_1_slice-Slice1/begin'], const['block1_0_reshape0/shape'], const['block1_0_reshape1/shape']),
            # block_2(116, const['block1_1_slice-Slice0/size'], const['slice-Slice0/begin'], const['block1_1_slice-Slice1/begin'], const['block1_0_reshape0/shape'], const['block1_0_reshape1/shape']),
            # block_2(116, const['block1_1_slice-Slice0/size'], const['slice-Slice0/begin'], const['block1_1_slice-Slice1/begin'], const['block1_0_reshape0/shape'], const['block1_0_reshape1/shape']),
            # block_2(116, const['block1_1_slice-Slice0/size'], const['slice-Slice0/begin'], const['block1_1_slice-Slice1/begin'], const['block1_0_reshape0/shape'], const['block1_0_reshape1/shape']),
            # block_2(116, const['block1_1_slice-Slice0/size'], const['slice-Slice0/begin'], const['block1_1_slice-Slice1/begin'], const['block1_0_reshape0/shape'], const['block1_0_reshape1/shape']),
            # block_2(116, const['block1_1_slice-Slice0/size'], const['slice-Slice0/begin'], const['block1_1_slice-Slice1/begin'], const['block1_0_reshape0/shape'], const['block1_0_reshape1/shape']),
            # block_1(232, const['block2_0_reshape0/shape'], const['block2_0_reshape1/shape']),
            # block_2(232, const['block2_1_slice-Slice0/size'], const['slice-Slice0/begin'], const['block2_1_slice-Slice1/begin'], const['block2_0_reshape0/shape'], const['block2_0_reshape1/shape']),
            # block_2(232, const['block2_1_slice-Slice0/size'], const['slice-Slice0/begin'], const['block2_1_slice-Slice1/begin'], const['block2_0_reshape0/shape'], const['block2_0_reshape1/shape']),
            # block_2(232, const['block2_1_slice-Slice0/size'], const['slice-Slice0/begin'], const['block2_1_slice-Slice1/begin'], const['block2_0_reshape0/shape'], const['block2_0_reshape1/shape']),
            layers.Conv2D(1024, (1, 1), padding='SAME', activation='relu'),
            layers.AveragePooling2D(pool_size=(28, 28), strides=(1, 1)),
            Linear2(label_count)
        ])

    return model

class Linear1(layers.Layer):
    def __init__(self):
        super(Linear1, self).__init__()
        self.w = tf.Variable([0.9999949932098389, 0.9999949932098389, 0.9999949932098389], name='_0__cf__0', dtype=tf.float32)
        self.b = tf.Variable([-103.99948120117188, -116.9994125366211, -122.99938201904297], name='_1__cf__1', dtype=tf.float32)

    def call(self, inputs):
        mul = tf.math.multiply(inputs, self.w)
        add_1 = tf.math.add(mul, self.b)

        return add_1

class Linear2(layers.Layer):
    def __init__(self, units):
        super(Linear2, self).__init__()
        self.units = units

    def build(self, input_shape):
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(initial_value=w_init(shape=(input_shape[-1], self.units), dtype='float32'), trainable=True)
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(initial_value=b_init(shape=(self.units,), dtype='float32'), trainable=True)

    def call(self, inputs):
        matmul = tf.linalg.matmul(inputs, self.w)
        biasadd = tf.nn.bias_add(matmul, self.b)

        return biasadd

class slice(layers.Layer):
    def __init__(self, begin, size):
        super(slice, self).__init__()
        self.begin = begin
        self.size = size

    def call(self, inputs):
        return tf.slice(inputs, (0,) + self.begin, (tf.shape(inputs)[0],) + self.size)

class block_0(layers.Layer):
    def __init__(self, strides):
        super(block_0, self).__init__()
        self.pad_size = layers.ZeroPadding2D(padding=((1, 1), (1, 1)))
        self.DepthwiseConv2D = layers.DepthwiseConv2D((3,3), strides=strides)

    def call(self, inputs):
        return self.DepthwiseConv2D(self.pad_size(inputs))

class block_1(layers.Layer):
    def __init__(self, filters, shape_0, shape_1):
        super(block_1, self).__init__()
        self.conv0 = layers.Conv2D(filters, (1, 1), padding='SAME', activation='relu')
        self.conv1 = block_0((2, 2))
        self.conv2 = layers.Conv2D(filters, (1, 1), padding='SAME', activation='relu')

        self.conv3 = block_0((2, 2))
        self.conv4 = layers.Conv2D(filters, (1, 1), padding='SAME', activation='relu')

        self.reshap0 = layers.Reshape((shape_0[0], shape_0[1], filters))
        self.reshap1 = layers.Reshape((shape_1[0], shape_1[1], filters * 2))

    def call(self, inputs):
        conv0_2 = self.conv0(inputs)
        conv0_2 = self.conv1(conv0_2)
        conv0_2 = self.conv2(conv0_2)

        conv3_4 = self.conv3(inputs)
        conv3_4 = self.conv4(conv3_4)

        concat = layers.concatenate([conv0_2, conv3_4], axis=2)
        reshape = self.reshap0(concat)
        permute_dimensions = backend.permute_dimensions(reshape, pattern=const['transpose_perm'])
        reshape = self.reshap1(permute_dimensions)

        return reshape

class block_2(layers.Layer):
    def __init__(self, filters, slice_size, slice0_begin, slice1_begin, shape_0, shape_1):
        super(block_2, self).__init__()
        self.slice0 = slice(slice0_begin, slice_size)

        self.slice1 = slice(slice1_begin, slice_size)
        self.conv0 = layers.Conv2D(filters, (1, 1), padding='SAME', activation='relu')
        self.conv1 = block_0((1, 1))
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
