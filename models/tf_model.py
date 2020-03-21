import tensorflow as tf
import numpy as np
import os

def load_parameter(path):
    parameters = {}
    for parameter in os.listdir(path):
        parameters[parameter.split('.')[0]] = np.load(path + '/' + parameter)
        
    return parameters

const = {
    'block0_0_reshape0/shape': (-1, 784, 2, 58),
    'block0_0_transpose/perm': (0, 1, 3, 2),
    'block0_0_reshape1/shape': (-1, 28, 28, 116),
    'block0_1_slice-Slice0/size': (-1, 28, 28, 58),
    'block0_1_slice-Slice1/begin': (0, 0, 0, 58),
    'block0_1_slice-Slice0/begin': (0, 0, 0, 0),
    'block1_0_reshape0/shape': (-1, 196, 2, 116),
    'block1_0_reshape1/shape': (-1, 14, 14, 232),
    'block1_1_slice-Slice0/size': (-1, 14, 14, 116),
    'block1_1_slice-Slice1/begin': (0, 0, 0, 116),
    'block2_0_reshape0/shape': (-1, 49, 2, 232),
    'block2_0_reshape1/shape': (-1, 7, 7, 464),
    'block2_1_slice-Slice0/size': (-1, 7, 7, 232),
    'block2_1_slice-Slice1/begin': (0, 0, 0, 232)
}

def data_bn(input):
    with tf.name_scope('data_bn'):
        with tf.name_scope('data_bn'):
            with tf.name_scope('Rsqrt'):
                weights = tf.constant([0.9999949932098389, 0.9999949932098389, 0.9999949932098389], name='_0__cf__0', dtype=tf.float32)

            with tf.name_scope('mul') as scope:
                mul = tf.math.multiply(input, weights, name=scope)
                biases = tf.constant([-103.99948120117188, -116.9994125366211, -122.99938201904297], name='_1__cf__1', dtype=tf.float32)

            add_1 = tf.math.add(mul, biases, name='add_1')

            return add_1

def conv0(input, parameters, name='conv0'):
    with tf.name_scope(name) as scope:
        with tf.name_scope('pad_size') as pad_scope:
            const['conv0/pad_size/paddings'] = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]], dtype=tf.int32, shape=(4,2), name='paddings')
            pad_size = tf.pad(input, const['conv0/pad_size/paddings'], name=pad_scope)

        weights = tf.constant(parameters['conv0_weights'], name='weights', dtype=tf.float32)
        biases = tf.constant(parameters['conv0_biases'], name='biases', dtype=tf.float32)

        conv = tf.nn.conv2d(input, weights, strides=[1, 2, 2, 1], padding='VALID', dilations=[1, 1, 1, 1], name='Conv2D')
        bias_add = tf.nn.bias_add(conv, biases, name='BiasAdd')
        relu = tf.nn.relu(bias_add, name=scope)

        return relu

def conv1(input, parameters, name='conv1'):
    with tf.name_scope(name) as scope:
        weights = tf.constant(parameters['conv1_weights'], name='weights', dtype=tf.float32)
        biases = tf.constant(parameters['conv1_biases'], name='biases', dtype=tf.float32)

        conv = tf.nn.conv2d(input, weights, strides=[1, 1, 1, 1], padding='VALID', dilations=[1, 1, 1, 1], name='Conv2D')
        bias_add = tf.nn.bias_add(conv, biases, name='BiasAdd')
        relu = tf.nn.relu(bias_add, name=scope)

        return relu

def block0(input, parameters, name_id):
    name_template = 'block'
    name = name_template + str(name_id[0]) + '_' + str(name_id[1])
    with tf.name_scope(name + '_conv0') as scope:
        weights = tf.constant(parameters[scope[:-1] + '_weights'], name='weights', dtype=tf.float32)
        biases = tf.constant(parameters[scope[:-1] + '_biases'], name='biases', dtype=tf.float32)

        conv = tf.nn.conv2d(input, weights, strides=[1, 1, 1, 1], padding='SAME', dilations=[1, 1, 1, 1], name='Conv2D')
        bias_add = tf.nn.bias_add(conv, biases, name='BiasAdd')
        conv0 = tf.nn.relu(bias_add, name=scope)

    with tf.name_scope(name + '_conv1') as scope:
        weights = tf.constant(parameters[scope[:-1] + '_weights'], name='weights', dtype=tf.float32)
        biases = tf.constant(parameters[scope[:-1] + '_biases'], name='biases', dtype=tf.float32)

        pad_size = tf.pad(conv0, const['conv0/pad_size/paddings'], name='pad_size')
        conv = tf.compat.v1.nn.depthwise_conv2d_native(pad_size, weights, [1, 2, 2, 1], 'VALID', dilations=[1, 1, 1, 1], name='depthwise')
        conv1 = tf.nn.bias_add(conv, biases, name='BiasAdd')

    with tf.name_scope(name + '_conv2') as scope:
        weights = tf.constant(parameters[scope[:-1] + '_weights'], name='weights', dtype=tf.float32)
        biases = tf.constant(parameters[scope[:-1] + '_biases'], name='biases', dtype=tf.float32)

        conv = tf.nn.conv2d(conv1, weights, strides=[1, 1, 1, 1], padding='SAME', dilations=[1, 1, 1, 1], name='Conv2D')
        bias_add = tf.nn.bias_add(conv, biases, name='BiasAdd')
        conv2 = tf.nn.relu(bias_add, name=scope)

    with tf.name_scope(name + '_conv3') as scope:
        weights = tf.constant(parameters[scope[:-1] + '_weights'], name='weights', dtype=tf.float32)
        biases = tf.constant(parameters[scope[:-1] + '_biases'], name='biases', dtype=tf.float32)

        pad_size = tf.pad(input, const['conv0/pad_size/paddings'], name='pad_size')
        conv = tf.compat.v1.nn.depthwise_conv2d_native(pad_size, weights, [1, 2, 2, 1], 'VALID', dilations=[1, 1, 1, 1], name='depthwise')
        conv3 = tf.nn.bias_add(conv, biases, name='BiasAdd')

    with tf.name_scope(name + '_conv4') as scope:
        weights = tf.constant(parameters[scope[:-1] + '_weights'], name='weights', dtype=tf.float32)
        biases = tf.constant(parameters[scope[:-1] + '_biases'], name='biases', dtype=tf.float32)

        conv = tf.nn.conv2d(conv3, weights, strides=[1, 1, 1, 1], padding='SAME', dilations=[1, 1, 1, 1], name='Conv2D')
        bias_add = tf.nn.bias_add(conv, biases, name='BiasAdd')
        conv4 = tf.nn.relu(bias_add, name=scope)

    concat = tf.concat([conv4, conv2], 3, name+'_concat')
    reshape0 = tf.reshape(concat, const[name_template + str(name_id[0]) + '_0_reshape0/shape'], name+'_reshape0')
    transpose = tf.transpose(reshape0, const['block0_0_transpose/perm'], name=name+'_transpose')
    reshape1 = tf.reshape(transpose, const[name_template + str(name_id[0]) + '_0_reshape1/shape'], name+'_reshape1')

    return reshape1

def block1(input, parameters, name_id):
    name_template = 'block'
    name = name_template + str(name_id[0]) + '_' + str(name_id[1])

    slice0 = tf.slice(input, const['block0_1_slice-Slice0/begin'], const[name_template + str(name_id[0]) + '_1_slice-Slice0/size'], name+'_slice0')

    slice1 = tf.slice(input, const[name_template + str(name_id[0]) + '_1_slice-Slice1/begin'], const[name_template + str(name_id[0]) + '_1_slice-Slice0/size'], name+'_slice1')
    with tf.name_scope(name + '_conv0') as scope:
        weights = tf.constant(parameters[scope[:-1] + '_weights'], name='weights', dtype=tf.float32)
        biases = tf.constant(parameters[scope[:-1] + '_biases'], name='biases', dtype=tf.float32)

        conv = tf.nn.conv2d(slice1, weights, strides=[1, 1, 1, 1], padding='SAME', dilations=[1, 1, 1, 1], name='Conv2D')
        bias_add = tf.nn.bias_add(conv, biases, name='BiasAdd')
        conv0 = tf.nn.relu(bias_add, name=scope)

    with tf.name_scope(name + '_conv1') as scope:
        weights = tf.constant(parameters[scope[:-1] + '_weights'], name='weights', dtype=tf.float32)
        biases = tf.constant(parameters[scope[:-1] + '_biases'], name='biases', dtype=tf.float32)

        pad_size = tf.pad(conv0, const['conv0/pad_size/paddings'], name='pad_size')
        conv = tf.compat.v1.nn.depthwise_conv2d_native(pad_size, weights, [1, 1, 1, 1], 'VALID', dilations=[1, 1, 1, 1], name='depthwise')
        conv1 = tf.nn.bias_add(conv, biases, name='BiasAdd')

    with tf.name_scope(name + '_conv2') as scope:
        weights = tf.constant(parameters[scope[:-1] + '_weights'], name='weights', dtype=tf.float32)
        biases = tf.constant(parameters[scope[:-1] + '_biases'], name='biases', dtype=tf.float32)

        conv = tf.nn.conv2d(conv1, weights, strides=[1, 1, 1, 1], padding='SAME', dilations=[1, 1, 1, 1], name='Conv2D')
        bias_add = tf.nn.bias_add(conv, biases, name='BiasAdd')
        conv2 = tf.nn.relu(bias_add, name=scope)

    concat = tf.concat([slice0, conv2], 3, name+'_concat')
    reshape0 = tf.reshape(concat, const[name_template + str(name_id[0]) + '_0_reshape0/shape'], name+'_reshape0')
    transpose = tf.transpose(reshape0, const['block0_0_transpose/perm'], name=name+'_transpose')
    reshape1 = tf.reshape(transpose, const[name_template + str(name_id[0]) + '_0_reshape1/shape'], name+'_reshape1')

    return reshape1

def fc(input, weights, biases):
    with tf.name_scope('fc') as scope:
        reshape = tf.reshape(input, [-1, 1024], name='Reshape')
        matmul = tf.linalg.matmul(reshape, weights, name='MatMul')
        biasadd = tf.nn.bias_add(matmul, biases, name=scope)

        return biasadd, weights, biases

class Model(object):
    def __init__(self, label_count):
        self.parameters = load_parameter('./models/parameters/')
        with tf.name_scope('fc') as scope:
            self.w = tf.Variable(tf.random.normal((1024, label_count)), name='weights', dtype=tf.float32)
            self.b = tf.Variable(tf.random.normal((label_count,)), name='biases', dtype=tf.float32)

    def __call__(self, input):
        layer = data_bn(input)
        layer = conv0(layer, self.parameters)
        pad_size = tf.pad(layer, const['conv0/pad_size/paddings'], name='pad_size')
        max_pool = tf.nn.max_pool(pad_size, [1, 3, 3, 1], [1, 2, 2, 1], 'VALID', name='pool0')
        layer = block0(max_pool, self.parameters, (0, 0))
        layer = block1(layer, self.parameters, (0, 1))
        layer = block1(layer, self.parameters, (0, 2))
        layer = block1(layer, self.parameters, (0, 3))
        layer = block0(layer, self.parameters, (1, 0))
        layer = block1(layer, self.parameters, (1, 1))
        layer = block1(layer, self.parameters, (1, 2))
        layer = block1(layer, self.parameters, (1, 3))
        layer = block1(layer, self.parameters, (1, 4))
        layer = block1(layer, self.parameters, (1, 5))
        layer = block1(layer, self.parameters, (1, 6))
        layer = block1(layer, self.parameters, (1, 7))
        layer = block0(layer, self.parameters, (2, 0))
        layer = block1(layer, self.parameters, (2, 1))
        layer = block1(layer, self.parameters, (2, 2))
        layer = block1(layer, self.parameters, (2, 3))
        layer = conv1(layer, self.parameters)
        layer = tf.nn.avg_pool(layer, [1, 7, 7, 1], [1, 1, 1, 1], 'VALID', name='pool1')
        layer, weights, biases = fc(layer, self.w, self.b)

        return layer
