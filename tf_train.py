from utils.image_processing import load_files
from models.tf_model import Model
from tensorflow.python.framework import graph_util

import numpy as np
import tensorflow as tf

def one_hot(indices, depth):
    return np.eye(depth)[indices.reshape(-1)]

images, labels = load_files('./dataset/train/')
images_shape = images.shape

label_count = len(set(labels))

one_hot_label = one_hot(labels, label_count)

steps = 1000

with tf.Session(graph=tf.Graph()) as sess:
    inputs = tf.placeholder("float", [None, images_shape[1], images_shape[2], images_shape[3]])
    labels = tf.placeholder("float", [None, label_count])

    model = Model(label_count)

    logits = model(inputs)
    prediction = tf.nn.softmax(logits, name='loss')

    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    optimizer = tf.train.AdamOptimizer()

    grads = tf.gradients(loss_op, [model.w, model.b])
    grads = list(zip(grads,[model.w, model.b]))
    train = optimizer.apply_gradients(grads)
    sess.run(tf.global_variables_initializer())
    
    for i in range(steps):
        loss, _ = sess.run([loss_op, train], feed_dict={inputs: images, labels: one_hot_label})
        print('steps [ %d / %d ]: loss - %f' %(i + 1, steps, loss))

    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['loss'])

    with tf.gfile.FastGFile('./model.pb', mode='wb') as f:
        f.write(constant_graph.SerializeToString())
