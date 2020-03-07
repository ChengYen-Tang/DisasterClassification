from utils.image_processing import load_files
from models.tf2_model import Model

import numpy as np
import tensorflow as tf

def one_hot(indices, depth):
    return np.eye(depth)[indices.reshape(-1)]

images, labels = load_files('./dataset/train/')
images_shape = images.shape

label_count = len(np.unique(labels))

one_hot_label = one_hot(labels, label_count)

optimizer = tf.optimizers.Adam()

def train(model, inputs, outputs):
    with tf.GradientTape() as t:
        current_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model(inputs), labels=outputs))
    grads = t.gradient(current_loss, [model.w, model.b])
    optimizer.apply_gradients(zip(grads,[model.w, model.b]))
    return current_loss

model = Model(label_count)

test_images, test_labels = load_files('./dataset/test/')

steps = 1000
for i in range(steps):
    print('steps [ %d / %d ]: loss - %f' %(i, steps + 1, train(model,images,one_hot_label)))

    prediction = tf.nn.softmax(model(test_images))
    correct_pred = tf.equal(tf.argmax(prediction, 1), test_labels)
    accuracy = tf.math.reduce_mean(tf.cast(correct_pred, tf.float32))

    print(accuracy*100)

