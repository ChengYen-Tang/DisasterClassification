from utils.image_processing import load_files
from models.tf2_model import Model

import numpy as np
import tensorflow as tf

def one_hot(indices, depth):
    return np.eye(depth)[indices.reshape(-1)]

images, labels = load_files('./dataset/train/')
images_shape = images.shape

one_hot_label = one_hot(labels,4)

optimizer = tf.optimizers.Adam()

def train(model, inputs, outputs):
    with tf.GradientTape() as t:
        current_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model(inputs), labels=outputs))
    grads = t.gradient(current_loss, [model.w, model.b])
    optimizer.apply_gradients(zip(grads,[model.w, model.b]))
    print(current_loss)

model = Model(4)

for i in range(10000):
    train(model,images,one_hot_label)