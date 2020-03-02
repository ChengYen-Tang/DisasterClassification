from utils.image_processing import load_files
from models.keras_model import setup_model
from tensorflow import keras

import numpy as np
import tensorflow as tf

def one_hot(indices, depth):
    return np.eye(depth)[indices.reshape(-1)]

images, labels = load_files('./dataset/train/')
images_shape = images.shape

one_hot_label = one_hot(labels,4)

model = setup_model((images_shape[1], images_shape[2], images_shape[3]), 4)
model.compile(optimizer=keras.optimizers.Adadelta(),
             loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy'])
model.summary()

history = model.fit(images, labels, batch_size=320, epochs=10000)
