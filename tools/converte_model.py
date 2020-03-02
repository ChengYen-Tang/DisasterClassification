import os, sys
sys.path.append('./')

from models.keras_model import setup_model
from tensorflow import keras

import numpy as np

def load_parameter(path):
    parameters = {}
    for parameter in os.listdir(path):
        parameters[parameter.split('.')[0]] = np.load(path + '/' + parameter)
        
    return parameters

def block1(parameters, model, parameters_index, block_index=None):
    weights = [parameters['block' + str(parameters_index[0]) + '_' + str(parameters_index[1]) + '_conv0_weights'], 
                parameters['block' + str(parameters_index[0]) + '_' + str(parameters_index[1]) + '_conv0_biases'],
                parameters['block' + str(parameters_index[0]) + '_' + str(parameters_index[1]) + '_conv1_weights'],
                parameters['block' + str(parameters_index[0]) + '_' + str(parameters_index[1]) + '_conv1_biases'],
                parameters['block' + str(parameters_index[0]) + '_' + str(parameters_index[1]) + '_conv2_weights'],
                parameters['block' + str(parameters_index[0]) + '_' + str(parameters_index[1]) + '_conv2_biases'],
                parameters['block' + str(parameters_index[0]) + '_' + str(parameters_index[1]) + '_conv3_weights'],
                parameters['block' + str(parameters_index[0]) + '_' + str(parameters_index[1]) + '_conv3_biases'],
                parameters['block' + str(parameters_index[0]) + '_' + str(parameters_index[1]) + '_conv4_weights'],
                parameters['block' + str(parameters_index[0]) + '_' + str(parameters_index[1]) + '_conv4_biases']]
    if block_index is None:
        model.get_layer('conv_block1').set_weights(weights)
    else:
        model.get_layer('conv_block1_' + str(block_index)).set_weights(weights)

def block2(parameters, model, parameters_index, block_index=None):
    weights = [parameters['block' + str(parameters_index[0]) + '_' + str(parameters_index[1]) + '_conv0_weights'], 
            parameters['block' + str(parameters_index[0]) + '_' + str(parameters_index[1]) + '_conv0_biases'],
            parameters['block' + str(parameters_index[0]) + '_' + str(parameters_index[1]) + '_conv1_weights'],
            parameters['block' + str(parameters_index[0]) + '_' + str(parameters_index[1]) + '_conv1_biases'],
            parameters['block' + str(parameters_index[0]) + '_' + str(parameters_index[1]) + '_conv2_weights'],
            parameters['block' + str(parameters_index[0]) + '_' + str(parameters_index[1]) + '_conv2_biases']]
    if block_index is None:
        model.get_layer('conv_block2').set_weights(weights)
    else:
        model.get_layer('conv_block2_' + str(block_index)).set_weights(weights)

def conv2d(parameters, model, parameters_index, block_index=None):
    weights = [parameters['conv' + str(parameters_index) + '_weights'], parameters['conv' + str(parameters_index) + '_biases']]
    if block_index is None:
        model.get_layer('conv2d').set_weights(weights)
    else:
        model.get_layer('conv2d_' + str(block_index)).set_weights(weights)

parameters = load_parameter('./tools/parameter/')
model = setup_model((224, 224, 3))

conv2d(parameters, model, 0)
conv2d(parameters, model, 1, 36)

block1(parameters, model, (0, 0))
block2(parameters, model, (0, 1))
block2(parameters, model, (0, 2), 1)
block2(parameters, model, (0, 3), 2)

block1(parameters, model, (1, 0), 1)
block2(parameters, model, (1, 1), 3)
block2(parameters, model, (1, 2), 4)
block2(parameters, model, (1, 3), 5)
block2(parameters, model, (1, 4), 6)
block2(parameters, model, (1, 5), 7)
block2(parameters, model, (1, 6), 8)
block2(parameters, model, (1, 7), 9)

block1(parameters, model, (2, 0), 2)
block2(parameters, model, (2, 1), 10)
block2(parameters, model, (2, 2), 11)
block2(parameters, model, (2, 3), 12)

model.save_weights('./models/keras_model_weights.h5')
