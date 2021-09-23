#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File       : model.py
# Modified   : 23.09.2021
# By         : Andreas Persson <andreas.persson@ai.se>

import argparse

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPool2D
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.losses import CategoricalCrossentropy, SparseCategoricalCrossentropy
from tensorflow_privacy.privacy.keras_models.dp_keras_model import DPModel

'''
Functions for creating subset of model layers.
''' 
def dense_module(x, dim, activation = 'relu', dropout = 0.0):
    # Dense fully connected module.
    if activation is not None:
        x = Dense(dim, activation = activation)(x)
    else:
        x = Dense(dim)(x)
    if dropout > 0.0:
        x = Dropout(dropout)(x)
    return x

def conv_maxpool_module(x, filters, kernel, strides = 2, padding = 'same', activation = 'relu'):
    # Convolutional moduel. 
    x = Conv2D(filters, kernel, strides = strides, padding = padding, activation = activation)(x)
    x = MaxPool2D(2, 1)(x)
    return x

'''
Function for creating a model (with or without differentially private (DP))
according to the Functional API.
'''
def sequential_model(input_shape = (28, 28, 1), num_classes = 10, model_type = 'cnn', dp = True, l2_norm_clip = 1.0, noise_multiplier = 0.1):
    inputs = Input(shape = input_shape)
    if model_type == 'cnn':
        x = conv_maxpool_module(inputs, 16, 8)
        x = conv_maxpool_module(x, 32, 4, padding = 'valid')
        x = Flatten()(x)
        x = dense_module(x, 32, dropout = 0.2)
    else:
        x = Flatten(input_shape = input_shape)(inputs)
        x = dense_module(x, 64)
        x = dense_module(x, 64, dropout = 0.2)
    outputs = dense_module(x, num_classes)

    # Create and return the model
    if dp:
        model = DPModel( l2_norm_clip,
                         noise_multiplier,
                         inputs = inputs,
                         outputs = outputs,
                         name = "dp-" + model_type)
    else:
        model = Model(inputs= inputs, outputs = outputs, name = model_type)    
    return model

'''
Function for creating an optimizer.
'''
def optimizer(opt_type = 'sgd', learning_rate = 0.1):
    if opt_type == 'adam':
        return Adam(learning_rate = learning_rate)
    return SGD(learning_rate = learning_rate)

'''
Function for defining a categorical loss function (either sparse or not).
Sparse should be used if labels are expected to be provided as integers.
'''
def categorical_loss(sparse = False):
    if sparse:
        return SparseCategoricalCrossentropy(from_logits = True)
    return CategoricalCrossentropy(from_logits = True)
    

# Main function (for testing)
if __name__ == '__main__':

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument( '--model', type = str, default = 'mlp',
	                 choices = ['mlp', 'cnn'],
	                 help = "type of learning model")
    parser.add_argument( '--dp', default = True, action='store_true',
                         help = "using differentially privacy...  ")  
    parser.add_argument('--no-dp', dest='dp', action='store_false', 
                         help = "  ...or not.")  
    args = vars(parser.parse_args())
    
    # Create model
    model = sequential_model(model_type = args['model'], dp = args['dp'])
    model.summary()

    
