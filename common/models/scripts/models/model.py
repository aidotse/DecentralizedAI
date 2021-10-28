#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File       : model.py
# Modified   : 04.10.2021
# By         : Andreas Persson <andreas.persson@ai.se>

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPool2D, Softmax
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.losses import CategoricalCrossentropy, SparseCategoricalCrossentropy
from tensorflow_privacy.privacy.keras_models.dp_keras_model import DPModel

'''
Wrapper class for creating a model according to the Functional API.
'''
class SequentialModel:

    # Static method building a sequential model.    
    @staticmethod
    def build(model_type = 'mlp', input_shape = (28, 28, 1), num_classes = 10, **kwargs):
        inputs = Input(shape = input_shape)
        if model_type == 'cnn':
            x = SequentialModel.conv_maxpool(inputs, 16, 8)
            x = SequentialModel.conv_maxpool(x, 32, 4, padding = 'valid')
            x = Flatten()(x)
            x = SequentialModel.dense(x, 32)
        else:
            x = Flatten(input_shape = input_shape)(inputs)
            x = SequentialModel.dense(x, 128)
            x = SequentialModel.dense(x, 64)
        outputs = SequentialModel.dense(x, num_classes)

        # Create and return the model
        if kwargs and 'dp' in kwargs and kwargs['dp']:
            if 'l2_norm_clip' in kwargs and type(kwargs['l2_norm_clip']) == float:
                l2_norm_clip = kwargs['l2_norm_clip']
            else:
                l2_norm_clip = 1.0
            if 'noise_multiplier' in kwargs and type(kwargs['noise_multiplier']) == float:
                noise_multiplier = kwargs['noise_multiplier']
            else:
                noise_multiplier = 0.1
            model = DPModel( l2_norm_clip,
                             noise_multiplier,
                             inputs = inputs,
                             outputs = outputs,
                             name = "dp-" + model_type)
        else:
            model = Model(inputs= inputs, outputs = outputs, name = model_type)    
        return model

    # Static method for creating a dense fully connected module.
    @staticmethod
    def dense(x, dim, activation = 'relu', dropout = 0.0):        
        x = Dense(dim, activation = activation)(x)
        if dropout > 0.0:
            x = Dropout(dropout)(x)
        return x

    
    # Static method for creating a convolutional module.
    @staticmethod
    def conv_maxpool(x, filters, kernel, strides = 2, padding = 'same', activation = 'relu'):
        x = Conv2D(filters, kernel, strides = strides, padding = padding, activation = activation)(x)
        x = MaxPool2D(2, 1)(x)
        return x


'''
Function for appending a sofmax layer (and turning a model into a probability model).
'''
def ProbabilityModel(model):
    model = tf.keras.Sequential([
        model,
        Softmax()
    ])
    return model

'''
Function for creating an optimizer.
'''
def optimizer(opt_type = 'sgd', learning_rate = 0.01, decay = 0.0):
    if opt_type == 'adam':
        return Adam(learning_rate = learning_rate, decay = decay)
    return SGD(learning_rate = learning_rate, decay = decay)

'''
Function for defining a categorical loss function (either sparse or not).
Sparse should be used if labels are expected to be provided as integers.
'''
def categorical_loss(sparse = False):
    if sparse:
        return SparseCategoricalCrossentropy(from_logits = True)
    return CategoricalCrossentropy(from_logits = True)
