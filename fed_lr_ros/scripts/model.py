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

'''
Wrapper class for creating a model 
according to the Functional API.
'''
class SequentialModel:

    # Static method building a sequential model.    
    @staticmethod
    def build(input_shape = (28, 28, 1), num_classes = 10, model_type = 'mlp'):
        inputs = Input(shape = input_shape)
        x = Flatten(input_shape = input_shape)(inputs)
        x = SequentialModel.dense(x, 128)
        x = SequentialModel.dense(x, 64)
        outputs = SequentialModel.dense(x, num_classes)

        # Create and return the model
        model = Model(inputs= inputs, outputs = outputs, name = model_type)    
        return model

    # Static method for creating a dense fully connected module.
    @staticmethod
    def dense(x, dim, activation = 'relu', dropout = 0.0):        
        x = Dense(dim, activation = activation)(x)
        if dropout > 0.0:
            x = Dropout(dropout)(x)
        return x

'''
Function for appending a sofmax layer (and turning a model into a probability model).
'''
def probability_model(model):
    model = tf.keras.Sequential([
        model,
        Softmax()
    ])
    return model

'''
Function for creating an optimizer.
'''
def optimizer(opt_type = 'sgd', learning_rate = 0.01, decay = 0.0, momentum = 0.9):
    if opt_type == 'adam':
        return Adam(learning_rate = learning_rate, decay = decay)
    return SGD(learning_rate = learning_rate, decay = decay, momentum = momentum)

'''
Function for defining a categorical loss function (either sparse or not).
Sparse should be used if labels are expected to be provided as integers.
'''
def categorical_loss(sparse = False):
    if sparse:
        return SparseCategoricalCrossentropy(from_logits = True)
    return CategoricalCrossentropy(from_logits = True)
