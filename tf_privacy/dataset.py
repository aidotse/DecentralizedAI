#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File       : dataset.py
# Modified   : 23.09.2021
# By         : Andreas Persson <andreas.persson@ai.se>

import numpy as np
import tensorflow as tf

'''
Load the MNIST dataset.
'''
def load_mnist(batch_size = 32):
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    
    # Add a channels dimension
    x_train = x_train[..., tf.newaxis].astype("float32")
    x_test = x_test[..., tf.newaxis].astype("float32")

    assert x_train.min() == 0.
    assert x_train.max() == 1.
    assert x_test.min() == 0.
    assert x_test.max() == 1.
    
    # Dataset dimensions
    ds_dims = {}
    ds_dims['train_samples'] = x_train.shape[0]
    ds_dims['test_samples'] = x_test.shape[0]
    ds_dims['input_shape'] = x_train.shape[1:]
    ds_dims['num_classes'] = len(np.unique(y_train))

    # Convert labels to categorical hot vectors
    y_train = tf.keras.utils.to_categorical(y_train.astype("int32"), num_classes = ds_dims['num_classes'])
    y_test = tf.keras.utils.to_categorical(y_test.astype("int32"), num_classes = ds_dims['num_classes'])

    # Batch and shuffle
    #train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(buffer_size=100).batch(batch_size)
    #test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

    # return train_ds, test_ds, ds_dims
    return x_train, y_train, x_test, y_test, ds_dims

# Main function (for testing)
if __name__ == '__main__':

    # Load dataset
    _, _, _, _, ds_dims = load_mnist()
    print("Summary of dataset dimensions:")
    print("  Input shape: {}".format(ds_dims['input_shape']))
    print("  Number of classes: {}".format(ds_dims['num_classes']))
    print("  Train samples: {}".format(ds_dims['train_samples']))
    print("  Test samples: {}".format(ds_dims['train_samples']))
