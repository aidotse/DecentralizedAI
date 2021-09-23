#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File       : .py
# Modified   : 23.09.2021
# By         : Andreas Persson <andreas.persson@ai.se>

import math
import argparse
import numpy as np
import matplotlib.pyplot as plt

from model import sequential_model, optimizer, categorical_loss
from dataset import load_mnist
from utils import PrivacyCallback

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Function for visualising (plotting) the results 
def visualise(results, title, epochs):
    n = np.arange(1, epochs + 1)
    plt.figure()
    plt.plot(n, results['loss'], label='train_loss')
    plt.plot(n, results['val_loss'], label='val_loss')
    plt.plot(n, results['accuracy'], label='train_accuracy')
    plt.plot(n, results['val_accuracy'], label = 'val_accuracy')
    if 'epsilon' in results.keys():
        plt.plot(n, results['epsilon'], 'o', color='purple', markersize = 3, label = 'epsilon')
        plt.ylabel('Loss/Accuracy/Privacy')
    else:
        plt.ylabel('Loss/Accuracy')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.xticks(n)
    plt.xlim(1, epochs)
    plt.legend(loc='center right')
    plt.grid()
    plt.show()

# Main function.
if __name__ == '__main__':

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument( '--model', type = str, default = 'mlp',
	                 choices = ['mlp', 'cnn'],
	                 help = "type of learning model")
    parser.add_argument( '--opt', type = str, default = 'sgd',
	                 choices = ['sgd', 'adam'],
	                 help = "optimizer used for training")
    parser.add_argument( '--lr',  metavar='N', type = float, default = 0.1,
	                 help = "learning rate for training")
    parser.add_argument( '--epochs', metavar='N', type = int, default = 15,
	                 help = "number of training epochs")
    parser.add_argument( '--batch-size', metavar='N', type = int, default = 32,
	                 help = "training batch size")
    parser.add_argument( '--dp', default = True, action='store_true',
                         help = "using differentially privacy...  ")  
    parser.add_argument('--no-dp', dest='dp', action='store_false', 
                         help = "  ...or not.")
    parser.add_argument( '--l2_norm_clip', metavar='N', type = float, default = 0.9,
	                 help = "clipping norm")
    parser.add_argument( '--noise_mult', metavar='N', type = float, default = 0.9,
	                 help = "ratio of the standard deviation to the clipping norm")
    args = vars(parser.parse_args())
    
    # Load dataset
    x_train, y_train, x_test, y_test, ds_dims = load_mnist()
    
    # Create a model
    model = sequential_model(
        input_shape = ds_dims['input_shape'],
        num_classes = ds_dims['num_classes'],
        model_type = args['model'],
        dp = args['dp'],
        l2_norm_clip = args['l2_norm_clip'],
        noise_multiplier = args['noise_mult']
    )
    model.summary()
    
    # Get optimizer and loss function
    optimizer = optimizer(opt_type = args['opt'], learning_rate = args['lr'])
    loss = categorical_loss(sparse = False)
    
    # Compile the model 
    model.compile( optimizer = optimizer, loss = loss, metrics = ['accuracy'])

    # Define callbacks (only used in the case differentially privacy)
    callbacks = []
    if args['dp']:
        delta = int(math.log10(ds_dims['train_samples'])) + 1
        delta = 10 ** (-delta)
        callbacks.append(
            PrivacyCallback(ds_dims['train_samples'], args['batch_size'], args['noise_mult'], delta)
        )
    
    # Train the model
    history = model.fit( x_train, y_train,
                         validation_data = (x_test, y_test),
                         callbacks = callbacks,
                         epochs = args['epochs'],
                         batch_size = args['batch_size'] )

    # Visualise the result
    if  args['model'] == 'cnn':
        title = "Convolutional Neural Network (CNN)"
    else:
        title = "Multi-Layer Perceptron (MLP)"
    if args['dp']:
        title += " with Differentially Privacy (DP)"
    else:
        title += " without Differentially Privacy (DP)"
    visualise(history.history, title, args['epochs'])
