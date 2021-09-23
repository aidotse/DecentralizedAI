#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File       : utils.py
# Modified   : 23.09.2021
# By         : Andreas Persson <andreas.persson@ai.se>

import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy_lib

'''
Customized Keras callback class.

Used for computing the privacy budget expended. 
'''
class PrivacyCallback(Callback):

    def __init__(self, train_samples, batch_size, noise_multiplier, delta = 1e-5):
        self.train_samples = train_samples
        self.batch_size = batch_size
        self.noise_multiplier = noise_multiplier
        self.delta = delta
        pass

    # Append current epsilon to log history
    def on_epoch_end(self, epoch, logs = None):
        eps, _ = compute_dp_sgd_privacy_lib.compute_dp_sgd_privacy(
            self.train_samples,
            self.batch_size,
            self.noise_multiplier,
            epoch + 1,
            self.delta
        )
        if 'epsilon' not in self.model.history.history.keys():
            self.model.history.history['epsilon'] = []
        self.model.history.history['epsilon'].append(eps)

