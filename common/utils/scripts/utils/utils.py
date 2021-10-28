#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File       : utils.py
# Modified   : 28.10.2021
# By         : Andreas Persson <andreas.persson@ai.se>

import numpy as np
import pickle
from dataset_msgs.msg import Sample
from model_msgs.msg import Weights
from std_msgs.msg import MultiArrayDimension

'''
Convert sample data (Numpy) to ROS message
From: https://github.com/neka-nat/ros_np_multiarray
'''
def np_to_msg(x, y):
    s = Sample()
    s.label = y.item()
    s.data.layout.dim = [MultiArrayDimension('dim%d' % i,
                                             x.shape[i],
                                             x.shape[i] * x.dtype.itemsize) for i in range(x.ndim)]
    s.data.data = x.reshape([1, -1])[0].tolist()
    return s

'''
Convert ROS message to sample data (Numpy)
'''
def msg_to_np(msg):
    y = np.uint8(msg.label)
    dims = list(map(lambda x: x.size, msg.data.layout.dim))
    x = [np.uint8(ch) for ch in msg.data.data]
    x = np.array([x]).reshape(dims[0], dims[1])
    return x, y

'''
Convert model weights to ROS message
'''
def weights_to_msg(weights):
    msg = Weights()
    msg.data = pickle.dumps({"weights": weights})
    return msg
                       
'''
Convert ROS message to model weights
'''
def msg_to_weights(msg):
    loaded = pickle.loads(msg.data)
    weights = loaded['weights']
    return weights

