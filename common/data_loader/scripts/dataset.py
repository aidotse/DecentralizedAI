#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File       : dataset.py
# Modified   : 28.10.2021
# By         : Andreas Persson <andreas.persson@ai.se>

import numpy as np
import tensorflow as tf

import rospy
from dataset_msgs.srv import GetDataset as GetDatasetRequest, GetDatasetResponse
from utils.utils import np_to_msg

'''
Wrapper class for loading datasets.
'''
class Dataset:

    def __init__(self):

        # Load the dataset
        self.load()
        self.used_train_data, self.used_test_data = 0.0, 0.0
        
        # Advertise ROS services for requesting data
        self.data_srv = rospy.Service("/data/request", GetDatasetRequest, self.request)
        
        # Set global ROS parameters
        rospy.set_param('/dataset/total/train', 0)
        rospy.set_param('/dataset/total/test', 0)
        
        # Start ROS loop (keeps running until the node is stopped)
        rospy.spin()        

    '''
    Function for loading the MNIST dataset.
    '''
    def load(self, shuffle = True):
        rospy.loginfo("[Dataset::load] Start loading dataset...")
        mnist = tf.keras.datasets.mnist
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data() 
        if shuffle:
            self.x_train, self.y_train = self.shuffle(self.x_train, self.y_train)
            self.x_test, self.y_test = self.shuffle(self.x_test, self.y_test)
        rospy.loginfo("[Dataset::load] Dataset loaded.")

    '''
    Function for shuffle a dataset.
    '''
    def shuffle(self, x, y):
        assert len(x) == len(y)
        indices = np.random.permutation(len(x))
        return x[indices], y[indices]

    '''
    Remote procedure callback function for requesting a portion of the dataset.
    '''
    def request(self, req):
        try:
            rospy.loginfo("[Dataset::request] Got request for {:.1f} % of the '{}' dataset.".format(req.percentage * 100.0, req.part))
            
            # Create response object
            resp = GetDatasetResponse()
            resp.num_classes = len(np.unique(self.y_train))
            
            # Add training data 
            if req.part == "train":
                start = int(len(self.x_train) * self.used_train_data)
                self.used_train_data += req.percentage
                if self.used_train_data >= 1.0:
                    self.used_train_data = 1.0
                    rospy.logwarn("[Dataset::request] All training data used.")                    
                stop = int(len(self.x_train) * self.used_train_data)
                for i in range(start, stop):
                    resp.dataset.append(np_to_msg(self.x_train[i], self.y_train[i]))
                rospy.set_param('/dataset/total/train', stop)
                
            # Add validation data 
            else:
                start = int(len(self.x_test) * self.used_test_data)
                self.used_test_data += req.percentage
                if self.used_test_data >= 1.0:
                    self.used_test_data = 1.0
                    rospy.logwarn("[Dataset::request] All test data used.")
                stop = int(len(self.x_test) * self.used_test_data)
                for i in range(start, stop):
                    resp.dataset.append(np_to_msg(self.x_test[i], self.y_test[i]))
                rospy.set_param('/dataset/total/test', stop)

            # Return response to caller
            return resp
        
        except Exception as e:
            rospy.logerr("[Dataset::request] {}".format(e))
        
'''
Main fn
'''
if __name__ == '__main__':
    rospy.init_node('data_loader_node', anonymous=True)
    try:
        node = Dataset()
    except rospy.ROSInterruptException:
        pass
