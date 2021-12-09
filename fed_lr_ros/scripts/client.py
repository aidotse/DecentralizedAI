#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File       : client.py
# Modified   : 12.11.2021
# By         : Andreas Persson <andreas.persson@ai.se>

import rospy
import numpy as np
import tensorflow as tf
from models.model import SequentialModel
from models.model import optimizer, categorical_loss
from dataset_msgs.srv import GetDataset
from model_msgs.srv import GetCombinedWeights, SetScaledWeights
from utils.utils import msg_to_np, msg_to_weights, weights_to_msg

# GPU memory allocation
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

'''
Class for a client in federated learning settings.
'''
class Client:

    def __init__(self):

        # Get private ROS parameters
        model_type = rospy.get_param('~model', 'mlp')
        dp = rospy.get_param('~dp', False)
        self.learning_rate = rospy.get_param('~learning_rate', 0.01)
        self.epochs = rospy.get_param('~epochs', 1)
        self.batch_size = rospy.get_param('~batch_size', 32)
        portion = rospy.get_param('~dataset_portion', 0.1)

        # Format client name (for output purposes)
        self.name = rospy.get_name()
        self.name = self.name.strip('/').rstrip('node')
        self.name = "".join(w.capitalize() for w in self.name.split('_'))
            
        # Get portion of train data from dataset
        rospy.loginfo("[{}::init] Waiting for dataset service...".format(self.name))
        rospy.wait_for_service('/data/request')
        get_dataset = rospy.ServiceProxy('/data/request', GetDataset)
        try:
            
            # Request data
            resp = get_dataset(portion, "train")
            self.x, self.y = [], []
            for s in resp.dataset:
                x, y = msg_to_np(s)
                self.x.append(x)
                self.y.append(y)

            # Converting to arrays
            self.x = np.asarray(self.x)
            self.y = np.asarray(self.y)

            # Normalize and add channel dimension
            self.x  = self.x / 255.
            self.x = self.x[..., tf.newaxis].astype("float32")

            # Convert labels to categorical hot vectors
            self.num_classes = resp.num_classes
            self.y = tf.keras.utils.to_categorical(self.y.astype("int32"), num_classes = self.num_classes)

            # Print log
            rospy.loginfo("[{}::init] Received {} train samples.".format(self.name, self.x.shape[0]))

        except rospy.ServiceException as e:
            rospy.logerr("[{}::init] {}".format(self.name, e))
        
        # Halt until server remote procedures are available
        rospy.loginfo("[{}::init] Waiting for server services...".format(self.name))
        rospy.wait_for_service('/weights/request')
        rospy.wait_for_service('/weights/update')

        # Get global ROS parameter(s)
        comm_rounds = rospy.get_param('/rounds/total', 15)
        
        # Controll varaible(s)
        self.rounds = 0

        # Create a local model
        self.model = SequentialModel.build( model_type = model_type,
                                            input_shape = self.x.shape[1:],
                                            num_classes = self.num_classes,
                                            dp = dp )
        rospy.loginfo("[{}::init] Model summary: ".format(self.name))
        self.model.summary()

        # Compile the model
        self.model.compile(
            optimizer = optimizer(
                learning_rate = self.learning_rate,
                decay = self.learning_rate / comm_rounds 
            ),
            loss = categorical_loss(sparse = False),
            metrics = ['accuracy']
        )
        
        # Start ROS loop
        self.run()
        
    ''' 
    Function for training the local model.
    '''
    def train(self, weights):

        # Set local model weight (to the weight of the global model)
        self.model.set_weights(weights)

        # Fit local model
        self.model.fit(self.x, self.y, epochs = 1, batch_size = self.batch_size)

        # Clear session to free memory 
        #tf.keras.backend.clear_session()

        # Return updated weights
        return self.model.get_weights()

    ''' 
    Function for scaling weight.
    '''
    def scale(self, weights):

        # Calcualte the scale factor
        scalar = 1.0
        if rospy.has_param('/dataset/total/train'):
            global_cnt = rospy.get_param('/dataset/total/train') * self.batch_size
            local_cnt = self.x.shape[0] * self.batch_size
            scalar = local_cnt / global_cnt

        # Scale weights
        for i in range(len(weights)):
            weights[i] = weights[i] * scalar

        # Return scaled weights
        return weights
        
    '''
    Initializing ROS loop - keeps running until the node is stopped (or training is done).
    '''
    def run(self):
        rate = rospy.Rate(1) # 1 hz
        while not rospy.is_shutdown():

            # Check if server is on next communcation round
            if rospy.has_param('/rounds/current'):
                if self.rounds < rospy.get_param('/rounds/current'):

                    # Request global weights
                    weights = None
                    get_weights = rospy.ServiceProxy('/weights/request', GetCombinedWeights)
                    try:
                        resp = get_weights(self.name)
                        if not resp.weights.data:
                            rate.sleep()
                            continue
                        weights = msg_to_weights(resp.weights)                
                    except rospy.ServiceException as e:
                        rospy.logerr("[{}::run] {}".format(self.name, e))

                    # Train local model with global weights
                    if weights is not None:
                        weights = self.train(weights)

                        # Scale and update local weights
                        weights = self.scale(weights)

                        # Send scaled local weights
                        set_weights = rospy.ServiceProxy('/weights/update', SetScaledWeights)
                        try:
                            set_weights(self.name, weights_to_msg(weights))
                        except rospy.ServiceException as e:
                            rospy.logerr("[{}::run] {}".format(self.name, e))

                        # Proceed to next communcation round
                        self.rounds += 1

            # Check if done with all communcation rounds
            if self.rounds >= rospy.get_param('/rounds/total', 15):
                    break

            # Wait for ctrl input
            rate.sleep()

'''
Main fn
'''
if __name__ == '__main__':
    rospy.init_node('client_node', anonymous=True)
    try:
        node = Client()
    except rospy.ROSInterruptException:
        pass
