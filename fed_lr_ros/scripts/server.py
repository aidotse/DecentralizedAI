#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File       : server.py
# Modified   : 12.11.2021
# By         : Andreas Persson <andreas.persson@ai.se>

import rospy
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from models.model import SequentialModel, ProbabilityModel
from models.model import categorical_loss 
from dataset_msgs.srv import GetDataset
from model_msgs.srv import GetCombinedWeights as GetCombinedWeightsRequest, GetCombinedWeightsResponse
from model_msgs.srv import SetScaledWeights as SetScaledWeightsRequest, SetScaledWeightsResponse
from utils.utils import msg_to_np, msg_to_weights, weights_to_msg

# GPU memory allocation
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)
  
'''
Class for a server in federated learning settings.
'''
class Server:

    def __init__(self):

        # Get private ROS parameters
        model_type = rospy.get_param('~model', 'mlp')
        comm_rounds = rospy.get_param('~comm_rounds', 15)
        self.expected_clients = rospy.get_param('~expected_clients', -1)
        
        # Set global ROS parameters
        self.rounds = 1
        rospy.set_param('/rounds/current', self.rounds)
        rospy.set_param('/rounds/total', comm_rounds)
        
        # Get all test data from dataset
        rospy.loginfo("[Sever::init] Waiting for dataset service...")
        rospy.wait_for_service('/data/request')
        get_dataset = rospy.ServiceProxy('/data/request', GetDataset)
        try:
            
            # Request data
            resp = get_dataset(1.0, 'test')
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
            self.x = self.x[..., tf.newaxis].astype('float32')
            
            # Convert labels to categorical hot vectors
            self.num_classes = resp.num_classes
            self.y = tf.keras.utils.to_categorical(self.y.astype('int32'), num_classes = self.num_classes)

            # Print log
            rospy.loginfo("[Sever::init] Received {} test samples.".format(self.x.shape[0]))

        except rospy.ServiceException as e:
            rospy.logerr("[Server::init] {}".format(e))

        # Initalize global model weights
        self.model = SequentialModel.build( model_type = model_type,
                                            input_shape = self.x.shape[1:],
                                            num_classes = self.num_classes )
        self.weights = self.model.get_weights()

        # Dictionary for storing scaled weights from client(s)
        self.scaled_weights_list = {}

        # List for keeping track of the number of clients (based on client names)
        self.unique_clients = []
            
        # Advertise ROS services for requesting and updating weights
        self.request_srv = rospy.Service('/weights/request', GetCombinedWeightsRequest, self.request)
        self.update_srv = rospy.Service('/weights/update', SetScaledWeightsRequest, self.update)

        # Start ROS loop
        self.run()

    '''
    Remote procedure callback function for requesting global weights from server.
    '''
    def request(self, req):
        try:
            rospy.loginfo("[Server::request] Got request for global weights from client: '{}'.".format(req.name))

            # Add client name to list of unique clients
            if req.name not in self.unique_clients:
                self.unique_clients.append(req.name)

            # Initalize weights list with client name
            self.scaled_weights_list[req.name] = None
            
            # Create response object
            resp = GetCombinedWeightsResponse()
            
            # Add global weights
            if len(self.unique_clients) < self.expected_clients:
                rospy.loginfo("[Server::request] Waiting for {} more client(s) before starting the training...".format(self.expected_clients - len(self.unique_clients)))
            else:
                resp.weights = weights_to_msg(self.weights)

            # Return response to caller
            return resp

        except Exception as e:
            rospy.logerr("[Server::request] {}".format(e))

    '''
    Remote procedure callback function for updating weights list with scaled weights from client(s).
    '''
    def update(self, req):
        try:
            rospy.loginfo("[Server::update] Got scaled weights from client: {}.".format(req.name))

            # Add scaled weights to weights list
            if req.name in self.scaled_weights_list:
                self.scaled_weights_list[req.name] = msg_to_weights(req.weights)

            # Return empty response
            return SetScaledWeightsResponse()

        except Exception as e:
            rospy.logerr("[Server::update] {}".format(e))

    '''
    Controll function for procceding with updating and evaluating global weights.
    '''
    def ready(self):
        if not self.scaled_weights_list:
            return False
        for client in self.scaled_weights_list:
            if self.scaled_weights_list[client] is None:
                return False
        return True
    
    '''
    Function for summarize the list of scaled weights, which is equivalent to calculating the scaled avg of the weights.
    '''
    def average(self):

        # Get the average grad accross all client gradients
        avg = []
        for grad_list_tuple in zip(*self.scaled_weights_list.values()):
            layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0)
            avg.append(layer_mean)
        return avg
    
    ''' 
    Function for evaluate the global model.
    '''
    def evaluate(self, weights):

        # Set weights of global model
        self.model.set_weights(weights)
        
        # Predict logits
        logits = self.model.predict(self.x)

        # Calculate the loss and accuracy
        loss = categorical_loss()(self.y, logits)
        acc = accuracy_score(tf.argmax(logits, axis=1), tf.argmax(self.y, axis=1))

        # Return results
        return acc, loss

    '''
    Initializing ROS loop - keeps running until the node is stopped (or training is done).
    '''
    def run(self):
        rate = rospy.Rate(1) # 1 hz
        while not rospy.is_shutdown():
            if self.ready():

                # Calculate avarage global weights
                self.weights = self.average()
                
                # Evaluatex updated global model
                acc, loss = self.evaluate(self.weights)

                # Print log
                rospy.logwarn("[Server::run] Communication round: {}/{}".format(self.rounds, rospy.get_param('/rounds/total', 15)))
                rospy.logwarn("[Server::run] Global loss: {:.3f}, global accuracy: {:.3f}".format(loss, acc))
                
                # Proceed to next communcation round
                self.rounds += 1
                rospy.set_param('/rounds/current', self.rounds)
                self.scaled_weights_list = {}

            # Check if done with all communcation rounds
            if self.rounds > rospy.get_param('/rounds/total', 15):
                    break

            # Wait for ctrl input
            rate.sleep()

        # Make some predictions to verify the global model (before shutting down)
        self.model = ProbabilityModel(self.model)
        res = self.model(self.x[:10])
        rospy.logwarn("[Server::shutdown] Verify global model...")
        for predictions, labels in zip(list(res), self.y[:10]):
            rospy.logwarn("[Server::shutdown] Prediction: {}, true: {}".format(np.argmax(predictions), np.argmax(labels)))

'''
Main fn
'''
if __name__ == '__main__':
    rospy.init_node('server_node', anonymous=False)
    try:
        node = Server()
    except rospy.ROSInterruptException:
        pass
