#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File       : client.py
# Modified   : 10.12.2021
# By         : Andreas Persson <andreas.persson@ai.se>

import rospy
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from models.model import SequentialModel, ProbabilityModel
from models.model import optimizer, categorical_loss
from dataset_msgs.srv import GetDataset
from model_msgs.srv import GetCombinedWeights as GetCombinedWeightsRequest, GetCombinedWeightsResponse
from tokens import TokenHandler
from gossip_lr_ros.msg import TokenQueue
from utils.utils import msg_to_np, msg_to_weights, weights_to_msg

# GPU memory allocation
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

'''
Class for a client in gossip learning settings. 
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
        self.expected_clients = rospy.get_param('~expected_clients', -1)
        
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
            self.train_x, self.train_y = [], []
            for s in resp.dataset:
                x, y = msg_to_np(s)
                self.train_x.append(x)
                self.train_y.append(y)
            resp = get_dataset(portion, "test")
            self.test_x, self.test_y = [], []
            for s in resp.dataset:
                x, y = msg_to_np(s)
                self.test_x.append(x)
                self.test_y.append(y)

            # Converting to arrays
            self.train_x = np.asarray(self.train_x)
            self.train_y = np.asarray(self.train_y)
            self.test_x = np.asarray(self.test_x)
            self.test_y = np.asarray(self.test_y)

            # Normalize and add channel dimension
            self.train_x = self.train_x / 255.
            self.train_x = self.train_x[..., tf.newaxis].astype("float32")
            self.test_x = self.test_x / 255.
            self.test_x = self.test_x[..., tf.newaxis].astype("float32")

            # Convert labels to categorical hot vectors
            self.num_classes = resp.num_classes
            self.train_y = tf.keras.utils.to_categorical(self.train_y.astype("int32"), num_classes = self.num_classes)
            self.test_y = tf.keras.utils.to_categorical(self.test_y.astype("int32"), num_classes = self.num_classes)

            # Print log
            rospy.loginfo("[{}::init] Received {} train samples.".format(self.name, self.train_x.shape[0]))
            rospy.loginfo("[{}::init] Received {} test samples.".format(self.name, self.test_x.shape[0]))

        except rospy.ServiceException as e:
            rospy.logerr("[{}::init] {}".format(self.name, e))
                
        # ROS publisher/subscriber for determine training order
        self.token_pub = rospy.Publisher('/tokens', TokenQueue, queue_size=1)
        rospy.Subscriber('/tokens', TokenQueue, self.callback)
        self.tokens = TokenHandler(self.name.lower(), self.epochs)

        # Advertise ROS services for requesting and updating weights
        self.base_topic = '/weights/request/'
        self.request_srv = rospy.Service(self.base_topic + self.name.lower(), GetCombinedWeightsRequest, self.request)
        
        # Create a local model
        self.model = SequentialModel.build( model_type = model_type,
                                            input_shape = self.train_x.shape[1:],
                                            num_classes = self.num_classes,
                                            dp = dp )
        rospy.loginfo("[{}::init] Model summary: ".format(self.name))
        self.model.summary()
        
        # Compile the model
        self.model.compile(
            optimizer = optimizer(
                learning_rate = self.learning_rate,
                decay = self.learning_rate / self.epochs
            ),
            loss = categorical_loss(sparse = False),
            metrics = ['accuracy']
        )
        
        # Start ROS loop
        self.run()


    '''
    Callback function for handling training order by tokens.
    '''
    def callback(self, msg):
        self.tokens.update(msg.tokens, self.expected_clients)
                            
    '''
    Remote procedure callback function for requesting weights from client.
    '''
    def request(self, req):
        try:
            rospy.loginfo("[{}::request] Got request for weights from client: '{}'.".format(self.name, req.name))
           
            # Create response object
            resp = GetCombinedWeightsResponse()
            
            # Add global weights
            weights = self.model.get_weights()
            resp.weights = weights_to_msg(weights)

            # Return response to caller
            return resp

        except Exception as e:
            rospy.logerr("[{}::request] {}".format(self.name, e))

    '''
    Initializing ROS loop - keeps running until the node is stopped (or training is done).
    '''
    def run(self):
        rate = rospy.Rate(11) # 1 hz
        while not rospy.is_shutdown() and not self.tokens.done():

            # Start the traning if all clients have reached consensus in training order
            if self.tokens.trainable():

                # Check if client is next in training queue
                if self.tokens.next():
                    
                    # Request weights
                    topic = self.tokens.previously
                    if topic:
                        topic = self.base_topic + topic 
                        get_weights = rospy.ServiceProxy(topic, GetCombinedWeightsRequest)
                        try:
                            resp = get_weights(self.name)
                            weights = msg_to_weights(resp.weights)

                            # Set local model weight (to the weight of the global model)
                            self.model.set_weights(weights)
                            
                        except rospy.ServiceException as e:
                            rospy.logerr("[{}::run] {}".format(self.name, e))

                    # Fit client model
                    self.model.fit( self.train_x,
                                    self.train_y,
                                    epochs = 1,
                                    batch_size = self.batch_size )
                    
                    # Predict logits
                    logits = self.model.predict(self.test_x)

                    # Calculate the loss and accuracy
                    loss = categorical_loss()(self.test_y, logits)
                    acc = accuracy_score(tf.argmax(logits, axis=1), tf.argmax(self.test_y, axis=1))
                    rospy.logwarn("[{}::run] Epoch: {}/{}".format(self.name, self.tokens.epoch, self.epochs))
                    rospy.logwarn("[{}::run] Global loss: {:.3f}, global accuracy: {:.3f}".format(self.name, loss, acc))

                    # Pop token from the queue 
                    self.tokens.pop()

            # Wating for clients
            else:
                rospy.loginfo("[{}::run] Waiting for more client(s) before starting the training...".format(self.name))

                    
            # Publish client token
            msg = TokenQueue()
            msg.tokens = self.tokens.tokens
            self.token_pub.publish(msg)
            
            # Wait for ctrl input
            rate.sleep()

        # Make some predictions to verify the model (before shutting down)
        self.model = ProbabilityModel(self.model)
        res = self.model.predict(self.test_x[:10])
        rospy.logwarn("[{}::shutdown] Verify model...".format(self.name))
        for predictions, labels in zip(list(res), self.test_y[:10]):
            rospy.logwarn("[{}::shutdown] Prediction: {}, true: {}".format(self.name, np.argmax(predictions), np.argmax(labels)))
        rospy.logwarn("[{}::shutdown] ----------------".format(self.name))

'''
Main fn
'''
if __name__ == '__main__':
    rospy.init_node('client_node', anonymous = True)
    try:
        node = Client()
    except rospy.ROSInterruptException:
        pass
