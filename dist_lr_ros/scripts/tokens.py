#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File       : client.py
# Modified   : 12.11.2021
# By         : Andreas Persson <andreas.persson@ai.se>

import rospy
from dist_lr_ros.msg import Token

'''
Class for handle tokens used for determine training order among workers.
'''
class TokenHandler:

    def __init__(self, name, epochs):
        self._name = name
        self._epochs = epochs
        self._tokens = []
        self._trainable = False

    '''
    Class properties used for access of tokens and token attributes.
    '''    
    @property
    def tokens(self):
        return self._tokens

    @property
    def epoch(self):
        if self._tokens:
            return self._tokens[0].epoch
        return None

    @property
    def previously(self):
        if self._tokens:
            return self._tokens[0].previously
        return None

    '''
    Functions for handling the control of training.
    '''
    def done(self):
        if len(self._tokens) == 0 and self._trainable:
            return True
        return False

    def next(self):
        if self._trainable and len(self._tokens) > 0 and self._tokens[0].name == self._name:
            return True
        return False
    
    def pop(self):
        if len(self._tokens) > 0:
            del self._tokens[0]

    def trainable(self):
        return self._trainable
        
    '''
    Function for updating the overall queue of tokens
    '''
    def update(self, tokens):

        # Remove tokens removed by another worker
        self._tokens = [t for t in self._tokens if t.name == self._name or self.__contain(t, tokens)]

        # Add tokens added by another worker
        for t in tokens:
            if t.name != self._name and not self.__contain(t, self.tokens):
                self._tokens.append(t)

        # Append token to overall queue of tokens
        if not self._trainable:
            if len(self._tokens) == 0:
                token = self.__tokenize(epoch = 1)
                self._tokens.append(token)
            else:
                epoch = -1
                for t in self._tokens:
                    if t.name == self._name and t.epoch > epoch:
                        epoch = t.epoch
                if epoch > 0:
                    epoch = epoch + 1
                    if epoch <= self._epochs:
                        token = self.__tokenize(epoch = epoch)
                        self._tokens.append(token)
                else:
                    epoch = self._epochs
                    for t in self._tokens:
                        if t.epoch < epoch:
                            epoch = t.epoch
                    token = self.__tokenize(epoch = epoch)
                    self._tokens.append(token)

        # Sort the queue of tokens according to epochs
        for i in range(0, len(self._tokens) - 1):
            for j in range(i + 1, len(self._tokens)):
                if self._tokens[i].epoch >  self._tokens[j].epoch:
                    self._tokens[i], self._tokens[j] = self._tokens[j], self._tokens[i]

        # Sort the queue of tokens according to timestamps
        for i in range(0, len(self._tokens) - 1):
            for j in range(i + 1, len(self._tokens)):
                if self._tokens[i].epoch == self._tokens[j].epoch:
                    if self._tokens[i].t.to_nsec() > self._tokens[j].t.to_nsec():
                        self._tokens[i], self._tokens[j] = self._tokens[j], self._tokens[i]
        
        # Update the chain to topics
        for i in range(1, len(self._tokens)):
            self._tokens[i].previously = self._tokens[i - 1].name
        if self._tokens and self._tokens[0].epoch == 1:
            self._tokens[0].previously = ""
        
        # Check if worker is trainable
        for t in self._tokens:
            if t.name == self._name and t.epoch == self._epochs:
                self._trainable = True
                        
    '''
    "Private" helper functions for checking if a queue of tokens contians a certain token.
    '''
    def __contain(self, token, tokens):
        for t in tokens:
            if token.name == t.name and token.epoch == t.epoch:
                return True
        return False
 
    '''
    "Private" helper functions for generating a token.
    '''
    def __tokenize(self, epoch):
        token = Token()
        token.epoch = epoch 
        token.name = self._name
        token.t = rospy.Time.now()
        token.previously = ""
        return token
