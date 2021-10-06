# Federated Learning using Robot Operating System (ROS)

This repository constitute a testbed environment for training [TensorFlow](https://www.tensorflow.org/) models in federated settings using the communication protocols available in the [Robot Operating System](https://www.ros.org/).


### What is ROS? 

*“The Robot Operating System (ROS) is a flexible framework for writing robot software. It is a collection of tools, libraries, and conventions that aim to simplify the task of creating complex and robust robot behavior across a wide variety of robotic platforms.”* -- [ROS Webpage](https://www.ros.org/about-ros/)

For this particular testbed is ROS, mainly, used to simplify the communication between _server_ and _clients_ while training a joint model in federated settings. This communication is facilitated by the communication protocols available in ROS, which support both message passing, remote services, as well as shared global parameters -- all of which are transparently communicated between devices in a local network configuration.

