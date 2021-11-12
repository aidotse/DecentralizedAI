# Decentralized AI

This is the repository for the overarching strategic program [Decentralized AI](https://www.ai.se/en/projects-9/decentralized-ai). So far, this repository contains the following sub-folders:

* [__fed_lr_ros__](https://github.com/aidotse/DecentralizedAI/tree/main/fed_lr_ros) -- a testbed for traning TensorFlow models in *federated learning* settings using ROS.

* [__dist_lr_ros__](https://github.com/aidotse/DecentralizedAI/tree/main/dist_lr_ros) -- a general testbed for traning TensorFlow models in *distributed learning* settings using ROS.

* [__common__](https://github.com/aidotse/DecentralizedAI/tree/main/common) -- common ROS mesages, utils, and packages used by both the *distributed learning* and the *federeated learning* testbed.

* [__tf_privacy__](https://github.com/aidotse/DecentralizedAI/tree/main/tf_privacy) -- a rudimentary tutorial for using _differential privacy_ together with TensorFlow 2.x.


## Dependencies

Common for all the packages in this repository is the use of [TensorFlow](https://www.tensorflow.org/) for defining machine learning models (using TensorFlow 2.x, i.e., no use of traditional `tf.Estimators`, etc., whatsoever!). An installing of TensorFlow (>= 2.0) is, therefore, a pre-requisite. You can find detailed instructions [here](https://www.tensorflow.org/install/). For better performance, it is also recommended to install TensorFlow with GPU support (detailed instructions on how to do this are available in the TensorFlow installation documentation). However, in most cases, TensorFlow can simply be installed by:

```
pip install tensorflow
```

...or for GPU support:


```
pip install tensorflow-gpu
```

As the majority of all the packages in this repository also use [ROS](https://www.ros.org/), it is also (highly) recommended installing basic ROS packages. Detailed instructions for installing ROS can be found [here](http://wiki.ros.org/noetic/Installation). However, a basic installation simply be installed by: 

## Installation


## Questions?

__Q:__ What is _differential privacy_?
__A:__ See the tutorial [__tf_privacy__](https://github.com/aidotse/DecentralizedAI/tree/main/tf_privacy).

__Q:__ What is _ROS_? 
__A:__ *“The Robot Operating System (ROS) is a flexible framework for writing robot software. It is a collection of tools, libraries, and conventions that aim to simplify the task of creating complex and robust robot behavior across a wide variety of robotic platforms.”* -- [ROS Webpage](https://www.ros.org/about-ros/)

