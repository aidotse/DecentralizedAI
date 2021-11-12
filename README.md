# Decentralized AI

This is the repository for the overarching strategic program [Decentralized AI](https://www.ai.se/en/projects-9/decentralized-ai).  
So far, this repository contains the following sub-folders:

* [__fed_lr_ros__](https://github.com/aidotse/DecentralizedAI/tree/main/fed_lr_ros): a testbed for traning TensorFlow models in *federated learning* settings using ROS.

* [__dist_lr_ros__](https://github.com/aidotse/DecentralizedAI/tree/main/dist_lr_ros): a general testbed for traning TensorFlow models in *distributed learning* settings using ROS.

* [__common__](https://github.com/aidotse/DecentralizedAI/tree/main/common): common ROS mesages, utils, and packages used by both the *distributed learning* and the *federeated learning* testbed.

* [__tf_privacy__](https://github.com/aidotse/DecentralizedAI/tree/main/tf_privacy): a rudimentary tutorial for using _differential privacy_ together with TensorFlow 2.x.


## Dependencies

Common for all the packages in this repository is the use of [TensorFlow](https://www.tensorflow.org/) for defining machine learning models (using TensorFlow 2.x, i.e., no use of traditional `tf.Estimators`, etc., whatsoever!). An installing of TensorFlow (>= 2.0) is, therefore, a pre-requisite. You can find detailed instructions [here](https://www.tensorflow.org/install/). For better performance, it is also recommended to install TensorFlow with GPU support (detailed instructions on how to do this are available in the TensorFlow installation documentation). However, in most cases, TensorFlow can simply be installed by:

```
pip install tensorflow
```

...or for GPU support:


```
pip install tensorflow-gpu
```

As the majority of all the packages in this repository use [ROS](https://www.ros.org/), it is also a pre-requisite to install basic ROS packages. Detailed instructions for installing ROS can be found [here](http://wiki.ros.org/noetic/Installation). However, a basic ROS installation for Ubuntu LTS (20.04) can, simply, be installed through standard package manager by the following steps:

1. Add software from `packages.ros.org` to the package:
```
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
```

2. Add authentication keys:
```
sudo apt install curl # if you haven't already installed curl
curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
```

3. Install basic ROS packages:
```
sudo apt update
sudo apt install ros-noetic-ros-base
```

4. Setup ROS environment:
```
source /opt/ros/noetic/setup.bash
```

5. Install dependencies for building packages and initialize `rosdep`:
```
sudo apt install python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential python3-rosdep
sudo rosdep init
rosdep update
```


## Installation

Provided that all dependencies above have been installed, the packages in this repository can be installed through a ROS [catkin workspace]( http://wiki.ros.org/catkin/Tutorials/create_a_workspace). First, make sure that the ROS environment is setup correctly:   
```
source /opt/ros/noetic/setup.bash
```

Next, create and build a catkin workspace:
```
mkdir -p ~/ai_sweden_ws/src
cd ~/ai_sweden_ws/
catkin_make
```

Finally, change the ROS environment to the newly created workspace, clone the code the code from this repository, and compile all packages:
```
source devel/setup.bash
cd src
git clone https://github.com/aidotse/DecentralizedAI.git
cd ..
catkin_make
```

_Installation done!_

_From hereon, see each separate package for usage._ 


## Questions?

__Q:__ What is _differential privacy_? \
__A:__ See the tutorial [__tf_privacy__](https://github.com/aidotse/DecentralizedAI/tree/main/tf_privacy).

__Q:__ What is _ROS_? \
__A:__ *“The Robot Operating System (ROS) is a flexible framework for writing robot software. It is a collection of tools, libraries, and conventions that aim to simplify the task of creating complex and robust robot behavior across a wide variety of robotic platforms.”* -- [ROS Webpage](https://www.ros.org/about-ros/)

