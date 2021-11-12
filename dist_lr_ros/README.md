# Distributed Learning using Robot Operating System (ROS)

This repository constitute a testbed environment for training [TensorFlow](https://www.tensorflow.org/) models in *distributed learning* settings using the communication protocols available in the [Robot Operating System](https://www.ros.org/).


### Why using ROS? 

For this particular testbed, ROS is, purely, used to simplify the communication between _workers_ while training a joint model in distributed settings. This communication is facilitated by the communication protocols available in ROS, which support both message passing, remote services, as well as shared global parameters -- all of which are transparently communicated between devices in a local network configuration.


## Usage

To train a joint model using two workers that jointly learning to classify handwritten digits using the _MNIST_ dataset (only supported, for now), simply open a terminal and launch:

```
cd ~/ai_sweden_ws/
source devel/setup.bash
roslaunch dist_lr_ros simple.launch
```

_Note:_ ROS environment setup (through `source devel/setup.bash`), is required, but only required once for each terminal!





