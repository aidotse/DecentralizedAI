# Federated Learning using Robot Operating System (ROS)

This repository constitute a testbed environment for training [TensorFlow](https://www.tensorflow.org/) models in *federated learning* settings using the communication protocols available in the [Robot Operating System](https://www.ros.org/).

### Why using ROS? 

For this particular testbed, ROS is, purely, used to simplify the communication between _server_ and _clients_ while training a joint model in federated settings. This communication is facilitated by the communication protocols available in ROS, which support both message passing, remote services, as well as shared global parameters -- all of which are transparently communicated between devices in a local network configuration.

## Usage

To train a joint model using one server and two clients that jointly learning to classify handwritten digits using the _MNIST_ dataset (only supported, for now), simply open a terminal and launch:

```
cd ~/ai_sweden_ws/
source devel/setup.bash
roslaunch fed_lr_ros simple.launch
```

_Note:_ ROS environment setup (through `source devel/setup.bash`), is required, but only required once for each terminal!
