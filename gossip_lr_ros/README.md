# Gossip Learning using Robot Operating System (ROS)

This repository constitute a testbed environment for training [TensorFlow](https://www.tensorflow.org/) models in *gossip learning* settings using the communication protocols available in the [Robot Operating System](https://www.ros.org/).


### Why using ROS? 

For this particular testbed, ROS is, purely, used to simplify the communication between _clients_ while training a joint model in gossip learning settings. This communication is facilitated by the communication protocols available in ROS, which support both message passing, remote services, as well as shared global parameters -- all of which are transparently communicated between devices in a local network configuration.


## Usage

For a simple example of training an MLP model using clients that jointly *gossip* and learn to classify handwritten digits using the _MNIST_ dataset (only supported, for now), simply open a terminal and launch:

```
cd ~/ai_sweden_ws/
source devel/setup.bash
roslaunch gossip_lr_ros example.launch
```

__*Note:*__ ROS environment setup (through `source devel/setup.bash`), is required, but only required once for each terminal!

In the example above, both clients are launched through the same ROS launch file. The clients can instead be launched separately and with different parameters. For a more advanced example of jointly training a CNN model using two clients, use the following steps:

1. In a terminal, launch the first `client`:
```
cd ~/ai_sweden_ws/
source devel/setup.bash
roslaunch gossip_lr_ros client.launch model:=cnn expected_clients:=2 name:=client_a
```

2. In another terminal, launch the second `client`:
```
cd ~/ai_sweden_ws/
source devel/setup.bash
roslaunch gossip_lr_ros client.launch model:=cnn expected_clients:=2 name:=client_b
```

3. Finally, in one more terminal, run the `data_loader`:
```
cd ~/ai_sweden_ws/
source devel/setup.bash
rosrun data_loader dataset.py
```

__*Tip:*__ the example with separate clients can also be launched across several physical devices (assuming that the [ROS network](http://wiki.ros.org/ROS/Tutorials/MultipleMachines) has been configured accordingly).
