# Federated Learning using Robot Operating System (ROS)

This repository constitute a testbed environment for training [TensorFlow](https://www.tensorflow.org/) models in *federated learning* settings using the communication protocols available in the [Robot Operating System](https://www.ros.org/).

### Why using ROS? 

For this particular testbed, ROS is, purely, used to simplify the communication between _server_ and _clients_ while training a joint model in federated settings. This communication is facilitated by the communication protocols available in ROS, which support both message passing, remote services, as well as shared global parameters -- all of which are transparently communicated between devices in a local network configuration.

## Usage

For a simple example of training an MLP model in federated settings using one server and two clients that jointly learn to classify handwritten digits using the _MNIST_ dataset (only supported, for now), simply open a terminal and launch:

```
cd ~/ai_sweden_ws/
source devel/setup.bash
roslaunch fed_lr_ros example.launch
```

__*Note:*__ ROS environment setup (through `source devel/setup.bash`), is required, but only required once for each terminal!

In the example above, both the server and the clients are all launched through the same ROS launch file. The server and the clients can instead be launched separately and with different parameters. For a more advanced example of jointly training a CNN model using one server and two clients, use the following steps:

1. In one terminal, launch the `server` (together with the `data_loader`):
```
cd ~/ai_sweden_ws/
source devel/setup.bash
roslaunch fed_lr_ros server.launch model:=cnn expected_clients:=2 
```

2. In another terminal, launch the first `client`:
```
cd ~/ai_sweden_ws/
source devel/setup.bash
roslaunch fed_lr_ros client.launch model:=cnn name:=client_a  
```

3. In a third terminal, launch the second `client`:
```
cd ~/ai_sweden_ws/
source devel/setup.bash
roslaunch fed_lr_ros client.launch model:=cnn name:=client_b  
```

__*Tip:*__ the example with separate launch files for server and clients can also be launched across several physical devices (assuming that the [ROS network](http://wiki.ros.org/ROS/Tutorials/MultipleMachines) has been configured accordingly).
