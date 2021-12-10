cmake_minimum_required(VERSION 3.0.2)
project(gossip_lr_ros)

## Compile as C++14 (supported in ROS Melodic and newer).
add_compile_options(-std=c++14)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  dataset_msgs
  message_generation
  model_msgs
  models
  rospy
  utils
)

## Uncomment this if the package has a setup.py.
# catkin_python_setup()

################################################
## Declare ROS messages, services and actions ##
################################################

## Generate messages in the 'msg' folder
add_message_files(
  FILES
  Token.msg
  TokenQueue.msg
)

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
)

###################################
## catkin specific configuration ##
###################################

catkin_package(
  CATKIN_DEPENDS dataset_msgs message_runtime model_msgs models rospy utils
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  ${catkin_INCLUDE_DIRS}
)

