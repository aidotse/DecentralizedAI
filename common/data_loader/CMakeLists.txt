cmake_minimum_required(VERSION 3.0.2)
project(data_loader)

## Compile as C++14 (supported in ROS Melodic and newer).
add_compile_options(-std=c++14)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  dataset_msgs
  rospy
  utils
)

## Uncomment this if the package has a setup.py.
# catkin_python_setup()

###################################
## catkin specific configuration ##
###################################

catkin_package(
  CATKIN_DEPENDS dataset_msgs rospy utils
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  ${catkin_INCLUDE_DIRS}
)

