cmake_minimum_required(VERSION 3.16)
project(react)

# Use the FindBoost provided by CMake, rather than the one provided by Boost
# (for CMake >=3.30).
if (${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.30")
    cmake_policy(SET CMP0167 OLD)
endif()


find_package(catkin REQUIRED COMPONENTS
  hydra_seg_ros
  roscpp
  rospy
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_python_setup()
catkin_package()
