# Sourcing ROS workspaces
source /opt/ros/noetic/setup.bash
if [ -f $HOME/catkin_ws/devel/setup.bash ]; then
    source $HOME/catkin_ws/devel/setup.bash
fi
if [ -f $HOME/hydra_ws/devel/setup.bash ]; then
    source $HOME/hydra_ws/devel/setup.bash
fi
