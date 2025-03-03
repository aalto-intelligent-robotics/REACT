# Set ROS_IP and ROS_MASTER_URI
ros_ip_stretch2_hotspot () {
    export ROS_IP=10.42.0.10
    export ROS_MASTER_URI=http://10.42.0.1:11311
    echo "Setting: ROS_IP=$ROS_IP | ROS_MASTER_URI=$ROS_MASTER_URI"
}

ros_ip_stretch2_lan () {
    export ROS_IP=130.233.123.110
    export ROS_MASTER_URI=http://130.233.123.111:11311
    echo "Setting: ROS_IP=$ROS_IP | ROS_MASTER_URI=$ROS_MASTER_URI"
}

# Sourcing ROS workspaces
source /opt/ros/noetic/setup.zsh
if [ -f $HOME/catkin_ws/devel/setup.zsh ]; then
    source $HOME/catkin_ws/devel/setup.zsh
fi
if [ -f $HOME/hydra_ws/devel/setup.zsh ]; then
    source $HOME/hydra_ws/devel/setup.zsh
fi
if [ -f $HOME/superglue_ws/devel/setup.zsh ]; then
    source $HOME/superglue_ws/devel/setup.zsh
fi
