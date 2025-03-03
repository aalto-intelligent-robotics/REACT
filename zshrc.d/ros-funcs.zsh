# ---- BEGIN STRETCH 2 -----
ssh_stretch2_hotspot() {
    kitten ssh hello-robot@10.42.0.1
}

ros_ip_stretch2_hotspot() {
    export ROS_IP=10.42.0.10
    export ROS_MASTER_URI=http://10.42.0.1:11311
}

ssh_stretch2_lan() {
    kitten ssh hello-robot@130.233.123.111
}

ros_ip_stretch2_lan() {
    export ROS_IP=130.233.123.110
    export ROS_MASTER_URI=http://130.233.123.111:11311
}
# ---- END STRETCH 2 ----

# ---- ROS1 NOETIC -----
# snoetic() {
#     source /opt/ros/noetic/setup.zsh
# }
# export NOETIC_PRJ="/home/phuoc101/irobotics/projects/ros/noetic_prj"
# source /usr/share/gazebo/setup.sh
# export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH$HOME/.gazebo/models
