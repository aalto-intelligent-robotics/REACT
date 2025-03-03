export DATASET_DIR=${HOME}/irobotics/projects/datasets
# AALTO SERVERS
export GUNDAM_IP=130.233.123.198
export MARVIN_IP=130.233.123.196

# ROS Stuffs
storch_noetic() {
    source ${HOME}/venvs/torch_ros_venv/bin/activate
}

sasymformer_noetic() {
    source ${HOME}/venvs/asymformer_ros/bin/activate
}
