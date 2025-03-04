# 🚀🚀🚀 REACT: Real-time Efficient Attribute Clustering and Transfer for Updatable 3D Scene Graph

## 🚀 Overview

![REACT gif](./assets/react_cut.gif)

This repository contains code to cluster object nodes of a 3D scene graph based on their appearance and match them with their correspondences on a reference graph in real-time. This repository is based on:

[REACT: Real-time Efficient Attribute Clustering and Transfer for Updatable 3D Scene Graph](https://arxiv.org)

If you find this code relevant for your work, please consider citing our paper. A bibtex entry is provided below:

## 🙏 Acknowledgements

This work was supported by the Research Council of Finland (decision 354909).

## 🤖 Requirements

We tested REACT on a laptop with an RTX 3070 GPU Mobile.

🐳 We highly recommend using Docker for deploying REACT. We provide pre-built Dockerfiles [here](https://github.com/aalto-intelligent-robotics/REACT-docker)

If you do not want to use Docker for some reason, REACT was tested on Ubuntu 20.04 with ROS Noetic installed. Follow the instructions [here](https://wiki.ros.org/ROS/Installation) if you do not yet have it on your system.

> **:warning: Warning**<br>
> We currently have no plan to support other platform. Issues requesting support on other platforms (e.g., Ubuntu 18.04, Windows) will be automatically closed. If you are not on Ubuntu 20.04, we highly recommend using our Docker image.

## 🐳 Setting up with Docker

Clone the Docker repo and build the image:

```bash
git clone https://github.com/aalto-intelligent-robotics/REACT-docker.git
cd REACT-docker/
docker compose build base
```

Go grab yourself a coffee because this can take a while ☕

To start a container:

```bash
docker compose up react -d
docker exec -it react_base bash
```

## 🧰 Building REACT

If you do not have vcs tool install, install it with:

```bash
pip3 install vcstool
# OR
sudo apt install python3-vcstool
```

Clone all the required repositories:

```bash
cd REACT-docker/
mkdir -p hydra_ws/src  # ROS workspace
vcs import ./hydra_ws/src < ./react.rosinstall
```

Create these directories:

- `logs`: For REACT's output (debugging)
- `dsg_output`: For 3d scene graph output of modified Hydra
- `models`: For ML models
- `bags`: Put your ROS bags here (or create a symlink)

```bash
mkdir logs/ dsg_output/ models/ bags/
```

Inside the container, build all packages by running:

```bash
cd hydra_ws
catkin config -DCMAKE_BUILD_TYPE=Release
cd src
rosdep install --from-paths . --ignore-src -r -y
cd ..
catkin build -j 4  # Adjust depending on how many cores your CPU have
```

## 🔥 Quickstart

To test REACT out, you can download 2 scans of a scene. We recommend the *CoffeeRoom* scene and it can be found [here](https://drive.google.com/drive/folders/13984WvqdFPlq2DJG-6iNYHLAHxNozDFg?usp=sharing). Each scene consists of 2 ROS bags with data collected from a Hello Stretch 2 robot, with an Orbbec Astra 2 RGB-D Camera pointing 10 degrees downwards. Put the saved ROS bags in `REACT-docker/bags/` We use the SLAM toolbox + RP-LiDAR on the Stretch to provide the robot's pose.

To start Hydra:

```bash
roslaunch hydra_stretch hydra_stretch_yolo.launch slam_mode:=slam dsg_output_prefix:=<scene_graph_name>_1 2> >(grep -v 'TF_REPEATED_DATA\|at line 278\|buffer_core')
```

Then, start playing the ROS bag in a separate terminal:

```bash
rosbag play --clock path/to/rosbag
```

After the Hydra has built the map, save it by going to another terminal and run:

```bash
rosservice call /hydra_ros_node/save_graph
```

If you are using our Dockerfile, it the graph should be saved in `/home/ros/dsg_output`.

Unlike the original work, our version of Hydra register object nodes on an instance level, and we use an instance segmentation model (YOLO11) to retrieve the segmentation (the ROS repo for it is [here](https://github.com/aalto-intelligent-robotics/Hydra-Seg-ROS)). Along with the scene graph, we also save the instance views (binary masks of the objects) and the map views (image of the scene).

### 🧠 Embedding Model training

TODO

### 😎 Online Matching

If you want to match the 3D scene graphs online, build the second graph with:

```bash
roslaunch hydra_stretch hydra_stretch_yolo.launch slam_mode:=localization dsg_output_prefix:=<scene_graph_name>_2 2> >(grep -v 'TF_REPEATED_DATA\|at line 278\|buffer_core')
```

Run REACT (NOTE: wait for it to finish clustering the nodes of the first graph):

```bash
rosrun react_ros react_ros_node.py -s <scene_graph_name>
```

After the first graph is clustered (there should be a console message: "Loaded graph..."), run the ROS bag:

```bash
rosbag play --clock path/to/2nd/rosbag
```

You should see the scene graphs being matched like the GIF at the [beginning](#overview) of this repo.

### 🤓 Offline Matching

If you want to match the 3D scene graphs offline, build the second graph with:

```bash
roslaunch hydra_stretch hydra_stretch_yolo.launch slam_mode:=localization dsg_output_prefix:=<scene_graph_name>_2 2> >(grep -v 'TF_REPEATED_DATA\|at line 278\|buffer_core')
```

Then, start playing the ROS bag in a separate terminal:

```bash
rosbag play --clock path/to/2nd/rosbag
```

With 2 graphs built, perform offline matching by going to `app/` and run:

```bash
python3 offline_matching.py -s <scene_graph_name>
```

## 📈 REACT Evaluation

With 2 graphs built, you can evaluate REACT's performance vs a non-clustering configuration by running:

```bash
python3 offline_eval.py -s <scene_graph_name>
```
