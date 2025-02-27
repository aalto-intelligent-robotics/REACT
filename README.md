# 🚀🚀🚀 REACT: Real-time Efficient Attribute Clustering and Transfer for Updatable 3D Scene Graph

## 🚀 Overview
![REACT gif](./assets/react_cut.gif)

This repository contains code to cluster object nodes of a 3D scene graph based on their appearance and match them with their correspondences on a reference graph in real-time. This repository is based on:

[REACT: Real-time Efficient Attribute Clustering and Transfer for Updatable 3D Scene Graph](https://arxiv.org)

If you find this code relevant for your work, please consider citing our paper. A bibtex entry is provided below:

## 🙏 Acknowledgements

This work was supported by the Research Council of Finland (decision 354909).

# 🐳 Installation and Running

We highly recommend you to use Docker for deploying REACT. We provide pre-built dockerfiles [here](https://github.com/aalto-intelligent-robotics/REACT-docker)

## 💻 Hardware requirements

We tested REACT on a laptop with an RTX 3070 GPU Mobile.

## 🤖 Requirements

If you do not want to use Docker for some reason, REACT was tested on Ubuntu 20.04 with ROS Noetic installed. Follow the instructions [here](https://wiki.ros.org/ROS/Installation) if you do not yet have it on your system.

## 🧰 Building REACT

If you are using our Dockerfile, start a container:
```bash
docker compose up react -d
docker exec -it react_base bash
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

To test REACT out, you can download 2 scans of a scene. We recommend the *CoffeeRoom* scene and it can be found [here](https://drive.google.com/drive/folders/13984WvqdFPlq2DJG-6iNYHLAHxNozDFg?usp=sharing). Each scene consists of 2 ROS bags with data collected from a Hello Stretch 2 robot, with an Orbbec Astra 2 RGB-D Camera pointing 10 degrees downwards. We use the SLAM toolbox + RP-LiDAR on the Stretch to provide the robot's pose.

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
