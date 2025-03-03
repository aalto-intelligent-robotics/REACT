ARG CUDA_VERSION=11.7.1

FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-runtime-ubuntu20.04 AS trt
LABEL maintainer="NVIDIA CORPORATION"

ENV TRT_VERSION=10.0.1.6
SHELL ["/bin/bash", "-c"]

#===========================================================================================================================================================================
# non-root username
ARG USERNAME=ros
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create a non-root user
RUN groupadd --gid $USER_GID $USERNAME \
	&& useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME \
	# Add sudo support for the non-root user
	&& apt-get update \
	&& apt-get install -y sudo \
	&& echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
	&& chmod 0440 /etc/sudoers.d/$USERNAME \
	&& rm -rf /var/lib/apt/lists/*

ENV USER=${USERNAME}
ENV TERM=xterm-256color
ENV HOME=/home/${USERNAME}
ENV PATH=${HOME}/.local/bin:/usr/local/bin:${PATH}
RUN echo "for f in ~/.bashrc.d/*.sh; do . \$f; done" >> ${HOME}/.bashrc
#===========================================================================================================================================================================

# Required to build Ubuntu 20.04 without user prompts with DLFW container
ENV DEBIAN_FRONTEND=noninteractive

# Update CUDA signing key
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub

# Install requried libraries
RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository ppa:ubuntu-toolchain-r/test
RUN apt-get update && apt-get install -y --no-install-recommends \
	libcurl4-openssl-dev \
	wget \
	git \
	pkg-config \
	sudo \
	ssh \
	libssl-dev \
	pbzip2 \
	pv \
	bzip2 \
	unzip \
	devscripts \
	lintian \
	fakeroot \
	dh-make \
	build-essential

# Install python3
RUN apt-get install -y --no-install-recommends \
	python3 \
	python3-pip \
	python3-dev \
	python3-wheel &&\
	cd /usr/local/bin &&\
	ln -s /usr/bin/python3 python &&\
	ln -s /usr/bin/pip3 pip;

#===========================================================================================================================================================================
# Install TensorRT
RUN if [ "${CUDA_VERSION:0:2}" = "11" ]; then \
	wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.0.1/tars/TensorRT-10.0.1.6.Linux.x86_64-gnu.cuda-11.8.tar.gz \
	&& tar -xf TensorRT-10.0.1.6.Linux.x86_64-gnu.cuda-11.8.tar.gz \
	&& cp -a TensorRT-10.0.1.6/lib/*.so* /usr/lib/x86_64-linux-gnu \
	&& pip install TensorRT-10.0.1.6/python/tensorrt-10.0.1-cp38-none-linux_x86_64.whl ;\
	elif [ "${CUDA_VERSION:0:2}" = "12" ]; then \
	wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.0.1/tars/TensorRT-10.0.1.6.Linux.x86_64-gnu.cuda-12.4.tar.gz \
	&& tar -xf TensorRT-10.0.1.6.Linux.x86_64-gnu.cuda-12.4.tar.gz \
	&& cp -a TensorRT-10.0.1.6/lib/*.so* /usr/lib/x86_64-linux-gnu \
	&& pip install TensorRT-10.0.1.6/python/tensorrt-10.0.1-cp38-none-linux_x86_64.whl ;\
	else \
	echo "Invalid CUDA_VERSION"; \
	exit 1; \
	fi

# Install PyPI packages
RUN pip3 install --upgrade pip
RUN pip3 install setuptools>=41.0.0
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --default-timeout=1000 --no-cache-dir -r /tmp/requirements.txt

# Download NGC client
RUN cd /usr/local/bin && wget https://ngc.nvidia.com/downloads/ngccli_cat_linux.zip && unzip ngccli_cat_linux.zip && chmod u+x ngc-cli/ngc && rm ngccli_cat_linux.zip ngc-cli.md5 && echo "no-apikey\nascii\n" | ngc-cli/ngc config set

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Set environment and working directory
ENV TRT_LIBPATH=/usr/lib/x86_64-linux-gnu
ENV TRT_OSSPATH=/workspace/TensorRT
ENV PATH="${PATH}:/usr/local/bin/ngc-cli"
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${TRT_OSSPATH}/build/out:${TRT_LIBPATH}"

#===========================================================================================================================================================================
# INSTALL ROS
RUN apt-get install -y curl lsb-release \
	&& curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add 
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu `lsb_release -sc` main" > /etc/apt/sources.list.d/ros-latest.list' \
	&& sh -c 'echo "deb http://packages.ros.org/ros/ubuntu `lsb_release -sc` main" > /etc/apt/sources.list.d/ros-latest.list' \
	&& sh -c 'echo "deb http://packages.ros.org/ros-shadow-fixed/ubuntu `lsb_release -sc` main" > /etc/apt/sources.list.d/ros-shadow.list' \
	&& apt-get update \
	&& apt-get install -y ros-noetic-ros-base

# INSTALL CATKIN
RUN apt-get update \
	&& apt-get install -y python3-osrf-pycommon python3-rosdep \
	python3-catkin-tools python3-vcstool python3-pip git unzip zip wget \
	build-essential \
	&& rm -rf /var/lib/apt/lists/*

# Slam_toolbox + hector + some utils
RUN apt-get update \
	&& apt-get install -y \
	ros-noetic-compressed-image-transport \
	ros-noetic-hector-mapping ros-noetic-hector-trajectory-server \
	ros-noetic-hector-localization ros-noetic-hector-nav-msgs ros-noetic-hector-slam \
	ros-noetic-hector-slam-launch ros-noetic-tf2-sensor-msgs ros-noetic-tf2 \
	ros-noetic-slam-toolbox* \
	libgflags-dev libeigen3-dev git libgoogle-glog-dev \
	python3-pip \
	&& rm -rf /var/lib/apt/lists/*

WORKDIR /home/${USERNAME}

ENV USER=${USERNAME}
ENV TERM=xterm-256color
ENV HOME=/home/${USERNAME}
ENV PATH=${HOME}/.local/bin:/usr/local/bin:${PATH}
RUN echo "for f in ~/.bashrc.d/*.sh; do . \$f; done" >> ${HOME}/.bashrc

#===========================================================================================================================================================================
# HYDRA DEPENDENCIES
RUN apt-get update \
	&& apt-get install -y --no-install-recommends -V libprotobuf-dev protobuf-compiler ros-noetic-rviz-imu-plugin \
	ros-noetic-image-geometry ros-noetic-camera-info-manager \
	ros-noetic-image-transport ros-noetic-image-publisher ros-noetic-rtabmap-ros \
	ros-noetic-py-trees-ros ocl-icd-opencl-dev opencl-headers ros-noetic-ros-controllers \
	libudev-dev ros-noetic-pcl-ros \
	ros-noetic-eigen-conversions ros-noetic-tf-conversions ros-noetic-image-pipeline \
	&& rm -rf /var/lib/apt/lists/*

RUN apt-get update \
	&& apt-get install -y nlohmann-json3-dev libzmqpp-dev \
	&& rm -rf /var/lib/apt/lists/*

RUN chsh -s /usr/bin/bash

#===========================================================================================================================================================================
# Remove this version of cmake to use cmake 3.16 to avoid warning messages when catkin build
# RUN rm /usr/local/bin/cmake
