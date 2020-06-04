# FROM ros:noetic
FROM osrf/ros:noetic-desktop-full

########## basis ##########
RUN apt-get update && apt-get install -y \
	vim \
	wget \
	unzip \
	git \
	cmake-curses-gui
########## ROS setup ##########
RUN mkdir -p /home/ros_catkin_ws/src && \
	cd /home/ros_catkin_ws/src && \
	/bin/bash -c "source /opt/ros/noetic/setup.bash; catkin_init_workspace" && \
	cd /home/ros_catkin_ws && \
	/bin/bash -c "source /opt/ros/noetic/setup.bash; catkin_make" && \
	echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc && \
	echo "source /home/ros_catkin_ws/devel/setup.bash" >> ~/.bashrc && \
	echo "export ROS_PACKAGE_PATH=\${ROS_PACKAGE_PATH}:/home/ros_catkin_ws" >> ~/.bashrc && \
	echo "export ROS_WORKSPACE=/home/ros_catkin_ws" >> ~/.bashrc && \
	echo "function cmk(){\n	lastpwd=\$OLDPWD \n	cpath=\$(pwd) \n	cd /home/ros_catkin_ws \n	catkin_make \$@ \n	cd \$cpath \n	OLDPWD=\$lastpwd \n}" >> ~/.bashrc
########## CUDA 10.2 ##########
# RUN apt-get update && apt-get install -y --no-install-recommends \
# 	gnupg2 curl ca-certificates && \
#     curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub | apt-key add - && \
# 	echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
# 	echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list && \
# 	apt-get purge --autoremove -y curl && \
# 	rm -rf /var/lib/apt/lists/*
#
# ENV CUDA_VERSION 10.2.89
# ENV CUDA_PKG_VERSION 10-2=$CUDA_VERSION-1
#
# # For libraries in the cuda-compat-* package: https://docs.nvidia.com/cuda/eula/index.html#attachment-a
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     cuda-cudart-$CUDA_PKG_VERSION \
# 	cuda-compat-10-2 && \
# 	ln -s cuda-10.2 /usr/local/cuda && \
# 	rm -rf /var/lib/apt/lists/*
#
# # Required for nvidia-docker v1
# RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
#     echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf
#
# ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
# ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64
#
# # nvidia-container-runtime
# ENV NVIDIA_VISIBLE_DEVICES all
# ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
# ENV NVIDIA_REQUIRE_CUDA "cuda>=10.2 brand=tesla,driver>=384,driver<385 brand=tesla,driver>=396,driver<397 brand=tesla,driver>=410,driver<411 brand=tesla,driver>=418,driver<419"
########## PyTorch ##########
RUN apt-get update && apt-get install -y python3-pip &&\
	pip3 install torch torchvision
########## python ##########
# RUN apt-get update && apt-get install -y \
# 	python3.6
########## dnn_attitude_estimation ##########
##### auto-git-pull #####
ARG CACHEBUST=1
##### git clone #####
RUN	cd /home/ros_catkin_ws/src && \
	git clone https://github.com/ozakiryota/dnn_attitude_estimation &&\
	cd /home/ros_catkin_ws && \
	/bin/bash -c "source /opt/ros/noetic/setup.bash; catkin_make"
##### authorize #####
RUN	chmod +x /home/ros_catkin_ws/src/dnn_attitude_estimation/pysrc/prediction.py
########## nvidia-docker1 hooks ##########
LABEL com.nvidia.volumes.needed="nvidia_driver"
ENV PATH /usr/local/nvidia/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}
######### initial position ##########
WORKDIR /home/ros_catkin_ws
