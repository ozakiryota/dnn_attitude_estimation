# FROM vistart/cuda:10.0-cudnn7-devel-ubuntu20.04
FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

########## nvidia-docker1 hooks ##########
LABEL com.nvidia.volumes.needed="nvidia_driver"
ENV PATH /usr/local/nvidia/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}
########## non-interactive ##########
## Avoiding "Country of origin for the keyboard" 
## in "apt-get install -y ros-noetic-desktop-full"
ENV DEBIAN_FRONTEND=noninteractive
########## time zone ##########
##### UTC #####
RUN echo 'Etc/UTC' > /etc/timezone && \
    ln -s /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
	apt-get update && \
	apt-get install -q -y --no-install-recommends tzdata && \
	rm -rf /var/lib/apt/lists/*
##### Asia/Tokyo #####
# RUN echo 'Asia/Tokyo' > /etc/timezone && \
#     ln -s /usr/share/zoneinfo/Asia/Tokyo /etc/localtime && \
# 	apt-get update && \
# 	apt-get install -q -y --no-install-recommends tzdata && \
# 	rm -rf /var/lib/apt/lists/*
########## basis ##########
RUN apt-get update && \
	apt-get install -y \
		vim \
		wget \
		unzip \
		git \
		build-essential
########## ROS Noetic insatall ##########
## NOTE: "lsb_release" -> "lsb-release"
RUN apt-get update && apt-get install -y lsb-release &&\
	sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' &&\
	apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654 &&\
	apt-get update && apt-get install -y ros-noetic-desktop-full
########## ROS setup ##########
RUN mkdir -p /home/ros_catkin_ws/src && \
	cd /home/ros_catkin_ws/src && \
	/bin/bash -c "source /opt/ros/noetic/setup.bash; catkin_init_workspace" && \
	cd /home/ros_catkin_ws && \
	/bin/bash -c "source /opt/ros/noetic/setup.bash; catkin_make" && \
	echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc && \
	echo "source /home/ros_catkin_ws/devel/setup.bash" >> ~/.bashrc && \
	echo "export ROS_PACKAGE_PATH=\${ROS_PACKAGE_PATH}:/home/ros_catkin_ws" >> ~/.bashrc && \
	echo "export ROS_WORKSPACE=/home/ros_catkin_ws" >> ~/.bashrc
## cmk
RUN echo "function cmk(){ \n \
		lastpwd=\$OLDPWD \n \
		cpath=\$(pwd) \n \
		cd /home/ros_catkin_ws \n \
		catkin_make \$@ \n \
		cd \$cpath \n \
		OLDPWD=\$lastpwd \n}" >> ~/.bashrc
########## PyTorch ##########
RUN apt-get update && \
	apt-get install -y \
		python3-pip && \
	pip3 install \
		# torch==1.5.0 torchvision==0.6.0 \
		torch==1.8.0+cu111 \
		torchvision==0.9.0+cu111 \
		torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
########## dnn_attitude_estimation ##########
##### NO cache #####
ARG CACHEBUST=1
##### git clone #####
RUN	cd /home/ros_catkin_ws/src && \
	git clone https://github.com/ozakiryota/dnn_attitude_estimation &&\
	cd /home/ros_catkin_ws && \
	/bin/bash -c "source /opt/ros/noetic/setup.bash; catkin_make"
# ##### script #####
# RUN echo "#!/bin/bash \n \
# 		source /opt/ros/noetic/setup.bash; \n \
# 		source /home/ros_catkin_ws/devel/setup.bash; \n \
# 		source /home/catkin_build_ws/install/setup.bash --extend; \n \
# 		roslaunch dnn_attitude_estimation camera_mle_inference.launch" >> /home/camera_mle_inference.launch.sh &&\
# 	chmod +x /home/camera_mle_inference.launch.sh
######### initial position ##########
WORKDIR /home/ros_catkin_ws/src/dnn_attitude_estimation
