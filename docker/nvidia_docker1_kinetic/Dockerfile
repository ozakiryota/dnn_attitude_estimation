# FROM nvidia/cuda:8.0-cudnn7-devel-ubuntu16.04
FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

########## nvidia-docker1 hooks ##########
LABEL com.nvidia.volumes.needed="nvidia_driver"
ENV PATH /usr/local/nvidia/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}
########## BASIS ##########
RUN apt-get update && apt-get install -y \
	vim \
	wget \
	unzip \
	git \
	build-essential
########## ROS kinetic insatall ##########
## NOTE: "lsb_release" -> "lsb-release"
RUN apt-get update && \
	apt-get install -y \
		lsb-release && \
	sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' && \
	apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654 && \
	apt-get update && \
	apt-get install -y \
		ros-kinetic-ros-base
########## ROS setup ##########
RUN mkdir -p /home/ros_catkin_ws/src && \
	cd /home/ros_catkin_ws/src && \
	/bin/bash -c "source /opt/ros/kinetic/setup.bash; catkin_init_workspace" && \
	cd /home/ros_catkin_ws && \
	/bin/bash -c "source /opt/ros/kinetic/setup.bash; catkin_make" && \
	echo "source /opt/ros/kinetic/setup.bash" >> ~/.bashrc && \
	echo "source /home/ros_catkin_ws/devel/setup.bash" >> ~/.bashrc && \
	echo "export ROS_PACKAGE_PATH=\${ROS_PACKAGE_PATH}:/home/ros_catkin_ws" >> ~/.bashrc && \
	echo "export ROS_WORKSPACE=/home/ros_catkin_ws" >> ~/.bashrc && \
	echo "function cmk(){\n lastpwd=\$OLDPWD \n cpath=\$(pwd) \n cd /home/ros_catkin_ws \n catkin_make \$@ \n cd \$cpath \n OLDPWD=\$lastpwd \n}" >> ~/.bashrc
########## PyTorch & Python3 on rospy ##########
RUN apt-get update && \
	apt-get install -y \
		python3-pip && \
	pip3 install --upgrade pip && \
	pip3 install \
		# torch==1.0.0 \
		# torchvision==0.2.1 \
		torch==1.1.0 \
		torchvision==0.3.0 \
		rospkg \
		catkin_pkg
########## cv_bridge on Python3 ##########
##### cv2 on Python2->3 #####
RUN apt-get update && \
	apt-get install -y \
		python-catkin-tools \
		python3-dev \
		python3-catkin-pkg-modules \
		python3-numpy \
		python3-yaml \
		ros-kinetic-cv-bridge && \
	pip3 install \
		opencv-python && \
	mkdir -p /home/catkin_build_ws/src && \
	cd /home/catkin_build_ws && \
	catkin init && \
	catkin config -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPYTHON_INCLUDE_DIR=/usr/include/python3.5m -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.5m.so && \
	catkin config --install && \
	cd /home/catkin_build_ws/src && \
	git clone https://github.com/ros-perception/vision_opencv.git && \
	cd vision_opencv && \
	git checkout 1.12.8 && \
	ln -s \
		/usr/lib/x86_64-linux-gnu/libboost_python-py35.so \
		/usr/lib/x86_64-linux-gnu/libboost_python3.so && \
	cd /home/catkin_build_ws && \
	/bin/bash -c "source /opt/ros/kinetic/setup.bash; catkin build cv_bridge" && \
	/bin/bash -c "source install/setup.bash --extend" && \
	echo "source /home/catkin_build_ws/install/setup.bash --extend" >> ~/.bashrc && \
	rm /opt/ros/kinetic/lib/python2.7/dist-packages/cv2.so
########## dnn_attitude_estimation ##########
##### Requirement #####
RUN apt-get update && \
	apt-get install -y \
		ros-kinetic-tf \
		python3-tk && \
	pip3 install \
		matplotlib
##### NO cache #####
ARG CACHEBUST=1
##### git clone #####
RUN	cd /home/ros_catkin_ws/src && \
	git clone https://github.com/ozakiryota/dnn_attitude_estimation &&\
	cd /home/ros_catkin_ws && \
	/bin/bash -c "source /opt/ros/kinetic/setup.bash; catkin_make"
# ##### script #####
# RUN echo "#!/bin/bash \n \
# 		source /opt/ros/kinetic/setup.bash; \n \
# 		source /home/ros_catkin_ws/devel/setup.bash; \n \
# 		source /home/catkin_build_ws/install/setup.bash --extend; \n \
# 		roslaunch dnn_attitude_estimation regression_prediction.launch" >> /home/regression_prediction.sh &&\
# 	chmod +x /home/regression_prediction.sh
######### initial position ##########
WORKDIR /home/ros_catkin_ws/src/dnn_attitude_estimation
