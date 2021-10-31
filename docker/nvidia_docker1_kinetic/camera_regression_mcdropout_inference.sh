#!/bin/bash

image_name="dnn_attitude_estimation"
tag_name="nvidia_docker1_kinetic"
root_path=$(pwd)

xhost +
nvidia-docker run -it --rm \
	--env="DISPLAY" \
	--env="QT_X11_NO_MITSHM=1" \
	--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
	--net=host \
	-v $root_path/../../weights:/home/ros_catkin_ws/src/$image_name/weights \
	-v $root_path/../../pysrc:/home/ros_catkin_ws/src/$image_name/pysrc \
	--env="OMP_NUM_THREADS=1" \
	$image_name:$tag_name \
	bash -c "\
		source /opt/ros/kinetic/setup.bash; \
		source /home/ros_catkin_ws/devel/setup.bash; \
		source /home/catkin_build_ws/install/setup.bash --extend; \
		roslaunch dnn_attitude_estimation camera_regression_mcdropout_inference.launch"
