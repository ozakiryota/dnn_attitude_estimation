#!/bin/bash

image_name="dnn_attitude_estimation"
root_path=$(pwd)

xhost +
nvidia-docker run -it --rm \
	--env="DISPLAY" \
	--env="QT_X11_NO_MITSHM=1" \
	--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
	--net=host \
	-v $root_path/../../weights:/home/ros_catkin_ws/src/$image_name/weights \
	-v $root_path/../../pysrc:/home/ros_catkin_ws/src/$image_name/pysrc \
	$image_name:nvidia_docker1 \
	/bin/bash /home/regression_prediction.sh
