#!/bin/bash

image_name="dnn_attitude_estimation"

xhost +
nvidia-docker run -it --rm \
	--env="DISPLAY" \
	--env="QT_X11_NO_MITSHM=1" \
	--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
	--net=host \
	$image_name:latest \
	/bin/bash /home/launch.sh
