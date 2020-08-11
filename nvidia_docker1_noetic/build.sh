#!/bin/bash

image_name="dnn_attitude_estimation"
tag_name="nvidia_docker1_noetic"

docker build . \
	-t $image_name:$tag_name \
	--build-arg CACHEBUST=$(date +%s)
