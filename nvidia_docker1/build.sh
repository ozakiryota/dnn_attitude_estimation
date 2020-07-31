#!/bin/bash

image_name="dnn_attitude_estimation"
docker build . \
	-t $image_name:nvidia_docker1 \
	--build-arg CACHEBUST=$(date +%s)
