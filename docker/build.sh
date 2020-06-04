#!/bin/bash

image_name="dnn_attitude_estimation"
docker build . \
	-t $image_name:latest \
	--build-arg CACHEBUST=$(date +%s)
