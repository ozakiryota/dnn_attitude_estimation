# dnn_attitude_estimation
## System architecture
![system_architecture](https://user-images.githubusercontent.com/37431972/100202469-b374f400-2f44-11eb-8c3c-c5ff9d42a835.png)
## Installation
### Build locally
If you are going to build it locally, install the requirements and follow the commands below.
#### Requirements
* ROS
* PyTorch
```bash
$ roscd
$ cd src
$ git clone https://github.com/ozakiryota/dnn_attitude_estimation
$ cd ..
$ catkin_make
```
### Build with Docker
If you can use Docker, follow the commands below.
```bash
$ git clone https://github.com/ozakiryota/dnn_attitude_estimation
$ cd dnn_attitude_estimation/docker/nvidia_docker1_kinetic
$ ./build.sh
```
## Usage using 1 camera
### Train DNN
Train the network with [ozakiryota/image_to_gravity](https://github.com/ozakiryota/image_to_gravity).
### Run
```bash
$ roslaunch dnn_attitude_estimation camera_mle_inference.launch
OR
$ cd dnn_attitude_estimation/docker/nvidia_docker1_kinetic
$ ./camera_mle_inference.sh
```
Open another terminal.
```bash
$ roslaunch dnn_attitude_estimation camera_mle_ekf_real.launch
OR
$ cd dnn_attitude_estimation/docker/nvidia_docker1_kinetic
$ ./run.sh
$ roslaunch dnn_attitude_estimation camera_mle_ekf_real.launch
```
### Citation
```bash
Preparing ...
```
## Usage using 4 cameras
### Train DNN
Train the network with [ozakiryota/multi_image_to_gravity](https://github.com/ozakiryota/multi_image_to_gravity).
### Run
```bash
$ roslaunch dnn_attitude_estimation combined_cameras_mle_inference.launch
OR
$ cd dnn_attitude_estimation/docker/nvidia_docker1_kinetic
$ ./combined_cameras_mle_inference.sh
```
Open another terminal.
```bash
$ roslaunch dnn_attitude_estimation combined_cameras_mle_ekf_real.launch
OR
$ cd dnn_attitude_estimation/docker/nvidia_docker1_kinetic
$ ./run.sh
$ roslaunch dnn_attitude_estimation combined_cameras_mle_ekf_real.launch
```
### Citation
```bash
Preparing ...
```
## Usage using LiDAR
### Train DNN
Train the network with [ozakiryota/depth_image_to_gravity](https://github.com/ozakiryota/depth_image_to_gravity).
### Run
```bash
$ roslaunch dnn_attitude_estimation lidar_regression_inference.launch
OR
$ cd dnn_attitude_estimation/docker/nvidia_docker1_kinetic
$ ./lidar_regression_inference.sh
```
Open another terminal.
```bash
$ roslaunch dnn_attitude_estimation lidar_regression_ekf_real.launch
OR
$ cd dnn_attitude_estimation/docker/nvidia_docker1_kinetic
$ ./run.sh
$ roslaunch dnn_attitude_estimation lidar_regression_ekf_real.launch
```
### Citation
```bash
Preparing ...
```
## Usage using 1 camera and LiDAR
### Train DNN
Train the network with [ozakiryota/color_and_depth_image_to_gravity](https://github.com/ozakiryota/color_and_depth_image_to_gravity).
### Run
```bash
$ roslaunch dnn_attitude_estimation lidar_camera_regression_inference.launch
OR
$ cd dnn_attitude_estimation/docker/nvidia_docker1_kinetic
$ ./lidar_camera_regression_inference.sh
```
Open another terminal.
```bash
$ roslaunch dnn_attitude_estimation lidar_camera_regression_ekf_real.launch
OR
$ cd dnn_attitude_estimation/docker/nvidia_docker1_kinetic
$ ./run.sh
$ roslaunch dnn_attitude_estimation lidar_camera_regression_ekf_real.launch
```
### Citation
```bash
Preparing ...
```
<!--
@Inproceedings{ozaki2021,
	author = {尾崎亮太 and 黒田洋司}, 
	title = {風景知識を学習するカメラ-LiDAR-DNNによる自己姿勢推定},
	booktitle = {第26回ロボティクスシンポジア予稿集},
	pages = {249--250},
	year = {2021}
}
-->
## Datasets
Some datasets are available at [ozakiryota/dataset_image_to_gravity](https://github.com/ozakiryota/dataset_image_to_gravity).
## Trained models
Some trained models are available at [dnn_attitude_estimation/keep](https://github.com/ozakiryota/dnn_attitude_estimation/tree/master/keep).
## Related repositories
- [ozakiryota/gyrodometory](https://github.com/ozakiryota/gyrodometry)
- [ozakiryota/msg_conversion](https://github.com/ozakiryota/msg_conversion)
- [ozakiryota/msg_printer](https://github.com/ozakiryota/msg_printer)
- [ozakiryota/velodyne_pointcloud_to_depthimage](https://github.com/ozakiryota/velodyne_pointcloud_to_depthimage)
