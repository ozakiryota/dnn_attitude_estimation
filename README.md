# dnn_attitude_estimation
## System architecture
![system_architecture](https://user-images.githubusercontent.com/37431972/100202469-b374f400-2f44-11eb-8c3c-c5ff9d42a835.png)
## Training the network
The deep neural networks are trained by ...
* [ozakiryota/image_to_gravity](https://github.com/ozakiryota/image_to_gravity)  
or
* [ozakiryota/multi_image_to_gravity](https://github.com/ozakiryota/multi_image_to_gravity)  
or
* [ozakiryota/depth_image_to_gravity](https://github.com/ozakiryota/depth_image_to_gravity)  
or
* [ozakiryota/color_and_depth_image_to_gravity](https://github.com/ozakiryota/color_and_depth_image_to_gravity)
### Dataset
Some datasets are available at [ozakiryota/dataset_image_to_gravity](https://github.com/ozakiryota/dataset_image_to_gravity).
## Usage
The following commands are just an example.
### With a camera
```bash
$ roscd dnn_attitude_estimation/docker/nvidia_docker1_kinetic
$ ./camera_mle_inference.sh
```
Open another terminal.
```bash
$ roslaunch dnn_attitude_estimation camera_mle_ekf_airsim.launch
```
### With 4 cameras
```bash
$ roscd dnn_attitude_estimation/docker/nvidia_docker1_kinetic
$ ./combined_cameras_mle_inference.sh
```
Open another terminal.
```bash
$ roslaunch dnn_attitude_estimation combined_cameras_mle_ekf_real.launch
```
### With a LiDAR
```bash
$ roscd dnn_attitude_estimation/docker/nvidia_docker1_kinetic
$ ./lidar_regression_inference.sh
```
Open another terminal.
```bash
$ roslaunch dnn_attitude_estimation lidar_regression_ekf_real.launch

```
### With a camera and a LiDAR
```bash
$ roscd dnn_attitude_estimation/docker/nvidia_docker1_kinetic
$ ./lidar_camera_regression_inference.sh
```
Open another terminal.
```bash
$ roslaunch dnn_attitude_estimation lidar_camera_regression_ekf_airsim.launch

```
## Related repositories
- [ozakiryota/gyrodometory](https://github.com/ozakiryota/gyrodometry)
- [ozakiryota/msg_conversion](https://github.com/ozakiryota/msg_conversion)
- [ozakiryota/msg_printer](https://github.com/ozakiryota/msg_printer)
