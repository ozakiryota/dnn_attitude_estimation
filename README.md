# dnn_attitude_estimation
## System architecture
![ekf_system_architecture](https://user-images.githubusercontent.com/37431972/97774314-f4722680-1b99-11eb-9bfe-5967e387925f.png)
## Training the network
The deep neural networks are trained by [ozakiryota/image_to_gravity](https://github.com/ozakiryota/image_to_gravity) or [ozakiryota/multi_image_to_gravity](https://github.com/ozakiryota/multi_image_to_gravity).
### Dataset
Some datasets are available at [ozakiryota/dataset_image_to_gravity](https://github.com/ozakiryota/dataset_image_to_gravity).
## Usage
The following commands are just an example.
### With 1 camera
```bash
$ roscd dnn_attitude_estimation/docker/nvidia_docker1_kinetic
$ ./mle_prediction.sh
```
Open another terminal.
```bash
$ roslaunch dnn_attitude_estimation airsim_mle_ekf.launch
```
### With 4 cameras
```bash
$ roscd dnn_attitude_estimation/docker/nvidia_docker1_kinetic
$ ./combine_mle_prediction.sh
```
Open another terminal.
```bash
$ roslaunch dnn_attitude_estimation real_mle_ekf.launch
```
