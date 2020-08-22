# dnn_attitude_estimation
## Training the network
A weight file is genereted by [ozakiryota/image_to_gravity](https://github.com/ozakiryota/image_to_gravity)
## Usage (example)
```bash
$ roscd dnn_attitude_estimation/nvidia_docker1_kinetic
$ ./mle_prediction.sh
```
Open another terminal.
```bash
$ roslaunch dnn_attitude_estimation airsim_mle_ekf.launch
```
