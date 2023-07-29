## Adversarial Driving in ROS

> Attacking End-to-End Autonomous Driving Systems


[[ Talk ]](https://driving.wuhanstudio.uk) [[ Video ]](https://youtu.be/I0i8uN2oOP0) [[ Code ]](https://github.com/wuhanstudio/adversarial-ros-driving) [[ Paper ]](https://arxiv.org/abs/2103.09151)

The behaviour of end-to-end autonomous driving model can be manipulated by adding unperceivable perturbations to the input image.

[![](doc/adversarial-ros-driving.png)](https://driving.wuhanstudio.uk)


### Quick Start

#### Step 0: Prerequisites

```
$ sudo apt install ros-noetic-desktop-full
$ sudo apt install ros-noetic-rosbridge-suite ros-noetic-turtlebot3-simulations ros-noetic-turtlebot3-gazebo ros-noetic-teleop-twist-keyboard

$ git clone https://github.com/wuhanstudio/adversarial-ros-driving
$ cd adversarial-ros-driving
```

#### Step 1: Setup the TurtleBot

```
$ git clone https://github.com/wuhanstudio/adversarial-driving
$ cd adversarial-driving

$ cd ros_ws
$ rosdep install --from-paths src --ignore-src -r -y

$ catkin_make
$ source devel/setup.sh
$ export TURTLEBOT3_MODEL=waffle
$ roslaunch turtlebot3_lane turtlebot3_lane.launch

# You may need to put the turtlebot on track first
# roslaunch turtlebot3_teleop turtlebot3_teleop_key.launch
```

#### Step 2: Setup the server

```
$ cd model/

$ # CPU
$ conda env create -f environment.yml
$ conda activate adversarial-driving

$ # GPU
$ conda env create -f environment_gpu.yml
$ conda activate adversarial-gpu-driving

$ # If you use anaconda as your defaut python3 environment
$ pip3 install catkin_pkg empy defusedxml numpy twisted autobahn tornado pymongo pillow service_identity

$ roslaunch rosbridge_server rosbridge_websocket.launch

# For real turtlebot3
$ python3 drive.py --env turtlebot --model model_turtlebot.h5

# For Gazebo Simulator
$ python3 drive.py --env gazebo --model model_gazebo.h5
```

The web page will be available at: http://localhost:8080/

<img src="./doc/client.png"  width="100%"/>

That's it!

### Training the model

#### Step 1: Collect the Data

The following script collects image data from the topic **/camera/rgb/image_raw** and corresponding control command in **/cmd_vel**. The log file is saved  in **driving_log.csv**, and images are saved in **IMG/** folder

```
$ cd model/data

$ # Collect left camera data
$ python3 line_follow.py --camera left --env gazebo
$ python3 ros_collect_data.py --camera left --env gazebo

$ # Collect center camera data
$ python3 line_follow.py --camera center --env gazebo
$ python3 ros_collect_data.py --camera center --env gazebo

$ # Collect right camera data
$ python3 line_follow.py --camera right --env gazebo
$ python3 ros_collect_data.py --camera right --env gazebo
```

#### Step 2: Train the model

Once the data is collected, we can train a model that tracks the lane.

```
$ cd model
$ python3 model.py
```

## Adversarial Driving

We also tested our attacks in Udacity autonomous driving simulator. 

https://github.com/wuhanstudio/adversarial-driving

[![](https://raw.githubusercontent.com/wuhanstudio/adversarial-driving/master/doc/adversarial-driving.png)](https://github.com/wuhanstudio/adversarial-driving)


## Citation

```
@INPROCEEDINGS{han2023driving,
  author={Wu, Han and Yunas, Syed and Rowlands, Sareh and Ruan, Wenjie and Wahlström, Johan},
  booktitle={2023 IEEE Intelligent Vehicles Symposium (IV)}, 
  title={Adversarial Driving: Attacking End-to-End Autonomous Driving}, 
  year={2023},
  volume={},
  number={},
  pages={1-7},
  doi={10.1109/IV55152.2023.10186386}
}
```
