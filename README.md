## Adversarial Driving in ROS

![](doc/adversarial-ros-driving.png)



### Step 1: Launch TurtleBot

```
$ export TURTLEBOT3_MODEL=waffle
$ cd ros_ws
$ catkin_make
$ source devel/setup.sh
$ roslaunch turtlebot3_gazebo turtlebot3_lane_world.launch
```



### Step 2: Collect Data

The following script collects image data from the topic **/camera/rgb/image_raw** and corresponding control command in **/cmd_vel**. The log file is saved  in **driving_log.csv**, and images are saved in **IMG/** folder

```
$ cd model/data
$ # Collect left camera data
$ python3 ros_collect_data.py  --camera left --env gazebo
$ # Collect center camera data
$ python3 ros_collect_data.py  --camera center --env gazebo
$ # Collect right camera data
$ python3 ros_collect_data.py  --camera right --env gazebo
```



### Step 3: Train model

Once the data are collected, we can train a model that tracks the lane.

```
$ cd model
$ python3 model.py
```



### Step 4: Attack

```
$ cd model
$ python3 drive.py
```