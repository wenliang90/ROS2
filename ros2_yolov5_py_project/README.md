## ROS2 YOLOv5 project

### 一、 prepare

#### 1.1 create workspace
```
cd ~
git clone git@github.com:wenliang90/ROS2.git -b ROS2_Humble
sudo mv ROS2 ros2_workspace
cd ~/ros2_workspace
source /opt/ros/humble/setup.bash

colcon build   ## this action will create build and install directory
```

#### 1.2 download data
```
sudo mkdir -p ~/dataset ~/temp

```

### 二、 Run 

#### 2.1 RUN ROS2+yolov5
```
cd ~/ros2_workspace/ros2_yolov5_py_project/src/ros2_yolov5/
ros2 pkg create --build-type ament_python --node-name data_read_node ros2_yolov5

source ~/ros2_workspace/install/setup.bash 
ros2 pkg list (check ros2_yolov5 pkg) 
ros2 run ros2_yolov5 inference_node
ros2 run ros2_yolov5 data_read_node
```

#### 2.2 RUN yolov5_distance
```
cd ~/ros2_workspace/ros2_yolov5_py_project/src/ros2_yolov5_distance/
ros2 pkg create --build-type ament_python --node-name rgbd_publish_node ros2_yolov5_distance 

source ~/ros2_workspace/install/setup.bash 
ros2 pkg list 
ros2 run ros2_yolov5_distance det_dis
ros2 run ros2_yolov5_distance rgbd_pub_node
```

#### 2.3 RUN yolop
```
cd ~/ros2_workspace/ros2_yolov5_py_project/src/ros2_yolop/
ros2 pkg create --build-type ament_python --node-name data_read_node ros2_yolop

source ~/ros2_workspace/install/setup.bash
ros2 pkg list 
ros2 run ros2_yolop inference_node
ros2 run ros2_yolop data_read_node
```