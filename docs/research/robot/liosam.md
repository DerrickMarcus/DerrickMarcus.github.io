---
comments: true
---

# Deploy LIO-SAM

## Reproduce the project 复现项目

LIO-SAM 项目地址：<https://github.com/TixiaoShan/LIO-SAM>

创建工作空间：

```bash
# in home directory
mkdir -p liosam_ws/src
cd liosam_ws/src
catkin_init_workspace

# clone the repo
git clone https://github.com/TixiaoShan/LIO-SAM.git
git clone git@github.com:TixiaoShan/LIO-SAM.git

# compile the source
cd ..
catkin_make
```

安装依赖的软件包：

```bash
sudo add-apt-repository ppa:borglab/gtsam-release-4.1
sudo apt install libgtsam-dev libgtsam-unstable-dev

sudo apt-get install -y ros-noetic-navigation
sudo apt-get install -y ros-noetic-robot-localization
sudo apt-get install -y ros-noetic-robot-state-publisher
```

在 Ubuntu 20.04 + ROS Noetic 中直接编译会报错，这时需要对文件做出修改：

对文件 `./src/LIO-SAM/CMakeLists.txt` 中的 Line 5，将 `-std=c++11` 改为 `-std=c++14`：

```cmake
# set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_FLAGS "-std=c++14")
```

对文件 `./src/LIO-SAM/include/utility.h` 中的 Line 18，注释掉原 OpenCV 头文件，改为：

```c++
// #include <opencv/cv.h>
#include <opencv2/opencv.hpp>
```

此时作出如上修改后，可能会编译成功。若仍不成功，可继续在文件 `./src/LIO-SAM/include/utility.h` 中，将 `#include <pcl/kdtree/kdtree_flann.h>` 一句放在 `#include <opencv2/opencv.hpp>` 的前面，然后再次编译。

编译完成后，配置环境变量，运行 launch 文件

```bash
# in ~/liosam_ws directory
source devel/setup.bash
roslaunch lio_sam run.launch
```

在项目仓库中下载数据包，在数据包所在的目录下新打开一个终端，使用以下命名，并把 bag_name 更换为需要使用的数据包名。

```bash
rosbag play bag_name
rosbag play bag_name -r 3 # 以3倍速率播放
```

步行的数据集不需要修改任何参数，可以直接运行。

公园数据集用于使用 GPS 数据测试 LIO-SAM。该数据集由 Yewei Huang(<https://robustfieldautonomylab.github.io/people.html>) 收集。要启用 GPS 功能，请将"params.yaml"中的"gpsTopic"更改为"odometry/gps"。在 Rviz 中，取消选中"地图（云）“并选中"地图（全局）”。还要检查"Odom GPS"，它可以可视化 GPS 里程计。可以调整"gpsCovThreshold"以过滤不良 GPS 读数。 “poseCovThreshold"可用于调整将 GPS 因子添加到图形的频率。例如，您会注意到 GPS 会不断修正轨迹，因为您将"poseCovThreshold"设置为 1.0。由于 iSAM 的重度优化(heavy optimization)，建议播放速度为”-r 1"。

保存地图：在 `~/liosam_ws/src/LIO-SAM/config/params.yaml` 文件中修改 `savePCD` 为 `true` ，以及修改 `savePCDDirectory` 为想要保存地图的目录。

## Run LIO-SAM in Gazebo 仿真环境中运行

在没有硬件设备的情况下，我们也可以在 Gazebo 中仿真，模拟真实的环境，并通过录制 rosbag 包方便后期调试排查问题。

现今在 github 上有较为可行的项目，地址为 <https://github.com/balmung08/Slam_Simulation> ，选择其中的“基于 Velodyne 雷达 SDK 与 Gazebo 的车辆模型与仿真环境”，也可以直接访问仓库 <https://github.com/linzs-online/robot_gazebo> ，经过测试，该仓库中的 `fdilink_ahrs` 实际上为冗余目录，而 `realsense_ros_gazebo` 代表相机的软件包实际上并非必须，也就是我们使用 VLP_16 激光雷达和 IMU 单元就可以完成建图，在录制包的过程中我们只需要录雷达和imu话题（/velodyne_points和/imu/data）就可以了。

该项目的主要功能：在机器人模型中添加 16线激光雷达、IMU、RGB-D 相机，然后使用 LIO-SAM 建图，其中小车的移动使用 ros 包 teleop_twist_keyboard 实现，后期笔者考虑在项目中添加 SLAM 算法，实现机器小车的自主路径规划和导航。

环境要求：Ubuntu 20.04 + ROS1 Noetic。

软件包要求，参见章节 “LIO-SAM 部署”。

在终端中运行：

```bash
cd catkin_ws/src
git clone https://github.com/tan-9729/liosam-gazebo.git

cd ..
catkin_make
```

运行 Gazebo 和加载机器人模型：

```bash
roslaunch scout_gazebo scout_gazebo.launch
```

启动 Rviz 界面：

```bash
roslaunch lio_sam run.launch
```

使用键盘控制小车移动：

```bash
rosrun teleop_twist_keyboard teleop_twist_keyboard.py
```

地图保存的相关配置在 `robot_gazebo/LIO-SAM/config/params.yaml` 文件中，即以下两行（注意：保存地图时会先清除该目录原有内容，因此务必选择一个空目录）：

```yaml
savePCD: true
savePCDDirectory: "/robot_ws/src/pcd_maps/"
```

完成见图之后到对应目录查看：

```bash
pcl_viewer xxx.pcd
```

如果报错，提示命令未找到，可以使用如下命令安装 `pcl_tools` 后再次执行上面的命令：

```bash
sudo apt install pcl_tools
```
