# Run LIO-SAM in Gazebo

在没有硬件设备的情况下，我们也可以在 Gazebo 中仿真，模拟真实的环境，并通过录制 rosbag 包方便后期调试排查问题。

现今在 github 上有较为可行的项目，地址为 <https://github.com/balmung08/Slam_Simulation> ，选择其中的“基于 Velodyne 雷达 SDK 与 Gazebo 的车辆模型与仿真环境”，也可以直接访问仓库 <https://github.com/linzs-online/robot_gazebo> ，经过测试，该仓库中的 `fdilink_ahrs` 实际上为冗余目录，而 `realsense_ros_gazebo` 代表相机的软件包实际上并非必须，也就是我们使用 VLP_16 激光雷达和 IMU 单元就可以完成建图，在录制包的过程中我们只需要录雷达和imu话题（/velodyne_points和/imu/data）就可以了。

该项目的主要功能：在机器人模型中添加 16线激光雷达、IMU、RGB-D 相机，然后使用 LIO-SAM 建图，其中小车的移动使用 ros 包 teleop_twist_keyboard 实现，后期笔者考虑在项目中添加 SLAM 算法，实现机器小车的自主路径规划和导航。

环境要求：Ubuntu 20.04 + ROS1 Noetic

软件包要求，参见章节 “LIO-SAM 部署”

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

启动 Rviz 界面

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

完成见图之后到对应目录查看

```bash
pcl_viewer xxx.pcd
```

如果报错，提示命令未找到，可以使用如下命令安装 `pcl_tools` 后再次执行上面的命令：

```bash
sudo apt install pcl_tools
```
