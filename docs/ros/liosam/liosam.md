# Deploy LIO-SAM

LIO-SAM 项目地址：<https://github.com/TixiaoShan/LIO-SAM>

创建工作空间：

```bash
# in ~ directory
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

对文件 `./src/LIO-SAM/include/utility.h` 中的 Line 18，改为：

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
rosbag play bag_name -r 3 # 以3倍速度播放
```

步行的数据集不需要修改任何参数，可以直接运行。

公园数据集用于使用 GPS 数据测试 LIO-SAM。该数据集由 Yewei Huang(<https://robustfieldautonomylab.github.io/people.html>) 收集。要启用 GPS 功能，请将"params.yaml"中的"gpsTopic"更改为"odometry/gps"。在 Rviz 中，取消选中"地图（云）“并选中"地图（全局）”。还要检查"Odom GPS"，它可以可视化 GPS 里程计。可以调整"gpsCovThreshold"以过滤不良 GPS 读数。 “poseCovThreshold"可用于调整将 GPS 因子添加到图形的频率。例如，您会注意到 GPS 会不断修正轨迹，因为您将"poseCovThreshold"设置为 1.0。由于 iSAM 的重度优化(heavy optimization)，建议播放速度为”-r 1"。

保存地图：在 `~/liosam_ws/src/LIO-SAM/config/params.yaml` 文件中修改 `savePCD` 为 `true` ，以及修改 `savePCDDirectory` 为想要保存地图的目录。
