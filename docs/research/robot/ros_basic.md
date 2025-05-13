# ROS Basic

Ubuntu 20.04 Focal, ROS 1 Noetic.

一些基本概念：

rviz，3D 可视化工具，用来创建地图、显示 3D 点云。

tf，坐标变换系统。两种坐标系：固定坐标系（用于表示世界的参考坐标系），目标坐标系（相对于摄像机视角的参考坐标系）

Gazebo，物理仿真工具。

SLAM(Simultaneous Localization and Mapping)，同步定位与建图

## 工作空间

创建工作空间：

```bash
# 创建空间
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src

# 初始化
cd ~/catkin_ws/src
catkin_init_workspace # 在src文件夹

# 编译工作空间，需要切换到工作空间根目录
cd ~/catkin_ws/
catkin_make
catkin_make install # 可选，生成 install 目录
# 从指定源代码目录进行编译
catkin_make --source my_src
catkin_make install --source my_src
# 指定软件包进行构建
catkin_make --pkg xxx


# 设置环境变量
source devel/local_setup.bash
# 查看当前环境变量
echo $ROS_PACKAGE_PATH

# 创建一个软件包
catkin_create_pkg <package_name> [depend1] [depend2] [depend3]
# 查看软件包的依赖


# 运行所有 ROS 命令之前，新开一个终端
roscore

# else
rospack
roscd
rosls
```

工作空间目录：

src 源代码

build 编译空间

devel 开发空间

某个软件包的目录结构：

CMakeLists.txt, package.xml, scripts/ , msg/ , srv/ , include/ , src/ , launch/ , urdf/ , config/ , meshes/

PS: 如果找不到软件包，用 `source devel/setup.bash` 而非 `source devel/local_setup.bash` 。

## ROS 节点

查看节点：

```bash
rosnode list
rosnode info <node_name>
```

运行一个节点，基于一个软件包：

```bash
rosrun [package_name] [node_name]
```

运行多个节点，基于 .launch 文件：

```bash
roslaunch [package] [filename.launch]
```

## launch 文件的结构

launch 文件：

- pkg：节点所在的功能包名称
- node：节点名称
- type：节点的可执行文件名称
- name：节点运行时的名称
- output：“log | screen” (可选)日志发送目标，可以设置为 log 日志文件，或 screen 屏幕,默认是 log
- group：组，有命名空间特性
- param / rosparam：参数
- arg ：launch 文件内部局部变量
- remap：重映射

具体来讲：

标签 node：

```xml
<node pkg="package_name" type="executable_node" name="node_name" args="$()" respawn="true" output="sceen">
```

pkg 节点所在包的名称，type 可执行文件（节点）的名称，name 节点运行时的名称，args 传递命令行设置的参数，respawn 是否自动重启，output 是否将节点输出到屏幕。

标签 param，改变程序变量的参数值：

```xml
<param name="param_name" type="param_type" value="param_value" />
<!-- param 标签可以嵌入到 node 标签中，以此来作为该 node 的私有参数 -->
<node>
 <param name="param_name" type="param_type" value="param_value" />
</node>
```

name 参数名称，type 参数类型（double，str，int等），value 参数值。

标签 rosparam，实现节点从参数服务器上 load，dump，delete YAML 文件：

```xml
<!-- 加载package_name功能包下的example.yaml文件 -->
<rosparam command="load" file="$(find package_name)/example.yaml">
<!-- 导出example_out.yaml文件到package_name功能包下 -->
<rosparam command="dump" file="$(find package_name)/example_out.yaml" />
<!-- 删除参数 -->
<rosparam command="delete" param="xxx/param">
```

command 功能类型，file 参数文件的路径，param 参数名称。

标签 include，导入其他 launch 文件：

```xml
<include file="$(find package_name)/launch_file_name">
```

标签 remap，实现节点名称的重映射，原始名称-->新名称：

```xml
<remap from="turtle1/cmd_vel" to="/cmd_vel" />
<!-- remap 标签同样可以嵌入到 node 标签中，以此来作为该 node 的私有重映射 -->
<node>
 <remap from="turtle1/cmd_vel" to="/cmd_vel" />
</node>
```

标签 arg，表示启动参数：

```xml
<arg name="arg_name" default="arg_default" />
<arg name="arg_name" value="arg_value" />
<!-- 命令行传递的 arg 参数可以覆盖 default，但不能覆盖 value。 -->
```

注意：

| arg   | 启动时的参数，只在launch文件中有意义   |
| ----- | -------------------------------------- |
| param | 运行时的参数，参数会存储在参数服务器中 |

标签 group，可以实现将一组配置应用到组内的所有节点，它也具有命名空间特点，可以将不同的节点放入不同的 namespace。

```xml
<!-- 用法1 -->
<group ns="namespace_1">
    <node pkg="pkg_name1" .../>
    <node pkg="pkg_name2" .../>
    ...
</group>

<group ns="namespace_2">
    <node pkg="pkg_name3" .../>
    <node pkg="pkg_name4" .../>
    ...
</group>
<!-- 用法2 -->
<!-- if = value：value 为 true 则包含内部信息 -->
<group if="$(arg foo1)">
    <node pkg="pkg_name1" .../>
</group>

<!-- unless = value：value 为 false 则包含内部信息 -->
<group unless="$(arg foo2)">
    <node pkg="pkg_name2" .../>
</group>
<!--
	当 foo1 == true 时包含其标签内部
	当 foo2 == false 时包含其标签内部
-->
```

**launch文件启动之前，无需再执行`roscore`指令启动`rosmaster`，launch文件可以自启动rosmaster**

## rqt 工具

使用 rqt_graph 查看节点状态图：

```bash
rosrun rqt_graph rqt_graph
rqt
rqt_graph
```

## 话题、服务、参数

关于话题的命令：

```bash
rostopic bw     display bandwidth used by topic
rostopic echo   print messages to screen
rostopic hz     display publishing rate of topic
rostopic list   print information about active topics
rostopic pub    publish data to topic
rostopic type   print topic type

rosmsg show <topic_type_name>
```

关于服务的命令：

```bash
rosservice list         print information about active services
rosservice call         call the service with the provided args
rosservice type         print service type
rosservice find         find services by service type
rosservice uri          print service ROSRPC uri
```

关于参数的命令：

```bash
rosparam set            set parameter
rosparam get            get parameter
rosparam load           load parameters from file
rosparam dump           dump parameters to file
rosparam delete         delete parameter
rosparam list           list parameter names
```

编辑某一个软件包内的某一个文件：

```bash
rosed [package_name] [filename]
```

添加软件包需要的依赖，修改：

package.xml 文件中的 `<depend>xxx</depend>` 标签。

CMakeLists.txt 文件中的

```cmake
find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  # Add more dependencies here
)
```

## rosdep 工具

使用 rosdep 检查是否满足依赖，以及安装未依赖的包：

```bash
rosdep check --from-path src --ignore-src -r -y
rosdep install --from-path src --ignore-src -r -y
```

## 录制数据包

.bag 文件可以保存 ROS 系统运行过程中产生的话题和服务数据，并播放出来供其他系统使用。

开始录制数据：

```bash
rosbag record -a
rosbag record -O xxx /turtle1/cmd_vel /turtle1/pose
```

录制全部话题，或者选择其中一部分话题。

```bash
rosbag info xxx.bag
rosbag play xxx.bag
rosbag play xxx.bag -r 4
```

查看数据包信息，以及播放数据包、按照一定速率播放数据包。
