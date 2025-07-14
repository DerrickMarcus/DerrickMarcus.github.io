# Gazebo Classic

!!! warning
    本文适用于 Gazebo Classic，搭配 ROS1 版本。新版 Gazebo 即 gz 请参考 <https://gazebosim.org/>

Gazebo 使用 `.world` 扩展名来表示仿真场景的配置文件， `.world` 文件实际上是 XML 格式的文件，只不过它使用的是 SDF 格式进行描述。

模型文件中，SDF 更适合 Gazebo 仿真，是 URDF 的升级版。URDF 更适合描述机器人模型，其中 Xacro 是 URDF 的升级版。有些使用场景需要注意：比如使用 Rviz 可视化只能使用 URDF 文件，并且 Xacro 兼容性更好，并联机器人的 Gazebo 仿真只能使用 SDF。

`.world` 文件是一个 XML 文档，它以 `<world>` 标签为根元素，包含了描述仿真世界的各种元素和其属性。通常 `.world` 文件包括了仿真世界中的各种模型、传感器、光源设置等信息。

Gazebo 通过命令行打开 `.world` 文件，加载仿真场景。

```bash
gazebo /usr/share/gazebo-11/worlds/empty_sky.world
```

ROS 必须通过 `gazebo_ros` 这个包来加载 `.world` 场景，其本质是 `gazebo_ros` 包调用 Gazebo 仿真器，继而加载 `.world` 场景。

ROS 打开仿真 `.world` 场景命令：

```bash
roslaunch xxx xxx.launch # roslaunch 软件包名称 launch文件名称
roslaunch gazebo_ros empty_world.launch
roslaunch gazebo_ros empty_world.launch world_name:=/path/to/your/world_file.world
```

`empty_world.launch` 用于加载 `empty.world` 或其他仿真世界文件，并可以包含其他 ROS 节点、参数设置和配置。

大致文件结构

在某一个软件包的根目录下，比如 `~/catkin_ws/src/xxx` 。

创建一个 `worlds` 目录，用于存放需要仿真的世界文件，比如 `a.world, b.sdf` .

创建一个 `launch` 目录，用于存放启动文件，比如 `c.launch` ，使用 XML 编写。

基本概念：

模型 models，世界 worlds，插件 Plugins，传感器 sensors，视觉和物理属性，通信接口。

注意：

`.world` 文件是一种特殊的 `.sdf` 文件。在实际使用中， `.world` 文件和 `.sdf` 文件经常配合使用。 `.world` 文件定义了仿真环境的整体框架和全局参数，而 `.sdf` 文件则用于定义环境中的具体对象。在 Gazebo中，可以通过 `<include>` 标签将 `.sdf` 文件中定义的模型包含到 `.world` 文件中，这样可以重用 `.sdf` 文件中定义的模型，并且可以更容易地管理和维护仿真环境。

`roslaunch` 是通过 `gazebo_ros` 包，启动 Gazebo，然后把`.world` 的路径作为参数传进去，本质还是用 Gazebo 打开 world 文件，相当于 ROS 在外部进行了封装。 `.launch` 文件提供了 `.world` 文件的路径供 gazebo_ros 包加载，传给 Gazebo 并打开 `.world` 文件。

`roslaunch` 加载 `.world` 文件，等效于直接在终端或命令行中使用 `gazebo world_file.world` 命令。

## xacro 文件

概念
Xacro 是 XML Macros 的缩写，Xacro 是一种 XML 宏语言，是可编程的 XML。

原理
Xacro 可以声明变量，可以通过数学运算求解，使用流程控制控制执行顺序，还可以通过类似函数的实现，封装固定的逻辑，将逻辑中需要的可变的数据以参数的方式暴露出去，从而提高代码复用率以及程序的安全性。

作用
较之于纯粹的 URDF 实现，可以编写更安全、精简、易读性更强的机器人模型文件，且可以提高编写效率。

注意：

较之于 Rviz，Gazebo 在集成 URDF 时，需要做些许修改，比如：必须添加 `collision` 碰撞属性相关参数、必须添加 `inertial` 惯性矩阵相关参数，另外，如果直接移植 Rviz 中机器人的颜色设置是没有显示的，颜色设置也必须做相应的变更。

惯性矩阵的设置需要结合 `link` 的质量与外形参数动态生成，标准的球体、圆柱与立方体的惯性矩阵公式如下(已经封装为 xacro 实现):

球体惯性矩阵

```xml
<!-- Macro for inertia matrix -->
    <xacro:macro name="sphere_inertial_matrix" params="m r">
        <inertial>
            <mass value="${m}" />
            <inertia ixx="${2*m*r*r/5}" ixy="0" ixz="0"
                iyy="${2*m*r*r/5}" iyz="0"
                izz="${2*m*r*r/5}" />
        </inertial>
    </xacro:macro>
```

圆柱惯性矩阵

```xml
<xacro:macro name="cylinder_inertial_matrix" params="m r h">
        <inertial>
            <mass value="${m}" />
            <inertia ixx="${m*(3*r*r+h*h)/12}" ixy = "0" ixz = "0"
                iyy="${m*(3*r*r+h*h)/12}" iyz = "0"
                izz="${m*r*r/2}" />
        </inertial>
    </xacro:macro>
```

立方体惯性矩阵

```xml
 <xacro:macro name="Box_inertial_matrix" params="m l w h">
       <inertial>
               <mass value="${m}" />
               <inertia ixx="${m*(h*h + l*l)/12}" ixy = "0" ixz = "0"
                   iyy="${m*(w*w + l*l)/12}" iyz= "0"
                   izz="${m*(w*w + h*h)/12}" />
       </inertial>
   </xacro:macro>
```

需要注意的是，原则上，除了 `base_footprint` 外，机器人的每个刚体部分都需要设置惯性矩阵，且惯性矩阵必须经计算得出，如果随意定义刚体部分的惯性矩阵，那么可能会导致机器人在 Gazebo 中出现抖动，移动等现象。

## ros_control

场景:同一套 ROS 程序，如何部署在不同的机器人系统上，比如：开发阶段为了提高效率是在仿真平台上测试的，部署时又有不同的实体机器人平台，不同平台的实现是有差异的，如何保证 ROS 程序的可移植性？ROS 内置的解决方式是 ros_control。

ros_control 是一组软件包，它包含了控制器接口，控制器管理器，传输和硬件接口。ros_control 是一套机器人控制的中间件，是一套规范，不同的机器人平台只要按照这套规范实现，那么就可以保证 与ROS 程序兼容，通过这套规范，实现了一种可插拔的架构设计，大大提高了程序设计的效率与灵活性。

Gazebo 已经实现了 ros_control 的相关接口，如果需要在 Gazebo 中控制机器人运动，直接调用相关接口即可。
