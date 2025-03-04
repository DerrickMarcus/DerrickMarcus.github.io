# Gazebo

**本为适用于新版 Gazebo，搭配 Ubuntu 22 或 24 + ROS2.**

gazebo 使用 SDF(Simulation Description Format) 文件保存模型，场景和机器人
有一个 sdf 是顶级标签，下面有一个 world 子元素，world 包含多种子元素，比如 scene, light, model

模型名称：ground_plane 地面，box 立方体，cylinder 圆柱体，sphere 球体，capsule 胶囊形状，epplisoid 椭球体

例如对于一个 box 模型：

```xml
    <model name="box">
      <pose>0 0 0.5 0 0 0</pose>
      <link name="box_link">
        <inertial>
          <inertia>
            <ixx>0.16666</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>0.16666</iyy>
            <iyz>0</iyz>
            <izz>0.16666</izz>
          </inertia>
          <mass>1.0</mass>
        </inertial>
        <collision name="box_collision">
          <geometry>
            <box>
              <size>1 1 1</size>
            </box>
          </geometry>
        </collision>

        <visual name="box_visual">
          <geometry>
            <box>
              <size>1 1 1</size>
            </box>
          </geometry>
          <material>
            <ambient>1 0 0 1</ambient>
            <diffuse>1 0 0 1</diffuse>
            <specular>1 0 0 1</specular>
          </material>
        </visual>
      </link>
    </model>
```

pose 元素设置初始位置和姿态，例如 `\<pose>X Y Z R P Y\</pose>` 前三个为直角坐标 xyz，后三个为欧拉旋转的坐标 rpy (Row, Pitch, Yaw)，默认弧度制。`\<pose degrees="true">0 0 0.5 0 0 0\</pose>` 使用角度制。

在另外一个终端中打开并使用如下命令，实现动态改变模型的位置：

```bash
gz service -s /world/shapes/set_pose --reqtype gz.msgs.Pose --reptype gz.msgs.Boolean --timeout 300 --req 'name: "box", position: {z: 5.0}'
```

link 元素表示机器人中的杆件，有三个子元素：

1. inertial 转动惯量，可自动计算。
2. collision 碰撞模型，内有一个 geometry 几何体模型，用于碰撞检测。
3. visual 3D模型，同样有 geometry 元素，还有 material 元素，用于改变杆件颜色，ambient,diffuse,specular 分别设置环境光照，漫反射和镜面反射中 4 个值红、绿、蓝、透明度取值，范围为 0-1。

Gazebo GUI 操作，放置物体，添加光源，改变位置，改变视图（透明，显示坐标轴，显示质心，显示碰撞模型，显示转动惯量，显示线框图），复制粘贴，……

右侧插件：

Apply force and torque 施加力和力矩

Resource Spawner 在线加载更多模型

Teleop 控制小车移动
