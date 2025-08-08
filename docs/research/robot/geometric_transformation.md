# Geometric Transformation 几何变换

!!! abstract
    主要讨论 3D 坐标变换中的旋转矩阵和平移向量，以 激光雷达 LiDAR 到相机 Camera 的姿态变换 为例。

## 3D 位姿变换

描述一个坐标系相对于另一个坐标系的位置和姿态（统称 位姿），分别使用平移向量和旋转矩阵描述。以坐标系 $A$ 为参考坐标系，坐标系 $B$ 的位姿用 旋转矩阵和平移矩阵 描述为：

$$
{}^A_B\mathbf{R}=\begin{pmatrix}
r_{11} & r_{12} & r_{13} \\
r_{21} & r_{22} & r_{23} \\
r_{31} & r_{32} & r_{33}
\end{pmatrix}
=\begin{pmatrix}
B_x \cdot A_x & B_y \cdot A_x & B_z \cdot A_x \\
B_x \cdot A_y & B_y \cdot A_y & B_z \cdot A_y \\
B_x \cdot A_z & B_y \cdot A_z & B_z \cdot A_z
\end{pmatrix},\quad \mathbf{t}=\overrightarrow{AB}
$$

其中 ${}^A_B\mathbf{R}$ 为**正交矩阵**，满足 ${}^A_B\mathbf{R}^T={}^A_B\mathbf{R}^{-1}={}^B_A\mathbf{R}$ .

从 $B$ 坐标系变换到 $A$ 坐标系的 齐次坐标变换矩阵  为：

$$
{}^A_B\mathbf{T}=\begin{pmatrix}
\mathbf{R} & \mathbf{t} \\
\mathbf{0}_3^T & 1
\end{pmatrix}
=\begin{pmatrix}
r_{11} & r_{12} & r_{13} & t_1 \\
r_{21} & r_{22} & r_{23} & t_2 \\
r_{31} & r_{32} & r_{33} & t_3 \\
0 & 0 & 0 & 1
\end{pmatrix}
$$

在坐标系 $B$ 中齐次坐标为 ${}^BP=[x,y,z,1]^T$ 的点，变换到坐标系 $A$ 之后坐标为 ${}^AP={}^A_B\mathbf{T}\cdot{}^BP$ .

复合变换：从坐标系 $C$ 变换到坐标系 $A$ ，可以拆分为先从坐标系 $C$ 到坐标系 $B$ 的变换，再乘以从坐标系 $B$ 到坐标系 $A$ 的变换： ${}^A_C\mathbf{T}={}^A_B\mathbf{T}\cdot {}^B_C\mathbf{T}$ .

---

MATLAB 代码实现 激光雷达和相机 坐标变换可视化：

```matlab
% LiDAR --> Camera
T_cam_lidar = [0.0, -1.0, 0.0, 0.0;
               0.6,  0.0, -0.8, -0.93094;
               0.8,  0.0,  0.6, -0.26592;
               0.0,  0.0,  0.0, 1.0];

% Camera --> LiDAR
T_lidar_cam = inv(T_cam_lidar);

% Rotation matrix and translation vector
R = T_lidar_cam(1:3, 1:3);
t = T_lidar_cam(1:3, 4);

% Unit vector of camera frame
camera_axes = transpose([1 0 0; 0 1 0; 0 0 1]);
rotated_camera_axes = R * camera_axes;
camera_origin = t;

figure;
hold on; grid on; axis equal;
xlabel('X'); ylabel('Y'); zlabel('Z');
title('Camera pose w.r.t. LiDAR');

% plot LiDAR frame
quiver3(0, 0, 0, 1, 0, 0, 'r', 'LineWidth', 2);
quiver3(0, 0, 0, 0, 1, 0, 'g', 'LineWidth', 2);
quiver3(0, 0, 0, 0, 0, 1, 'b', 'LineWidth', 2);
text(1, 0, 0, 'LiDAR X', 'Color', 'r');
text(0, 1, 0, 'LiDAR Y', 'Color', 'g');
text(0, 0, 1, 'LiDAR Z', 'Color', 'b');

% plot camera frame
quiver3(camera_origin(1), camera_origin(2), camera_origin(3), ...
        rotated_camera_axes(1,1), rotated_camera_axes(2,1), rotated_camera_axes(3,1), ...
        'r', 'LineWidth', 2);
quiver3(camera_origin(1), camera_origin(2), camera_origin(3), ...
        rotated_camera_axes(1,2), rotated_camera_axes(2,2), rotated_camera_axes(3,2), ...
        'g', 'LineWidth', 2);
quiver3(camera_origin(1), camera_origin(2), camera_origin(3), ...
        rotated_camera_axes(1,3), rotated_camera_axes(2,3), rotated_camera_axes(3,3), ...
        'b', 'LineWidth', 2);

text(camera_origin(1) + rotated_camera_axes(1,1), ...
     camera_origin(2) + rotated_camera_axes(2,1), ...
     camera_origin(3) + rotated_camera_axes(3,1), ...
     'Camera X', 'Color', 'r');
text(camera_origin(1) + rotated_camera_axes(1,2), ...
     camera_origin(2) + rotated_camera_axes(2,2), ...
     camera_origin(3) + rotated_camera_axes(3,2), ...
     'Camera Y', 'Color', 'g');
text(camera_origin(1) + rotated_camera_axes(1,3), ...
     camera_origin(2) + rotated_camera_axes(2,3), ...
     camera_origin(3) + rotated_camera_axes(3,3), ...
     'Camera Z', 'Color', 'b');

legend('LiDAR X','LiDAR Y','LiDAR Z','Camera X','Camera Y','Camera Z');
view(3);
```

## 位姿的其他描述

### 欧拉角

考虑一个坐标系绕着**某一坐标轴**旋转任意角度 $\beta$ （右手定则，大拇指指向坐标轴正方向，则四指方向为旋转正方向）得到的旋转矩阵：

$$
\mathbf{R}_x(\beta)=\begin{pmatrix}
1 & 0 & 0 \\
0 & \cos\beta & -\sin\beta \\
0 & \sin\beta & \cos\beta
\end{pmatrix},\quad
\mathbf{R}_y(\beta)=\begin{pmatrix}
\cos\beta & 0 & \sin\beta \\
0 & 1 & 0 \\
-\sin\beta & 0 & \cos\beta
\end{pmatrix},\quad
\mathbf{R}_z(\beta)=\begin{pmatrix}
\cos\beta & -\sin\beta & 0 \\
\sin\beta & \cos\beta & 0 \\
0 & 0 & 1
\end{pmatrix}
$$

由此，旋转也可以表示为坐标系 先后分别绕 自身的各个坐标轴旋转一定的角度，例如分别绕着 $x,y,z$ 轴旋转 $\gamma,\alpha,\beta$ 角度，则旋转矩阵可以表示为 $\mathbf{R}=\mathbf{R}_z(\alpha)\mathbf{R}_y(\beta)\mathbf{R}_x(\gamma)$ . 但是由于矩阵乘法不具有交换性，因此不同的旋转顺序会得到不同的结果，例如一般情况下 $\mathbf{R}_z(\alpha)\mathbf{R}_y(\beta)\mathbf{R}_x(\gamma)\neq\mathbf{R}_x(\gamma)\mathbf{R}_y(\beta)\mathbf{R}_z(\alpha)$。对旋转顺序做排列组合，共有 $3\times2\times2=12$ 种顺序：

!!! success ""
    xyz, xyx, xzy, xzx,
    yxz, yxy, yzx, yzy,
    zxy, zyx, zyz, zxz

同时，还需要考虑旋转时的参考坐标系：

（1）绕着一个固定的坐标系的坐标轴进行旋转，称为**固定角欧拉角**或**固定轴旋转**；

这种情况比较简单，以 “绕固定轴以 **XYZ** 顺序旋转欧拉角 $\gamma,\beta,\alpha$ ” 的情况为例，总的旋转矩阵为 $\mathbf{R}_{xyz}(\gamma,\beta,\alpha)=\mathbf{R}_z(\alpha)\cdot\mathbf{R}_y(\beta)\cdot\mathbf{R}_x(\gamma)$ .

（2）每次旋转都以自身坐标轴为轴进行旋转，称为**非固定旋转轴的欧拉角**。

假设坐标系 $A$ 按照 **XYZ** 的顺序 旋转 $\gamma,\beta,\alpha$ 角度之后变为坐标系 $B$ （中间两步的结果分别为 $B',B''$ ），则 $B\to A$ 的位姿变换为 ${}^A_B\mathbf{R}={}^A_{B'}\mathbf{R}\cdot{}^{B'}_{B''}\mathbf{R}\cdot{}^{B''}_{B}\mathbf{R}=\mathbf{R}_x(\gamma)\cdot\mathbf{R}_y(\beta)\cdot\mathbf{R}_z(\alpha)$ .

可见，绕固定坐标轴 X-Y-Z 旋转 $(\gamma,\beta,\alpha)$ 和绕自身坐标轴 Z-Y-X 旋转 $(\alpha,\beta,\gamma)$ 的结果相同。

由此可见欧拉角有 $12\times2=24$ 种旋转方式。

一般将绕 Z-Y-X 旋转的角度分别称为 Yaw-Pitch-Roll .

### 轴角

对于上述欧拉角的旋转，都是绕着坐标系的主轴 XYZ 旋转（无论是固定坐标系还是自身坐标系），但实际上，对于任何旋转，我们都可以找到一个**向量**，两个坐标系之间的变换仅绕着这个向量旋转得到。

假设初始时坐标系 $A,B$ 重合，坐标系 $B$ 绕着坐标系 $A$ 中的向量 ${}^AK=[k_x,k_y,k_z]^T$ 按照右手定则旋转 $\beta$ 角度。旋转之后， $B$ 相对于 $A$ 的位姿，或者说 $B\to A$ 的旋转矩阵为：

$$
{}^A_B\mathbf{R}(K,\beta)=\begin{pmatrix}
k_x^2(1 - \cos\beta) + \cos\beta & k_x k_y(1 - \cos\beta) - k_z \sin\beta & k_x k_z(1 - \cos\beta) + k_y \sin\beta \\
k_x k_y(1 - \cos\beta) + k_z \sin\beta & k_y^2(1 - \cos\beta) + \cos\beta & k_y k_z(1 - \cos\beta) - k_x \sin\beta \\
k_x k_z(1 - \cos\beta) - k_y \sin\beta & k_y k_z(1 - \cos\beta) + k_x \sin\beta & k_z^2(1 - \cos\beta) + \cos\beta
\end{pmatrix}
$$

### 四元数

四元数的由1个实部和3个虚部组成 $q=w+x\cdot\vec{i}+y\cdot\vec{j}+z\cdot\vec{k}$ ，满足 $w^2+x^2+y^2+z^2=1$ .

四元数 转 旋转矩阵：

$$
\mathbf{R} = \begin{pmatrix}
1 - 2y^2 - 2z^2 & 2(xy - zw) & 2(xz + yw) \\
2(xy + zw) & 1 - 2x^2 - 2z^2 & 2(yz - xw) \\
2(xz - yw) & 2(yz + xw) & 1 - 2x^2 - 2y^2
\end{pmatrix}
$$

旋转矩阵 转 四元数：

$$
\begin{align*}
x &= \frac{r_{32} - r_{23}}{4w} \\
y &= \frac{r_{13} - r_{31}}{4w} \\
z &= \frac{r_{21} - r_{12}}{4w} \\
w &= \frac{1}{2}\sqrt{1 + r_{11} + r_{22} + r_{33}}
\end{align*}
$$

四元数转欧拉角

$$
\begin{pmatrix}
\gamma \\
\beta \\
\alpha
\end{pmatrix}
=
\begin{pmatrix}
\text{atan2}(2(wx + yz), 1 - 2(x^2 + y^2)) \\
\text{arcsin}(2(wy - zx)) \\
\text{atan2}(2(wz + xy), 1 - 2(y^2 + z^2))
\end{pmatrix}
$$

欧拉角 转 四元数：

$$
\begin{pmatrix}
x \\ y \\ z \\ w
\end{pmatrix}=
\begin{pmatrix}
\cos(\alpha/2) \\ 0 \\ 0 \\ \sin(\alpha/2)
\end{pmatrix}
\begin{pmatrix}
\cos(\beta/2) \\ 0  \\ \sin(\beta/2) \\ 0
\end{pmatrix}
\begin{pmatrix}
\cos(\gamma/2) \\ \sin(\gamma/2) \\ 0 \\ 0
\end{pmatrix}
=
\begin{pmatrix}
\cos(\gamma/2)\cos(\beta/2)\cos(\alpha/2) + \sin(\gamma/2)\sin(\beta/2)\sin(\alpha/2) \\
\sin(\gamma/2)\cos(\beta/2)\cos(\alpha/2) - \cos(\gamma/2)\sin(\beta/2)\sin(\alpha/2) \\
\cos(\gamma/2)\sin(\beta/2)\cos(\alpha/2) + \sin(\gamma/2)\cos(\beta/2)\sin(\alpha/2) \\
\cos(\gamma/2)\cos(\beta/2)\sin(\alpha/2) - \sin(\gamma/2)\sin(\beta/2)\cos(\alpha/2)
\end{pmatrix}
$$

这里的向量乘法是**四元数乘法**，具体规则为：

$$
\begin{pmatrix}
a \\ b \\ c \\ d
\end{pmatrix}
\begin{pmatrix}
e \\ f \\ g \\ h
\end{pmatrix}
=\begin{pmatrix}
ae - bf - cg - dh \\ af + be + ch - dg \\ ag - bh + ce + df \\ ah + bg - cf + de
\end{pmatrix}
$$

相同向量的乘法按照复数相乘规则 $\vec{i}\cdot\vec{i}=-1,\;\vec{j}\cdot\vec{j}=-1,\;\vec{k}\cdot\vec{k}=-1$ . 正交向量的乘法按照叉乘的右手定则 $\vec{i}\cdot\vec{j}=\vec{k},\;\vec{j}\cdot\vec{i}=-\vec{k},\cdots$ .
