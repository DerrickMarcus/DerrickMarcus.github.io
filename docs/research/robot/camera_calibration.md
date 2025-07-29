# Camera Calibration

!!! abstract
    在之前《数字图像处理》课程 [7 几何校正和修补](../../course_notes/digital_image/chapter7.md) 中我们简单介绍了相机成像原理和相机内参标定的张正友方法。这里我们将进一步详细讨论。

首先明确4个概念：

1. 世界坐标系 $W:O_W-X_WY_WZ_W$ ，三维，真实物体所在的世界坐标系，通常自定义。
2. 相机坐标系 $C:O_C-X_CY_CZ_C$ ，三维，以相机的光心 $O_C$ 为原点，向前为 $+Z_C$ 方向，向右为 $+X_C$ 方向，向下为 $+Y_C$ 方向。
3. 图像坐标系 $o-xy$ ，二维，原点为 成像平面与光轴交点（理想主点），位于图像中心，向右为 $+x$ 方向，向下为 $+y$ 方向。
4. 像素坐标系 $o-uv$ ，二维，原点为图像右上角，向右为 $+u$ 方向，向下为 $+v$ 方向。坐标离散化， $(u,v)$ 直接代表最终图像中某个像素的数组索引。

外参矩阵是一个三维齐次坐标变换矩阵，实现 世界坐标系 到 相机坐标系的变换；内参矩阵考虑畸变系数，实现图像坐标系 到 像素坐标系的非线性映射。

## World -> Camera

相机标定时，通常以标定板的角点为世界坐标系原点。

世界坐标系到相机坐标系的 齐次变换为：

$$
\begin{pmatrix}
X_C \\ Y_C \\ Z_C
\end{pmatrix}=\bold{R}\begin{pmatrix}
X_W \\ Y_W \\ Z_W
\end{pmatrix}+\bold{t} \implies
\begin{pmatrix}
X_C \\ Y_C \\ Z_C \\ 1
\end{pmatrix}
=\begin{pmatrix}
\bold{R} & \bold{t} \\
\bold{0} & 1
\end{pmatrix}
\begin{pmatrix}
X_W \\ Y_W \\ Z_W \\ 1
\end{pmatrix}
$$

通常也把 $[\bold{R}|\bold{t}]\in\mathbb{R}^{3\times 4}$ 称为外参矩阵。

## Camera -> Image

## Image -> Pixel

假设单个像素对应相机成像平面对应的实际物理尺寸为 $d_x,d_y$ （类比感受野），图像坐标 $(x, y)$ 和像素坐标 $(u, v)$ 间的转换关系如下：

$$
\begin{align*}
u - u_0 &= \frac{x}{d_x} \\
v - v_0 &= \frac{y}{d_y}
\end{align*}
$$

可以写出变换矩阵：

$$
\begin{pmatrix}
u \\ v \\ 1
\end{pmatrix}
=\begin{pmatrix}
\dfrac{1}{d_x} & 0 & u_0 \\
0 & \dfrac{1}{d_y} & v_0 \\
0 & 0 & 1
\end{pmatrix}
\begin{pmatrix}
x \\ y \\ 1
\end{pmatrix}
$$

一般由于工艺偏差 $d_x\neq d_y$ ，导致一个像素对应的感受野实际上是矩形而不是正方形。
