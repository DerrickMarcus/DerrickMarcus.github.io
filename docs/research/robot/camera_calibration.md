# Camera Calibration

!!! abstract
    在之前《数字图像处理》课程章节 [7 图像校正和修补](../../course_notes/digital_image/chapter7.md) 中我们简单介绍了相机成像原理和张正友相机内参标定方法。这里我们将进一步详细讨论。

首先明确4个概念：

1. **世界坐标系** $W:O_W-X_WY_WZ_W$ ，三维，真实物体所在的世界坐标系，通常自定义。
2. **相机坐标系** $C:O_C-X_CY_CZ_C$ ，三维，以相机透镜的光心 $O_C$ 为原点，向前为 $+Z_C$ 方向，向右为 $+X_C$ 方向，向下为 $+Y_C$ 方向。
3. **图像坐标系** $o-xy$ ，二维，即为透镜后的成像平面（为了便于分析计算，将其对调至透镜前方），原点为 成像平面与光轴交点（理想主点），位于图像中心，且在相机坐标系的 $+Z_C$ 轴 $Z_C=f$ **焦距处**，向右为 $+x$ 方向，向下为 $+y$ 方向。
4. **像素坐标系** $o-uv$ ，二维，原点为图像左上角，向右为 $+u$ 方向，向下为 $+v$ 方向。坐标离散化，坐标 $(u,v)$ 直接代表最终图像中某个像素的数组索引。

外参矩阵是一个三维齐次坐标变换矩阵，实现 世界坐标系 到 相机坐标系的变换；内参矩阵考虑畸变系数，实现图像坐标系 到 像素坐标系的非线性映射。

## World to Camera

相机标定时，通常以标定板的角点为世界坐标系原点。

世界坐标系到相机坐标系的 齐次变换为：

$$
\begin{pmatrix}
X_C \\ Y_C \\ Z_C
\end{pmatrix}=\mathbf{R}\begin{pmatrix}
X_W \\ Y_W \\ Z_W
\end{pmatrix}+\mathbf{t} \implies
\begin{pmatrix}
X_C \\ Y_C \\ Z_C \\ 1
\end{pmatrix}
=\begin{pmatrix}
\mathbf{R} & \mathbf{t} \\
\mathbf{0}_3^T & 1
\end{pmatrix}
\begin{pmatrix}
X_W \\ Y_W \\ Z_W \\ 1
\end{pmatrix}
$$

通常也把 ${}^C_W\mathbf{T}=[\mathbf{R}\;|\;\mathbf{t}]\in\mathbb{R}^{3\times 4}$ 称为外参矩阵，后续会进一步介绍。

## Camera to Image

在相机坐标系中，物体 $P$ 点坐标为 $X_C,Y_C,Z_C$ ，其与原点的连线投射到成像平面（图像坐标系）上，坐标为 $(x,y,f)$ ，根据相似性：

$$
\frac{x}{X_C}=\frac{y}{Y_C}=\frac{f}{Z_C}
$$

写出坐标变换关系：

$$
\begin{pmatrix}
x \\ y \\ 1
\end{pmatrix}=
\begin{pmatrix}
\dfrac{f}{Z_C} & 0 & 0 \\
0 & \dfrac{f}{Z_C} & 0 \\
0 & 0 & \dfrac{1}{Z_C}
\end{pmatrix}
\begin{pmatrix}
X_C \\ Y_C \\ Z_C
\end{pmatrix}
$$

## Image to Pixel

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

## Intrinsics

根据 `Camera -> Image` 以及 `Image -> Pixel` 的变换关系，可以写出 `Camera -> Pixel` 的坐标变换关系：

$$
\begin{pmatrix}
u \\ v \\ 1
\end{pmatrix}=\frac{1}{Z_C}\begin{pmatrix}
\dfrac{f}{d_x} & 0 & u_0 \\
0 & \dfrac{f}{d_y} & v_0 \\
0 & 0 & 1
\end{pmatrix}
\begin{pmatrix}
X_C \\ Y_C \\ Z_C
\end{pmatrix}
$$

令 $f_x=\dfrac{f}{d_x},\;f_y=\dfrac{f}{d_y}$ ，代表将物理尺寸的焦距 $f$ 转换为以像素为单位的焦距。同时考虑到图像可能产生偏斜，用 $\gamma$ 表示像素纵向边界相比于 $y$ 轴的倾斜因子，则相机的**内参 Intrinsic**可以表示为：

$$
\mathbf{K}=\begin{pmatrix}
f_x & \gamma & u_0 \\
0 & f_y & v_0 \\
0 & 0 & 1
\end{pmatrix}
$$

相机的内参只由相机本身决定，且是固定不变的，不随物体的移动而改变。

## Extrinsics

前文提到的 `World -> Camera` 的变换中的旋转矩阵和平移向量就是相机的**外参 Extrinsinc**： $[\mathbf{R}\;|\;\mathbf{t}]\in \mathbb{R}^{3\times 4}$ . 外参描述的是相机和外部世界的坐标变换关系，因此 外参是不断改变的，每一张照片的外参都不同。

已知相机的内参和外参，就可以写出 `World -> Pixel` 的变换关系：

$$
Z_C \begin{pmatrix}
u \\ v \\ 1
\end{pmatrix}=
\begin{pmatrix}
\dfrac{1}{d_x} & 0 & u_0 \\
0 & \dfrac{1}{d_y} & v_0 \\
0 & 0 & 1
\end{pmatrix}
\begin{pmatrix}
f & 0 & 0 & 0 \\
0 & f & 0 & 0 \\
0 & 0 & 1 & 0
\end{pmatrix}
\begin{pmatrix}
\mathbf{R} & \mathbf{t} \\
\mathbf{0}_3^T & 1
\end{pmatrix}
\begin{pmatrix}
X_W \\ Y_W \\ Z_W \\ 1
\end{pmatrix}
=\mathbf{K}_{3\times 3}\; [\mathbf{R}_{3\times 3}\quad \mathbf{t}_{3\times 1}]\;
\begin{pmatrix}
X_W \\ Y_W \\ Z_W \\ 1
\end{pmatrix}
$$

实现直接从 世界坐标系 到 像素坐标 的变换。

## Distortion

透镜的畸变主要分为**径向畸变**和**切向畸变**。

径向畸变是由于透镜形状的制造工艺导致，且越向透镜边缘移动径向畸变越严重。径向畸变有两种类型：**桶形畸变**和**枕形畸变**。

通常使用 $r=0$ 处的泰勒展开描述径向畸变，其中 $r$ 为该点到成像中心的距离。矫正后的坐标为：

$$
\begin{align*}
x_{\text{corrected}}&=x(1+k_1r^2+k_2r^4+k_3r^6) \\
y_{\text{corrected}}&=y(1+k_1r^2+k_2r^4+k_3r^6) \\
\end{align*}
$$

切向畸变是由于透镜制造上的缺陷使得透镜本身与图像平面不平行而产生的。如果存在切向畸变，一个矩形被投影到成像平面上时很可能会变成一个梯形。矫正后的坐标为：

$$
\begin{align*}
x_{\text{corrected}}&=x+2p_1xy+p_2(r^2+2x^2) \\
y_{\text{corrected}}&=y+2p_2xy+p_1(r^2+2y^2)
\end{align*}
$$

我们一共需要 $[k_1,k_2,k_3,p_1,p_2]$ 共 5 个畸变系数描述相机的畸变。一般对于质量较好的相机，切向畸变可以忽略，认为 $p_1=p_2=0$ .

## Calibration

相机内参标定方法参考**张正友棋盘格标定法**，论文链接 [Flexible camera calibration by viewing a plane from unknown orientations](https://ieeexplore.ieee.org/document/791289)。

张氏标定法 只考虑径向畸变，不考虑切向畸变，默认 $p_1=p_2=0$ .

由于世界坐标系是自定义的，因此对于棋盘格，我们定义棋盘格所在平面为 $Z_W=0$ ，原点位于整个棋盘格的左上角。令尺度因子 $s=Z_C$ ，得到：

$$
s \begin{pmatrix}
u \\ v \\ 1
\end{pmatrix}=
\mathbf{K}\; [\mathbf{r}_1\quad \mathbf{r}_2\quad\mathbf{r}_3\quad \mathbf{t}]\;
\begin{pmatrix}
X_W \\ Y_W \\ 0 \\ 1
\end{pmatrix}
=\mathbf{K}\; [\mathbf{r}_1\quad \mathbf{r}_2\quad \mathbf{t}]\;
\begin{pmatrix}
X_W \\ Y_W \\ 1
\end{pmatrix}
=\mathbf{H}\begin{pmatrix}
X_W \\ Y_W \\ 1
\end{pmatrix}
$$

称 $\mathbf{H}=\mathbf{K}\; [\mathbf{r}_1\quad \mathbf{r}_2\quad \mathbf{t}]\in\mathbb{R}^{3\times 3}$ 为单映性矩阵，**具有 8 个自由度**。这是因为 $\mathbf{H}$ 描述的是**齐次坐标** 下两个平面之间的投影变换，对 $\mathbf{H}$ 乘以任意一个非零常数 $\lambda\neq 0$ 不会改变投影后的归一化像素坐标，也即 $\mathbf{H}$ 和 $\lambda\mathbf{H}$ 表示的是同一个投影变换。因此齐次像素坐标 $[u,v,1]^T$ 前面的尺度因子 $s=Z_C$ 不需要考虑。这种性质称为**尺度不确定性**。

我们通常设定 $h_{33}=1\text{ or } \|\mathbf{H}\|^2=1$ 消除尺度不确定性，由此 $\mathbf{H}$ 只有 8 个自由度。

在棋盘格标定中，由于设定棋盘格平面 $Z_W=0$ 且棋盘格规格（例如 7x11, 15mm）已知，则对于拍摄的每一张照片，各个特征点（通产是棋盘格交叉处的角点）的世界坐标 $[X_W,Y_W,0]^T$ 和像素坐标 $[u,v,1]^T$ 都是已知的：

$$
\begin{pmatrix}
u \\ v \\ 1
\end{pmatrix}\sim
\begin{pmatrix}
h_{11} & h_{12} & h_{13} \\
h_{21} & h_{22} & h_{23} \\
h_{31} & h_{32} & h_{33} \\
\end{pmatrix}\begin{pmatrix}
X_W \\ Y_W \\ 1
\end{pmatrix}
$$

因此有：

$$
\begin{align*}
u&=\dfrac{h_{11}X_W+h_{12}Y_W+h_{13}}{h_{31}X_W+h_{32}Y_W+h_{33}} \\
v&=\dfrac{h_{21}X_W+h_{22}Y_W+h_{23}}{h_{31}X_W+h_{32}Y_W+h_{33}}
\end{align*}
$$

由于尺度不确定性，通常令 $h_{33}=1$ ，因此：

$$
\begin{align*}
u&=\dfrac{h_{11}X_W+h_{12}Y_W+h_{13}}{h_{31}X_W+h_{32}Y_W+1} \\
v&=\dfrac{h_{21}X_W+h_{22}Y_W+h_{23}}{h_{31}X_W+h_{32}Y_W+1}
\end{align*}
$$

整理得到：

$$
\begin{align*}
h_{11}X_W+h_{12}Y_W+h_{13}-h_{31}uX_W-h_{32}uY_W-u=0 \\
h_{21}X_W+h_{22}Y_W+h_{23}-h_{31}vX_W-h_{32}vY_W-v=0 \\
\end{align*}
$$

因此每个特征点可以提供 2 个这样的关于 $[h_{11},h_{12},\cdots,h_{32}]^T\in\mathbb{R}^{8}$ 的线性方程。理论上**至少需要 4 个特征点**，就可以得到一个形如 $\mathbf{A}_{8\times 8}\cdot[h_{11},h_{12},\cdots,h_{32}]^T=\mathbf{b}_{8\times 1}$ 的非齐次线性方程组，可唯一确定一个单映性矩阵 $\mathbf{H}$ . 考虑到真实场景中噪声和计算误差等因素，实际使用的棋盘格的格点数远多于4，得到一个超定方程组，使用优化方法求解，得到较为精确的近似解。

一张图像上的所有特征点对应一个单映性矩阵 $\mathbf{H}$ .

<br>

上文了解了如何得到单映性矩阵 $\mathbf{H}=[\mathbf{h}_1,\mathbf{h}_2,\mathbf{h}_3]$ ，下面讲解如何求解内参矩阵 $\mathbf{K}$ （先求内参是因为内参固定不变，更容易求解）。 由于 单映性矩阵 $\mathbf{H}$ 同时包含了内参 $\mathbf{A}$ 和外参，因此我们应想办法消去外参的影响。

由于旋转矩阵 $\mathbf{R}$ 为正交矩阵，满足约束（Ⅰ）正交性 $\mathbf{r}_1^T\mathbf{r}_2=0$ .（Ⅱ）基底模长为1： $\mathbf{r}_1^T\mathbf{r}_1=\mathbf{r}_2^T\mathbf{r}_2=1$ . 带入得到：

$$
\begin{gather*}
\mathbf{h}_1^T\mathbf{K}^{-T}\mathbf{K}^{-1}\mathbf{h}_2=0 \\
\mathbf{h}_1^T\mathbf{K}^{-T}\mathbf{K}^{-1}\mathbf{h}_1=\mathbf{h}_2^T\mathbf{K}^{-T}\mathbf{K}^{-1}\mathbf{h}_2=1
\end{gather*}
$$

令：

$$
\mathbf{B}=\mathbf{K}^{-T}\mathbf{K}^{-1}=
\begin{pmatrix}
\dfrac{1}{f_x^2} & -\dfrac{\gamma}{f_x^2 f_y} & \dfrac{v_0 \gamma - u_0 f_y}{f_x^2 f_y} \\
-\dfrac{\gamma}{f_x^2 f_y} & \dfrac{\gamma^2}{f_x^2 f_y^2} + \dfrac{1}{f_y^2} & -\gamma \dfrac{v_0 \gamma - u_0 f_y}{f_x^2 f_y^2} - \dfrac{v_0}{f_y^2} \\
\dfrac{v_0 \gamma - u_0 f_y}{f_x^2 f_y} & -\gamma \dfrac{v_0 \gamma - u_0 f_y}{f_x^2 f_y^2} - \dfrac{v_0}{f_y^2} & \dfrac{(v_0 \gamma - u_0 f_y)^2}{f_x^2 f_y^2} + \dfrac{v_0^2}{f_y^2} + 1
\end{pmatrix}=
\begin{pmatrix}
b_{11} & b_{12} & b_{13} \\
b_{21} & b_{22} & b_{23} \\
b_{31} & b_{32} & b_{33}
\end{pmatrix}
$$

显然 $\mathbf{B}$ 为对称矩阵，其有用元素只有 6 个（对角线的一侧）。之前得到的约束式 $\mathbf{h}_i^T\mathbf{B}\mathbf{h}_j=0\text{ or }1,\;i,j=1,2$ 展开可以得到关于 $[b_{11},b_{12},b_{13},b_{22},b_{23},b_{33}]$ 的齐次线性方程组。因此**至少需要 3 张图片**，可以求解出矩阵 $\mathbf{B}$ （带有比例因子）。根据矩阵 $\mathbf{B}$ 进一步求解内参：

$$
\begin{cases}
f_x = \sqrt{s / b_{11}} \\
f_y = \sqrt{s b_{11} / (b_{11}b_{22} - b_{12}^2)} \\
u_0 = s v_0 / f_y - b_{13} f_x^2 / s \\
v_0 = (b_{12}b_{13} - b_{11}b_{23}) / (b_{11}b_{22} - b_{12}^2) \\
\gamma = -b_{12} f_x^2 f_y / s \\
s = b_{33} - [b_{13}^2 + v_0 (b_{12}b_{13} - b_{11}b_{23})] / b_{11}
\end{cases}
$$

可见我们似乎可以直接求得尺度因子 $s$ . 但是根据 $\|\mathbf{r}_1\|=\|s\mathbf{K}^{-1}\mathbf{h}_1\|=1$ 我们又有 $s=\dfrac{1}{\|\mathbf{K}^{-1}\mathbf{h}_1\|}$ . 为什么出现两个计算结果？

> 实际情况下，数据中是存在噪音的，所以计算得到的旋转矩阵 $\mathbf{R}$ ，并不一定能满足旋转矩阵的性质。所以通常根据奇异值分解来得到旋转矩阵 $\mathbf{R}$ .
>
> 上述的推导结果是基于理想情况下的解，从理论上证明了张氏标定算法的可行性。但在实际标定过程中，一般使用最大似然估计进行优化。

假设拍摄了 $n$ 张标定图片，每张图片里有 $m$ 个棋盘格角点。三维空间点 $X$ 在图片上对应的二维像素为 $x$ ，三维空间点经过相机内参 $M$ ，外参 $\mathbf{R},\mathbf{t}$ 变换后得到的二维像素为 $x'$ ，类似于一个函数，输入参数和世界坐标，输出一个二维投影图像坐标 $(x', y')$ . 假设噪声是独立同分布的，我们通过最小化 $x,\;x'$ 的位置来求解上述最大似然估计问题：

$$
\sum_{i=1}^{n} \sum_{j=1}^{m} \left\| x_{ij} - x'(M, R_i, t_i, X_j) \right\|^2
$$

考虑透镜畸变的影响，由于径向畸变的影响相对较明显，所以主要考虑径向畸变参数。根据经验，通常只考虑径向畸变的前两个参数 $k_1, k_2$ 即可（增加更多的参数会使得模型变的复杂且不稳定）。实际求解中，通常把 $k_1, k_2$ 也作为参数加入上述函数一起进行优化，待优化函数为：

$$
\sum_{i=1}^{n} \sum_{j=1}^{m} \left\| x_{ij} - x'(M, k_1, k_2, R_i, t_i, X_j) \right\|^2
$$

另一种说法：每一个单映性矩阵 $\mathbf{H}$ 可以得到关于内参矩阵 $\mathbf{K}$ 的 2 个线性约束方程，而内参矩阵 $\mathbf{K}$ 有 5 个未知数，因此需要**至少 3 组不同的单映性矩阵** $\mathbf{H}$ ，也就是**至少 3 张不同位姿的图片**，能够求得内参矩阵 $\mathbf{K}$ .

最后根据 $\mathbf{K},\mathbf{H}$ 求出外参：

$$
\begin{align*}
\mathbf{r}_1&=s\mathbf{K}^{-1}\mathbf{h}_1 \\
\mathbf{r}_2&=s\mathbf{K}^{-1}\mathbf{h}_2 \\
\mathbf{r}_3&=\mathbf{r}_1\times\mathbf{r}_2 \\
\mathbf{t}&=s\mathbf{K}^{-1}\mathbf{h}_3
\end{align*}
$$

## Toolbox

常见的内参标定方法：

（1）MATLAB 标定程序

MATLAB 界面上方工具栏中，选择 "APP"，找到 "Camera Calibrator" 和 "Stereo Camera Calibrator"。其中 "Camera Calibrator" 用于单目相机标定，"Stereo Camera Calibrator" 用于双目相机标定。

（2）OpenCV 代码标定

以下为内参标定的一个示例代码，取自《数字图像处理》课程大作业中的 [GitHub: digital-image-project/src/calibrate.py](https://github.com/DerrickMarcus/digital-image-project/blob/main/src/calibrate.py)

```py title="camera_calibration.py"
import glob

import cv2
import numpy as np


def detect_corners(
    images_path: str, board_size: tuple[int, int], square_size: float
) -> tuple[list[np.ndarray], list[np.ndarray], tuple[int, int]]:
    """Detect chessboard corners in calibration images and prepare object/image points.

    Args:
        images_path (str): Glob pattern to calibration images.
        board_size (tuple[int, int]): Number of corners in chessboard (cols, rows).
        square_size (float): Size of one square edge on the board.

    Returns:
        object_points (list[np.ndarray]): 3D points in real world space for each image.
        image_points (list[np.ndarray]): 2D points in image plane for each image.
        image_shape (tuple[int, int]): Shape of calibration images (width, height).
    """
    obj_p = np.zeros((board_size[1] * board_size[0], 3), np.float32)
    obj_p[:, :2] = np.indices((board_size[0], board_size[1])).T.reshape(-1, 2)
    obj_p *= square_size

    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    image_shape: tuple[int, int] = (0, 0)

    for fname in glob.glob(images_path):
        img = cv2.imread(fname)
        if img is None:
            continue
        if image_shape == (0, 0):
            image_shape = (img.shape[1], img.shape[0])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        found, corners = cv2.findChessboardCorners(
            gray,
            board_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        if not found:
            print(f"Warning: Chessboard not found in {fname}")
            continue

        cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
        )

        object_points.append(obj_p)
        image_points.append(corners)

        cv2.drawChessboardCorners(img, board_size, corners, found)
        cv2.namedWindow("Detected Corners", cv2.WINDOW_NORMAL)
        cv2.imshow("Detected Corners", img)
        cv2.waitKey(1000)

    cv2.destroyAllWindows()

    return object_points, image_points, image_shape


def calibrate_camera(
    object_points: list[np.ndarray],
    image_points: list[np.ndarray],
    image_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Perform camera calibration to compute intrinsic matrix and distortion coefficients.

    Args:
        object_points (list[np.ndarray]): 3D world points from detect_corners().
        image_points (list[np.ndarray]): 2D image points from detect_corners().
        image_shape (tuple[int, int]): (width, height) of calibration images.

    Returns:
        camera_matrix (np.ndarray): Intrinsic parameters matrix (3x3).
        dist_coeffs (np.ndarray): Distortion coefficients (k1, k2, p1, p2, k3).
    """
    _, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, image_shape, None, None
    )

    total_error = 0
    for i in range(len(object_points)):
        projected, _ = cv2.projectPoints(
            object_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs
        )
        error = cv2.norm(image_points[i], projected, cv2.NORM_L2) / len(projected)
        total_error += error
    mean_error = total_error / len(object_points)
    print(f"Calibration completed. Mean reprojection error: {mean_error:.4f}px")

    return camera_matrix, dist_coeffs


if __name__ == "__main__":
    pattern = "images/chessboard/*.jpg"
    board_size = (10, 7)
    square_size = 25
    obj_p, img_p, shape = detect_corners(pattern, board_size, square_size)
    print(f"Detected {len(obj_p)} images with corners.")
    K, d = calibrate_camera(obj_p, img_p, shape)
    np.savez("src/calib_params.npz", camera_matrix=K, dist_coeffs=d)
```

## Practice

现在有一台机器人 SLAM 设备，传感器为激光雷达 LiDAR 和相机 Camera。相机的内参矩阵 $\mathbf{K}$ 和 `LiDAR -> Camera` 的外参矩阵 ${}^C_L\mathbf{T}$ 已经标定好，LiDAR 相对于世界坐标系的位姿 pose （因为以 LiDAR 中心为系统的原点）可以由 SLAM 节点发布的消息获取，记为 ${}^W_L\mathbf{T}$ 。现在对相机拍摄到的图片进行 YOLO 物体识别，假设只识别 bottle 这一类，且一段时间内图像中只出现同一个 bottle，我们需要对识别到的 bottle 物体在图像中位置的变化，估算出 bottle 在世界坐标系中的坐标 $(X_W,Y_W,Z_W)$ . 其中 bottle 物体在图像中的位置定义为 YOLO 检测框的中心像素坐标 $(u,v)$ .

### Analysis

首先，我们应求出世界坐标系到相机坐标系 `World -> Camera` 的变换 ${}^C_W\mathbf{T}={}^C_L\mathbf{T}{}^L_W\mathbf{T}={}^C_L\mathbf{T}\left({}^W_L\mathbf{T}\right)^{-1}$ . 取出其中的旋转矩阵和平移向量 $\mathbf{R},\mathbf{t}$ .

注意这里的世界坐标系并不是选定的棋盘格，而是由 SLAM 节点决定的，我们不能默认 $Z_W=0$ . 计算世界坐标系到像素坐标系 `World -> Pixel` 的变换：

$$
Z_C\begin{pmatrix}
u \\ v \\ 1
\end{pmatrix}=
\mathbf{K}\left[\mathbf{R}\begin{pmatrix}
X_W \\ Y_W \\ Z_W
\end{pmatrix}+\mathbf{t}\right]=
\mathbf{K}\;(\mathbf{R}\;|\;\mathbf{t})\begin{pmatrix}
X_W \\ Y_W \\ Z_W \\ 1
\end{pmatrix}=
\mathbf{A}\begin{pmatrix}
X_W \\ Y_W \\ Z_W \\ 1
\end{pmatrix}
$$

记 $\mathbf{A}=\mathbf{K}\;(\mathbf{R}\;|\;\mathbf{t})\in\mathbb{R}^{3\times 4}$ . 消去尺度因子 $Z_C$ ，得到：

$$
u=\frac{a_{11}X_W+a_{12}Y_W+a_{13}Z_W+a_{14}}{a_{31}X_W+a_{32}Y_W+a_{33}Z_W+a_{34}} ,\quad
v=\frac{a_{21}X_W+a_{22}Y_W+a_{23}Z_W+a_{24}}{a_{31}X_W+a_{32}Y_W+a_{33}Z_W+a_{34}}
$$

写为矩阵形式：

$$
\begin{pmatrix}
a_{11}-ua_{31} & a_{12}-ua_{32} & a_{13}-ua_{33} \\
a_{21}-va_{31} & a_{22}-va_{32} & a_{23}-va_{33}
\end{pmatrix}
\begin{pmatrix}
X_W \\ Y_W \\ Z_W
\end{pmatrix}+
\begin{pmatrix}
a_{14}-ua_{34} \\ a_{24}-va_{34}
\end{pmatrix}=\mathbf{0}
$$

可以写为 $\mathbf{M}\mathbf{p}=\mathbf{b},\;\mathbf{M}\in\mathbb{R}^{2\times 3},\;\mathbf{b}\in\mathbb{R}^2$ 的形式。这是一个欠定方程。因此只有一帧图像和位姿的情况下，无法确定物体的真实坐标，只能确定物体所在的方向。这也就是**单目相机无法测量深度信息**的原因。

如果有多帧图像和位姿，也就是有 $N$ 个 $\mathbf{M},\mathbf{b}$ ，记为  $\mathbf{M}_1,\cdots,\mathbf{M}_N,\mathbf{b}_1,\cdots,\mathbf{b}_N$ ，我们就可以写出很多个这样的方程组竖向堆叠，合成为一个：

$$
\begin{pmatrix}
\mathbf{M}_1 \\ \vdots \\ \mathbf{M}_N
\end{pmatrix}\mathbf{p}=
\begin{pmatrix}
\mathbf{b}_1 \\ \vdots \\ \mathbf{b}_N
\end{pmatrix}
$$

假设有相邻 2 帧图像和位姿，物体在图像中的像素坐标分别为 $(u,v),\;(u',v')$ ，其对应的 $\mathbf{A}$ 矩阵也是已知的，则：

$$
\begin{pmatrix}
a_{11}-ua_{31} & a_{12}-ua_{32} & a_{13}-ua_{33} \\
a_{21}-va_{31} & a_{22}-va_{32} & a_{23}-va_{33} \\
a'_{11}-u'a'_{31} & a'_{12}-u'a'_{32} & a'_{13}-u'a'_{33} \\
a'_{21}-v'a'_{31} & a'_{22}-v'a'_{32} & a'_{23}-v'a'_{33}
\end{pmatrix}
\begin{pmatrix}
X_W \\ Y_W \\ Z_W
\end{pmatrix}+
\begin{pmatrix}
a_{14}-ua_{34} \\ a_{24}-va_{34} \\ a'_{14}-u'a'_{34} \\ a'_{24}-v'a'_{34}
\end{pmatrix}=\mathbf{0}
$$

此时又得到一个超定方程。它的意义是：三维空间中有若干条形如 $Ax+By+Cz+D=0$ 的直线，理论上它们交于同一点即 $(X_W,Y_W,Z_W)$ . 但是由于误差和噪声的影响，这些直线不会交于一点，而是会有很多交点。我们可以使用优化方法求近似解，例如最常见的最小二乘法。

求得坐标 $(X_W,Y_W,Z_W)$ 之后，我们就可以通过发布消息、Rviz 显示来可视化识别目标物体和机器人的相对位置，此时我们还需要将其转换到 LiDAR 坐标系： ${}^LP={}^L_W\mathbf{T}\cdot(X_W,Y_W,Z_W)^T$ .

### Code

首先，在 YOLO 识别到物体之后，我们获取其对应检测框的中心像素坐标，记为物体在像素坐标系 `Pixel` 下的坐标 $(u,v)$ ：

```py
results = self.model.predict(
                cv_img,
                conf=self.conf,
                iou=self.iou,
                classes=[0],
                max_det=1,
                verbose=False,
            )[0]

if len(results.boxes) == 1:
    box = results.boxes[0]
    cx, cy, w, h = box.xywh[0]
    cx, cy = int(round(cx)), int(round(cy))
```

我们需要从 SLAM 节点输出的里程计消息中提取出旋转矩阵和平移向量。假设我们订阅 `/Odometry` 话题，消息类型为 `nav_msgs/Odometry` ，其父坐标系为 `#!cpp header.frame_id = "mapFrame"` ，子坐标系为 `#!cpp header.child_frame_id = "base_link_frame"` 。说明该里程计给出的是 `base_link` （通常是机器人质心或者 LiDAR 中心）在世界坐标系中的位姿。

由于我们需要用到同一时刻（或者接近同一时刻）的里程计和图像消息，因此我们使用 ROS 提供的 `message_filters` 模块，实现两个话题的同步。

```py
img_sub = message_filters.Subscriber(img_sub_topic, CompressedImage)
odom_sub = message_filters.Subscriber(odom_sub_topic, Odometry)
sync = message_filters.ApproximateTimeSynchronizer(
    [img_sub, odom_sub], queue_size=10, slop=0.05
)
sync.registerCallback(self.sync_callback)
```

上面的 `#!py slop=0.05` 参数的含义是“当两个消息时间戳之差小于 `slop` 这个阈值时，认为两个消息为同一时间的消息，即判定消息同步”。在我使用的测试包中，话题 `/Odometry` 发布的里程计消息频率约 10Hz，话题 `usb_cam/image_raw/compressed` 发布的图像消息频率约为 20Hz。频率不同意味着一段时间内，两者的消息数量不同，且相机消息数量几乎是里程计消息的 2 倍。在进行消息同步时， `message_filters` 会把两路消息分别押入 2 个队列，然后从队首开始匹配，匹配成功则把这两条消息取出并调用回调函数。如果队列已满，最旧的消息会自动丢弃并压入新的消息。理论上会有一半的图像消息被丢弃。由于图像消息的周期约为 50ms，里程计消息的频率约为 100ms，则设置 slop 略大于 50ms 比较合适。

通过 `#!bash rosmsg info nav_msgs/Odometry` 可以看出，该消息类型存储的位置信息是三维坐标 $(x,y,z)$ ，而姿态信息通过四元数 $(x,y,z,w)$ 存储，我们需要将四元数转换为旋转矩阵。

```text
std_msgs/Header header
  uint32 seq
  time stamp
  string frame_id
string child_frame_id
geometry_msgs/PoseWithCovariance pose
  geometry_msgs/Pose pose
    geometry_msgs/Point position
      float64 x
      float64 y
      float64 z
    geometry_msgs/Quaternion orientation
      float64 x
      float64 y
      float64 z
      float64 w
  float64[36] covariance
geometry_msgs/TwistWithCovariance twist
  geometry_msgs/Twist twist
    geometry_msgs/Vector3 linear
      float64 x
      float64 y
      float64 z
    geometry_msgs/Vector3 angular
      float64 x
      float64 y
      float64 z
  float64[36] covariance
```

```py
def odom_to_transformation(msg: Odometry):
    # 平移向量
    t = np.array([
        msg.pose.pose.position.x,
        msg.pose.pose.position.y,
        msg.pose.pose.position.z
    ])

    # 四元数 (x, y, z, w)
    q = [
        msg.pose.pose.orientation.x,
        msg.pose.pose.orientation.y,
        msg.pose.pose.orientation.z,
        msg.pose.pose.orientation.w
    ]
    T = tf.transformations.quaternion_matrix(q)
    T[0:3, 3] = t

    print(f"R =\n{T[0:3, 0:3]}")
    print(f"t ={t}")

    return T

T_lidar_world = odom_to_transformation(msg)
```

由此我们可以得到 ${}^W_L\mathbf{T}$ 记为变量 `T_lidar_world` . 而 `LiDAR -> Camera` 的外参矩阵 ${}^C_L\mathbf{T}$ 和相机内参 $\mathbf{K}$ 为测量值，我们将内参和外参矩阵写入 `.yaml` 配置文件，通过 `.launch` 文件参数读取传入节点，然后直接构造：

```yaml title="usb_cam.yaml"
camera_intrinsics:
  - [984.444138, 0.000000, 956.120990]
  - [0.000000, 982.875802, 521.503842]
  - [0.000000, 0.000000, 1.000000]
lidar_camera_extrinsics:
  - [0.0305422, -0.999457, -0.0123873, -0.0313723]
  - [0.629366, 0.0288575, -0.776573, -0.0853413]
  - [0.776509, 0.0159221, 0.629906, 0.00275532]
  - [0.0, 0.0, 0.0, 1.0]
```

```py
K = np.array(rospy.get_param("~camera_intrinsics"), dtype=np.float64)
T_lidar_camera = np.array(rospy.get_param("~lidar_camera_extrinsics"), dtype=np.float64)
# world -> camera
T_world_camera = T_lidar_camera @ np.linalg.inv(T_lidar_world)
A = K @ T_world_camera[:3, :] # A: 3x4
```

接下来根据 $(u,v), \mathbf{A}$ 构造线性方程组 $\mathbf{M}\mathbf{p}=\mathbf{b}$ ：

```py
def generate_coefficients(u, v, A):
    # Mp=b
    M = np.array(
        [
            [A[0, 0] - u * A[2, 0], A[0, 1] - u * A[2, 1], A[0, 2] - u * A[2, 2]],
            [A[1, 0] - v * A[2, 0], A[1, 1] - v * A[2, 1], A[1, 2] - v * A[2, 2]],
        ],
        dtype=np.float64,
    )
    b = -np.array([A[0, 3] - u * A[2, 3], A[1, 3] - v * A[2, 3]], dtype=np.float64)
    return M, b
```

获取多帧图像和位姿之后，我们得到多个这样的 `M, b` ，堆叠起来之后，利用 `np.linalg.lstsq` 最小二乘法求近似解。

```py
M_list, b_list = zip(
    *(generate_coefficients(u, v, A) for (u, v, A) in self.deque)
)
M_ls = np.vstack(M_list)
b_ls = np.hstack(b_list)
Xw, *_ = np.linalg.lstsq(M_ls, b_ls, rcond=None)
```

---

接下来，我们将从头构建一个 ROS 功能包，写入 python 代码并编译、构建。

```bash
cd ~/catkin_ws/src
catkin_create_pkg yolo_detector rospy std_msgs nav_msgs sensor_msgs geometry_msgs tf message_filters cv_bridge
```

最终的文件结构如下：

```text
.
|-- CMakeLists.txt
|-- config/
|   |-- Mid360_new.yaml
|   |-- usb_cam.yml
|   `-- yolo.yaml
|-- launch/
|   |-- mapping.launch
|   |-- record.launch
|   |-- test.launch
|   |-- usb_cam.launch
|   `-- yolo_detector.launch
|-- package.xml
|-- scripts/
|   `-- run.py
|-- setup.py
`-- src/
    `-- detector.py
```

首先编写 `./setup.py` 文件，使 Python 代码变为 Python 模块：

```py title="setup.py"
from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=['yolo_detector'],
    package_dir={'': 'src'}
)

setup(**d)
```

改动 `./CMakeLists.txt` 文件：

```cmake title="CMakeLists.txt"
cmake_minimum_required(VERSION 3.0.2)
project(yolo_detector)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  nav_msgs
  sensor_msgs
  geometry_msgs
  tf
  message_filters
  cv_bridge
)

catkin_python_setup()

catkin_package()

include_directories(${catkin_INCLUDE_DIRS})

catkin_install_python(
  PROGRAMS scripts/run_detector.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY config launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
```

我们编写源代码 `detector.py` 放在 `./src/` 目录中，经过编译构建后成为可导入的 Python 模块：

```py title="detector.py"
# -*- coding: utf-8 -*-
from collections import deque

import cv2
import message_filters
import numpy as np
import rospy
import tf.transformations
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, CompressedImage
from ultralytics import YOLO


def odom_to_transformation(msg: Odometry):
    R = tf.transformations.quaternion_matrix(
        [
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w,
        ]
    )
    T = np.eye(4)
    T[:3, :3] = R[:3, :3]
    T[:3, 3] = [
        msg.pose.pose.position.x,
        msg.pose.pose.position.y,
        msg.pose.pose.position.z,
    ]
    return T


def generate_coefficients(u, v, A):
    # Mp=b
    M = np.array(
        [
            [A[0, 0] - u * A[2, 0], A[0, 1] - u * A[2, 1], A[0, 2] - u * A[2, 2]],
            [A[1, 0] - v * A[2, 0], A[1, 1] - v * A[2, 1], A[1, 2] - v * A[2, 2]],
        ],
        dtype=np.float64,
    )
    b = -np.array([A[0, 3] - u * A[2, 3], A[1, 3] - v * A[2, 3]], dtype=np.float64)
    return M, b


class YoloDetector:
    def __init__(self):
        rospy.init_node("yolo_detector", anonymous=False)

        img_sub_topic = rospy.get_param("~img_sub_topic", "/usb_cam/image_raw/compressed")
        odom_sub_topic = rospy.get_param("~odom_sub_topic", "/Odometry")
        img_pub_topic = rospy.get_param("~img_pub_topic", "/usb_cam/image_detected")

        self.img_pub = rospy.Publisher(img_pub_topic, Image, queue_size=3)

        self.K = np.array(rospy.get_param("~camera_intrinsics"), dtype=np.float64)
        self.T_lidar_camera = np.array(
            rospy.get_param("~lidar_camera_extrinsics"), dtype=np.float64
        )

        model_path = rospy.get_param("~model_path", "yolov8n.pt")
        self.model = YOLO(model_path)
        self.model.fuse()
        self.model.to("cuda")
        self.model.half()

        self.conf = rospy.get_param("~conf", 0.5)
        self.iou = rospy.get_param("~iou", 0.5)
        self.show_window = rospy.get_param("~show_window", True)

        self.deque = deque(maxlen=rospy.get_param("~deque_size", 3))

        self.last_T = None
        self.thres_distance = rospy.get_param("~thres_distance", 1)
        self.thres_angle = rospy.get_param("~thres_angle", 1)

        img_sub = message_filters.Subscriber(img_sub_topic, CompressedImage)
        odom_sub = message_filters.Subscriber(odom_sub_topic, Odometry)
        sync = message_filters.ApproximateTimeSynchronizer(
            [img_sub, odom_sub], queue_size=30, slop=rospy.get_param("~time_sync_slop", 0.1)
        )
        sync.registerCallback(self.sync_callback)

        self.bridge = CvBridge()

        rospy.spin()

    def sync_callback(self, img_msg: CompressedImage, odom_msg: Odometry):
        cv_img = self.bridge.compressed_imgmsg_to_cv2(img_msg, "bgr8")
        if cv_img is None or cv_img.size == 0:
            rospy.logwarn("Receive empty image!")
            return

        # detect at most one person
        results = self.model.predict(
            cv_img,
            conf=self.conf,
            iou=self.iou,
            classes=[0],
            max_det=1,
            verbose=False,
        )[0]
        if len(results.boxes) == 0:
            rospy.logwarn("No objects detected.")
            return
        else:
            for box in results.boxes:
                rospy.loginfo(f"YOLO result: {box.cls.item()}, {box.conf.item()}, {box.xyxy.tolist()}.")

        # publish image with bounding box
        annotated = results.plot()
        if self.show_window:
            cv2.imshow("YOLO Detection", annotated)
            cv2.waitKey(1)
        img_pub_msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
        img_pub_msg.header = img_msg.header
        self.img_pub.publish(img_pub_msg)

        # get the center pixel coordinates (u,v)
        box = results.boxes[0]
        u, v, _, _ = box.xywh[0].cpu().numpy()
        u, v = int(np.round(u)), int(np.round(v))

        T_lidar_world = odom_to_transformation(odom_msg)
        T_world_camera = self.T_lidar_camera @ np.linalg.inv(T_lidar_world)

        if self.last_T is not None:
            d_T = np.linalg.inv(self.last_T) @ T_world_camera
            d_distance = np.linalg.norm(d_T[:3, 3])
            d_R = d_T[:3, :3]
            d_angle = np.arccos(np.clip((np.trace(d_R) - 1) / 2, -1.0, 1.0)) * 180 / np.pi
            if d_distance < self.thres_distance and d_angle < self.thres_angle:
                rospy.logwarn("Skip frame: camera motion too small")
                return

        A = self.K @ T_world_camera[:3, :]
        self.deque.append((u, v, A))
        self.last_T = T_world_camera.copy()
        rospy.loginfo(f"Add matrix (u, v, A), (u, v): {(u, v)}")

        # solve the equation
        if len(self.deque) == self.deque.maxlen:
            M_list, b_list = zip(
                *(generate_coefficients(u, v, A) for (u, v, A) in self.deque)
            )
            M_ls = np.vstack(M_list)
            b_ls = np.hstack(b_list)
            # rospy.loginfo(f"Solving MX=b:\nM: {M_ls},\nb: {b_ls}")
            rospy.loginfo("Solving MX=b")

            Xw, *_ = np.linalg.lstsq(M_ls, b_ls, rcond=None)
            rospy.loginfo(f"Detected person in World Frame: {Xw}")
            Xc = T_world_camera[:3, :3] @ Xw + T_world_camera[:3, 3]
            distance = np.linalg.norm(Xc)
            rospy.loginfo(f"Detected person in Camera Frame: {Xc}")
            rospy.loginfo(f"Direction: {'forward' if Xc[2] > 0 else 'backforward'}, {'right' if Xc[0] > 0 else 'left'}, {'down' if Xc[1] > 0 else 'up'}")
            rospy.loginfo(f"Distance: {distance}")
```

然后在 `./scripts/` 目录下创建 `run.py` 脚本文件，作为节点的启动入口：

```py title="run.py"
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
from yolo_detector.YoloDetector import YoloDetector

if __name__ == "__main__":
    try:
        YoloDetector()
    except rospy.ROSInterruptException as e:
        rospy.logerr(e)
```

在 `./config/` 目录下存放 `.yaml` 格式的配置文件，例如运行 YOLO 检测节点的配置文件 `./scripts/yolo.yaml` 文件：

```yaml title="yolo.yaml"
camera_intrinsics:
  - [984.444138, 0.000000, 956.120990]
  - [0.000000, 982.875802, 521.503842]
  - [0.000000, 0.000000, 1.000000]
lidar_camera_extrinsics:
  - [0.0305422, -0.999457, -0.0123873, -0.0313723]
  - [0.629366, 0.0288575, -0.776573, -0.0853413]
  - [0.776509, 0.0159221, 0.629906, 0.00275532]
  - [0.0, 0.0, 0.0, 1.0]
model_path: yolov8n.pt
img_sub_topic: /robot1/usb_cam/image_raw/compressed
odom_sub_topic: /robot1/Odometry
img_pub_topic: /yolo_detector/image_detected
conf: 0.4
iou: 0.5
show_window: true
deque_size: 4
thres_distance: 2
thres_angle: 2
time_sync_slop: 0.1
```

在 `./launch/` 目录下存放用于启动节点的 `.launch` 文件，例如：

启动 YOLO 检测节点的 `yolo_detector.launch` ：

```xml title="yolo_detector.launch"
<launch>
    <rosparam file="$(find yolo_detector)/config/yolo.yaml" command="load" ns="yolo_detector" />
    <node pkg="yolo_detector" type="yolo_detector.py" name="yolo_detector" output="screen"></node>
</launch>
```

录制 LiDAR 和相机图像消息为 rosbag 包的 `record.launch` ：

```xml title="record.launch"
<launch>
    <arg name="robot_id" default="robot1" />
    <arg name="bag_path" default="/home/nvidia/Documents/yolo.bag" />

    <!-- using relative path for topics -->
    <arg name="imu_topic" default="livox/imu" />
    <arg name="lidar_topic" default="livox/lidar" />
    <arg name="image_topic" default="usb_cam/image_raw/compressed" />

    <group ns="$(arg robot_id)">
        <include file="$(find livox_ros_driver2)/launch/msg_MID360.launch">
            <arg name="rviz_enable" value="false" />
        </include>

        <include file="$(find usb_cam)/launch/usb_cam.launch" />

        <node name="rosbag_record" pkg="rosbag" type="record"
        args="$(arg imu_topic) $(arg lidar_topic) $(arg image_topic) -O $(arg bag_path)"
        output="screen" />
    </group>

</launch>
```

启动建图节点、启动 YOLO 节点、播放 rosbag 进行测试的 `test.launch` ：

```xml title="test.launch"
<launch>
    <include file="$(find fast_lio_sam)/launch/mapping.launch" />
    <include file="$(find yolo_detector)/launch/yolo_detector.launch" />
</launch>
```

最后进行编译和构建：

```bash
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```
