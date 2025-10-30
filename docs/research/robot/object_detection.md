# Target Detection (YOLO+SLAM)

> 书接上回：在 [Research - Camera Calibration](./camera_calibration.md) 中，我们探讨了一个实际的工程问题：一个装有单目相机的 SLAM 小车系统，在已知车体相对于 map 系下的位姿、相机相对于车体的外参、相机图像经过 YOLO 检测目标之后，推算出检测到目标的实际坐标（map 系下坐标，或车体坐标）？在之前的解答中，我们使用了非常直接的求解方法，但是在实际测试中发现，效果不是很理想。另外，从理论上来说，单次求解使用的帧数越多，结果越稳定，但是由于多个帧的方程是叠加起来的，帧数越多维数越大，求解速度会变慢。因此，本博客尝试使用另外一种方法：射线法，试图得到更加稳定的解。

## 射线法求解

假设：

1. 图像通过 YOLO 模型，得到检测框，以检测框中心像素坐标 $(u,v)$ 作为检测目标物位置。
2. 去畸变后，相机内参矩阵为 $\mathbf{K}\in\mathbb{R}^{3\times 3}$ ，车体到相机的静态外参为 ${}_L^C\mathbf{T}\in\mathbb{R}^{4\times 4}$ .
3. 从里程计消息中，提取出车体到世界系的位姿变换 ${}_L^W\mathbf{T}\in\mathbb{R}^{4\times 4}$ .
4. 检测目标物在世界系中静止不动，在世界系、车体系、相机系中的坐标分别为 $\mathbf{X}_W,\mathbf{X}_L,\mathbf{X}_C$ . 我们需要先求解 $\mathbf{X}_W$ ，因为它是常数，然后可以变换到车体系中用于可视化。

首先，目标物从像素系到相机系的变换为：

$$
Z_C\begin{pmatrix}
u \\ v \\1
\end{pmatrix}=\mathbf{K}\cdot\mathbf{X}_C=\mathbf{K}\begin{pmatrix}
X_C \\ Y_C \\ Z_C
\end{pmatrix}
$$

忽略系数 $Z_C$ ，得到 $\mathbf{X}_C=\mathbf{K}^{-1}\cdot[u,v,1]^T$ ，因为我们不关心它的模长，只关心它所在的射线，这条射线起点为相机系原点（光心），经过目标物。对其归一化得到该射线方向的单位向量：

$$
\mathbf{d}_C=\frac{\mathbf{K}^{-1}\cdot[u,v,1]^T}{\|\mathbf{K}^{-1}\cdot[u,v,1]^T\|}
$$

然后计算相机系到世界系的位姿变换：

$$
{}_C^W\mathbf{T}={}_L^W\mathbf{T} \cdot {}_C^L\mathbf{T}={}_L^W\mathbf{T} \cdot \left({}_L^C\mathbf{T}\right)^{-1}
$$

为避免矩阵求逆，我们直接写出 ${}_C^W\mathbf{T}$ 的表达式：

$$
\begin{gather*}
{}_C^W\mathbf{T}=\begin{pmatrix}
\mathbf{R}_{CW} & \mathbf{t}_{CW} \\
\mathbf{0} & 1
\end{pmatrix},\quad
{}_L^W\mathbf{T}=\begin{pmatrix}
\mathbf{R}_{LW} & \mathbf{t}_{LW} \\
\mathbf{0} & 1
\end{pmatrix},\quad
{}_L^C\mathbf{T}=\begin{pmatrix}
\mathbf{R}_{LC} & \mathbf{t}_{LC} \\
\mathbf{0} & 1
\end{pmatrix},
\\ \mathbf{R}_{CW}=\mathbf{R}_{LW}\left(\mathbf{R}_{LC}\right)^T,\quad \mathbf{t}_{CW}=\mathbf{t}_{LW}-\mathbf{R}_{CW}\mathbf{t}_{LC}
\end{gather*}
$$

然后把单位方向向量变换到世界系： $\mathbf{d}_W=\mathbf{R}_{CW}\mathbf{d}_C$ ，同时，相机的光心坐标即为平移向量 $\mathbf{C}_W=\mathbf{t}_{CW}$ .

因此，在世界系下，一帧图像+位姿，对应一条“以相机光心为起点，经过目标物的射线”，即 $\mathbf{C}_W+\lambda\mathbf{d}_W,\;\lambda>0$ .

在相机随车体运动过程中，每一帧都对应一条这样的射线，理论上这些射线相交于同一点，也就是目标点所在位置 $\mathbf{X}_W$ . 实际中由于噪声和误差的存在，这些若干条射线可能有多个交点，因此使用最小二乘法求最优解。

对于世界系下的任意一点 $\mathbf{X}$ ，若以射线上的某一点为起点，以该点 $\mathbf{X}$ 为终点组成的向量，可以表示为 $\mathbf{X}-\mathbf{C}$ ，该向量在射线上投影对应的向量为 $\mathbf{d}_W\mathbf{d}_W^T(\mathbf{X}-\mathbf{C})$ （因为 $\mathbf{d}_W$ 也是单位向量），因此点 $\mathbf{X}$ 到射线的最短距离向量，即为残差 $\mathbf{r}$ ：

$$
\mathbf{r}=(\mathbf{X}-\mathbf{C})-\mathbf{d}_W\mathbf{d}_W^T(\mathbf{X}-\mathbf{C})=(\mathbf{I}-\mathbf{d}_W\mathbf{d}_W^T)(\mathbf{X}-\mathbf{C})
$$

因此，点 $\mathbf{X}$ 到射线上的最短距离即为残差的模长 $\|\mathbf{r}\|$ . 令 $\mathbf{P}=\mathbf{I}-\mathbf{d}_W\mathbf{d}_W^T$ ，有 $\mathbf{P}^T\mathbf{P}=\mathbf{P}$ .

$$
\|\mathbf{r}\|^2=(\mathbf{X}-\mathbf{C})^T\mathbf{P}^T\mathbf{P}(\mathbf{X}-\mathbf{C})=(\mathbf{X}-\mathbf{C})^T\mathbf{P}(\mathbf{X}-\mathbf{C})
$$

我们希望求解出来的坐标 $\mathbf{X}_W$ 到各条直线的距离之和最小，写为加权最小二乘形式：

$$
L(\mathbf{X})=\sum_i w_i\|\mathbf{r}_i\|^2
$$

转换为求解方程：

$$
\frac{\partial L}{\partial \mathbf{X}}=2\sum_i w_i\mathbf P_i(\mathbf X-\mathbf C_i)=0 \implies
\left(\sum_i w_i\mathbf P_i\right)\mathbf X=\sum_i w_i\mathbf P_i\mathbf C_i
$$

初始时刻设置 $\mathbf{A}=\mathbf{0},\mathbf{b}=\mathbf{0}$ . 在求解过程中，我们需要逐个构造 $\mathbf{P}_i,\mathbf{C}_i$ 并加入已有的 $\mathbf{A},\mathbf{b}$ ，最终求解 $\mathbf{A}\mathbf{X}=\mathbf{b}$ 可得到目标物的世界系坐标 $\mathbf{X}_W$ ，可以再变换到车体系 $\mathbf{X}_L={}_W^L\mathbf{T}\cdot\mathbf{X}_W=(\mathbf{R}_{LW})^T(\mathbf{X}_W-\mathbf{t}_{LW})$ ，方便进行可视化。

存在的问题：

1. 当车体自身纯旋转时，也就是相机光心不动（在相机和车体位于同一竖直方向时），此时即使是多帧图像和位姿，它们对应的也是同一个射线。
2. 当车体只前进，尤其是当目标物接近图像中心时，也就是相机光心几乎向着目标物前进，此时多帧图像和位姿对应的也是同一条射线。

## 地面假设

之前的推导中我们知道，在只有一帧图像+位姿的情况下，缺少深度信息，是无法确定人员在地图中的三维坐标的。但是如果我们做出如下假设：**场景中人员始终静止站在地面上**，可以转化为条件：也就是人员脚部的地图坐标的高度 $Z_W$ 是一个定值，等于地面的 $Z$ 分量坐标 $Z_p$。

经过 YOLO 推理后的检测框，将其底边中心的坐标作为人员脚部对应的像素坐标 $(u,v)$ . 也就是说，我们将求解的坐标限制在一个平面 $Z_W=Z_p$ 上，既能减小自由度，又能尽量减小前述方法中射线交点对噪声的敏感度（因为两个夹角很小的射线可能收到噪声影响，交点变得很远，但是如果先求出两条射线在平面 $Z_W=Z_p$ 的交点再求中心点，会更加准确）。

首先，我们还是根据已知的 LiDAR-相机外参，即 LiDAR->Camera 变换 $\mathbf{T}_{LC}$ 和由里程计得到的 LiDAR->World 变换 $\mathbf{T}_{LW}$ ，计算出世界坐标系到相机坐标系的 World->Camera 变换：

$$
\mathbf{T}_{WC}=\mathbf{T}_{LC}\mathbf{T}_{WL}=\mathbf{T}_{LC}\left(\mathbf{T}_{LW}\right)^{-1}
$$

令 $\mathbf{T}_{WC}=[\mathbf{R}\mid \mathbf{t}]$ ，可以直接写出其表达式以避免对矩阵 $\mathbf{T}_{LW}$ 求逆：

$$
\mathbf{R} = \mathbf{R}_{LC}\left(\mathbf{R}_{LW}\right)^T, \quad \mathbf{t}=\mathbf{t}_{LC}-\mathbf{R}\cdot\mathbf{t}_{LW}
$$

然后得到世界坐标系（地图系）到像素坐标系的变换关系：

$$
Z_C\begin{pmatrix}
u \\ v \\ 1
\end{pmatrix}=
\mathbf{K}\left[\mathbf{R}\begin{pmatrix}
X_W \\ Y_W \\ Z_W
\end{pmatrix}+\mathbf{t}\right]=
\mathbf{K}\;(\mathbf{R}\mid\mathbf{t})\begin{pmatrix}
X_W \\ Y_W \\ Z_W \\ 1
\end{pmatrix}=
\mathbf{A}\begin{pmatrix}
X_W \\ Y_W \\ Z_W \\ 1
\end{pmatrix}
$$

其中 $\mathbf{K}$ 为相机内参矩阵，通过标定得到，是已知量。令 $Z_W=Z_p$ ，消去尺度因子 $Z_C$ ，得到：

$$
Z_C\begin{pmatrix}
u \\ v \\ 1
\end{pmatrix}=
\mathbf{A}\begin{pmatrix}
X_W \\ Y_W \\ Z_p \\ 1
\end{pmatrix}=
\begin{pmatrix}
a_{11} & a_{12} & a_{13} & a_{14} \\
a_{21} & a_{22} & a_{23} & a_{24} \\
a_{31} & a_{32} & a_{33} & a_{34} \\
\end{pmatrix}
\begin{pmatrix}
X_W \\ Y_W \\ Z_p \\ 1
\end{pmatrix}
$$

$$
u=\frac{a_{11}X_W+a_{12}Y_W+a_{13}Z_p+a_{14}}{a_{31}X_W+a_{32}Y_W+a_{33}Z_p+a_{34}} ,\quad
v=\frac{a_{21}X_W+a_{22}Y_W+a_{23}Z_p+a_{24}}{a_{31}X_W+a_{32}Y_W+a_{33}Z_p+a_{34}}
$$

写为矩阵形式：

$$
\begin{pmatrix}
a_{11}-ua_{31} & a_{12}-ua_{32} \\
a_{21}-va_{31} & a_{22}-va_{32}
\end{pmatrix}
\begin{pmatrix}
X_W \\ Y_W
\end{pmatrix}+
\begin{pmatrix}
a_{13}Z_p+a_{14}-ua_{33}Z_p-ua_{34} \\ a_{23}Z_p+a_{24}-va_{33}Z_p-va_{34}
\end{pmatrix}=\mathbf{0}
$$

由于我们限制人员坐标 $Z_W=Z_p$ ，因此一帧图像+位姿即可确定该人员的地图坐标 $(X_W,Y_W,Z_p)$ .

为了求解的稳定性和减小噪声的影响，我们还希望根据多帧坐标进行优化求解，动态寻找人员最可能的坐标。也就是我们存储多帧图像+位姿计算得到的二维平面坐标 $\mathbf{p}_i=(x_i,y_i)$ ，求解最可能的坐标 $\mathbf{p}$ 作为最终解。这实际上是一个估计问题。

### 最小化距离的平方和

最常用的为最小二乘准则，也就是寻找一个点，使得该点到所有观测点的欧氏距离平方和最小。在统计学上，如果假设噪声服从均值为零的高斯分布（正态分布），那么最小二乘准则得到的结果与最大似然估计的结果是完全一致的。目标函数为：

$$
L(\mathbf{p})=\sum_{i=1}^N w_i\|\mathbf{p}-\mathbf{p}_i\|^2=\sum_{i=1}^N w_i[(x-x_i)^2+(y-y_i)^2]
$$

目标函数的最小值有闭式解：

$$
\hat{\mathbf p}=\frac{\sum_i w_i\,\mathbf p_i}{\sum_i w_i}
$$

即为所有点的**重心**。

### 最小化距离之和

在最小二乘法中，离群点对目标函数的残差惩罚是平方律的，一个极端离群点可能会显著将均值“拉向”自己。我们希望减小极端离群点对均值的影响，因此使用 L1 的目标函数，在存在误识别/跳变/遮挡等偶发大误差时更稳健和鲁棒：

$$
L(\mathbf{p})=\sum_{i=1}^N w_i\|\mathbf{p}-\mathbf{p}_i\|=\sum_{i=1}^N w_i\sqrt{(x-x_i)^2+(y-y_i)^2}
$$

与最小二乘法不同，该目标函数的最小值没有一个简单的封闭解，因为其导数包含平方根，代数形式复杂。但是，这个函数是一个凸函数，这意味着它只有一个全局最小值，没有局部最小值。因此，我们可以通过迭代算法来稳定地找到这个解。

最常用的算法是 Weiszfeld 算法，它本质上是一种迭代重加权最小二乘法（Iteratively Reweighted Least Squares, IRLS）。

目标函数的梯度为：

$$
\nabla L(\mathbf{p})=\sum_{i=1}^N w_i \frac{\mathbf{p}-\mathbf{p}_i}{\|\mathbf{p}-\mathbf{p}_i\|}=0
$$

$$
\mathbf{p} =\frac{\sum_{i=1}^{N} \dfrac{w_i \mathbf{p}_i}{\|\mathbf{p} - \mathbf{p}_i\|}}{\sum_{i=1}^{N} \dfrac{w_i}{\|\mathbf{p} - \mathbf{p}_i\|}}
$$

该式天然构成了一个迭代式。因此我们设计迭代步骤为：

（1）初始化。

选择一个初始点 $\mathbf{p}^{(0)}$ ，可以选择第一个观测点，或者前两三个观测点的重心。

（2）迭代更新。当前第 $k$ 步的时候，用 $\mathbf{p}^{(k)}$ 估计 $\mathbf{p}^{(k+1)}$ ：

$$
\mathbf{p}^{(k+1)} = \frac{\sum_{i=1}^{N} \dfrac{w_i \mathbf{p}_i}{\|\mathbf{p}^{(k)} - \mathbf{p}_i\|}}{\sum_{i=1}^{N} \dfrac{w_i}{\|\mathbf{p}^{(k)} - \mathbf{p}_i\|}}
$$

（3）收敛条件。相邻两次迭代的结果差异较小 $\|\mathbf{p}^{(k+1)}-\mathbf{p}^{(k)}\|<\varepsilon$ .

### Kalman 滤波优化

上述方法中，无论每一帧独立求解，还是结合多帧使用最小二乘法联合求解，我们都能在每一帧算出来一个人员坐标。但实际调试过程中发现，由于几何约束关系较弱，算出来的人员坐标形成了一个“运动轨迹”，但是我们期望的是人员坐标是一个不变的常数。因此我们可以考虑使用一个二维静止模型的卡尔曼滤波器，对输出的一系列坐标进行平滑和限制。

设定：

状态变量为二维坐标 $[x,y]$ .

由于模型假定目标静止，也就是坐标恒定不变，因此状态转移矩阵为单位阵 $\boldsymbol{F}=\boldsymbol{I}_{2\times 2}$ .

由于观测变量仍为二维坐标 $[x,y]$ ，因此观测矩阵为单位阵 $\boldsymbol{H}=\boldsymbol{I}_{2\times 2}$ .

过程噪声协方差、测量噪声协方差，由于可以认为 X, Y 方向不相关，协方差为0，故协方差矩阵为对角矩阵，且两个方向的方差相同：

$$
\boldsymbol{Q}=\begin{pmatrix}
\sigma_1^2 & 0 \\
0 & \sigma_2^2
\end{pmatrix} \quad
\boldsymbol{R}=\begin{pmatrix}
\sigma_2^2 & 0 \\
0 & \sigma_2^2
\end{pmatrix}
$$

然后可写出以下代码：

```py
class KalmanFilter:
    def __init__(self, proc_var=1e-2, meas_var=1e-2, P_init=1.0):
        """
        proc_var: 过程噪声方差
        meas_var: 测量噪声方差
        P_init: 初始时刻的估计协方差
        """
        # 状态向量 [x,y]
        self.x = np.zeros((2,), dtype=np.float64)
        # 状态转移，单位阵
        self.F = np.eye(2, dtype=np.float64)
        # 观测矩阵，单位阵
        self.H = np.eye(2, dtype=np.float64)
        # 估计协方差
        self.P = np.eye(2, dtype=np.float64) * P_init
        # 过程噪声协方差
        self.Q = np.eye(2, dtype=np.float64) * proc_var
        # 测量噪声协方差
        self.R = np.eye(2, dtype=np.float64) * meas_var
        self.initialized = False

    def initiate(self, meas, P_init=None):
        """meas: shape (2,) 用第一次测量值初始化"""
        self.x = np.array(meas, dtype=np.float64)
        if P_init is not None:
            self.P = np.eye(2, dtype=np.float64) * P_init
        self.initialized = True

    def predict(self):
        # 状态外插
        self.x = self.F @ self.x
        # 协方差外插
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, meas, R_meas=None, thresh=9.0) -> bool:
        """meas: shape (2,), R_meas: shape (2,2), thresh"""
        if R_meas is not None:
            self.R = R_meas
        # 残差
        y = np.array(meas, dtype=np.float64) - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        # 求逆
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
        if (y.T @ S_inv @ y) > thresh:
            # 异常值，忽略
            rospy.logwarn("Kalman filter: measurement rejected.")
            return False
        # 卡尔曼增益
        K = self.P @ self.H.T @ S_inv
        # 状态更新
        self.x = self.x + K @ y
        # 协方差更新
        self.P = (np.eye(2) - K @ self.H) @ self.P @ (
            np.eye(2) - K @ self.H
        ).T + K @ self.R @ K.T
        return True
```
