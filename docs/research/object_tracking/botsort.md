# BoT-SORT

> BoT-SORT: Robust Associations Multi-Pedestrian Tracking
>
> arXiv: <https://arxiv.org/abs/2206.14651>
>
> Code: <https://github.com/NirAharon/BoT-SORT>

BoT-SORT 把运动信息（Kalman 预测 + IoU）与外观信息（ReID 特征）融合，同时显式做相机运动补偿（CMC），并把 Kalman Filter 的状态向量从传统的“中心+面积+纵横比”改为“中心 + 宽高”，集成到 ByteTrack 中，提出了 BoT-SORT 和 BoT-SORT-ReID 两种跟踪器。两者的主干相同：

1. 基于 ByteTrack 的两阶段匹配（高置信 + 低置信检测）。
2. Kalman Filter 预测更新。
3. 相机运动补偿。
4. IoU 匹配 / 门控 / 匈牙利算法
5. 轨迹生命周期管理

区别是，在匹配时是否引入 ReID 外观特征。

## Kalman Filter

与 SORT、DeepSORT 都不同，BoT-SORT 选定的状态变量为 $\boldsymbol{x}=[x,y,w,h,\dot{x},\dot{y},\dot{w},\dot{h}]^T$ ，即二维坐标、宽度高度及其变化量。观测变量为 $\boldsymbol{z}=[x,y,w,h]^T$ .

状态转移矩阵和观测矩阵分别为：

$$
\boldsymbol{F}=\begin{pmatrix}
1 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 & 0 & 1 & 0 & 0 \\
0 & 0 & 1 & 0 & 0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1 & 0 & 0 & 0 & 1 \\
0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 \\
\end{pmatrix},\quad
\boldsymbol{H}=\begin{pmatrix}
1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
\end{pmatrix}
$$

采用与时间相关的过程噪声和测量噪声：

$$
\begin{align*}
\boldsymbol{Q}_k = \text{diag}(&(\sigma_p \hat{w}_{k-1|k-1})^2, (\sigma_p \hat{h}_{k-1|k-1})^2,\\
&(\sigma_p \hat{w}_{k-1|k-1})^2, (\sigma_p \hat{h}_{k-1|k-1})^2,\\
&(\sigma_v \hat{w}_{k-1|k-1})^2, (\sigma_v \hat{h}_{k-1|k-1})^2,\\
&(\sigma_v \hat{w}_{k-1|k-1})^2, (\sigma_v \hat{h}_{k-1|k-1})^2)
\\
\boldsymbol{R}_k = \text{diag}(&(\sigma_m \hat{w}_{k|k-1})^2, (\sigma_m \hat{h}_{k|k-1})^2,\\
&(\sigma_m \hat{w}_{k|k-1})^2, (\sigma_m \hat{h}_{k|k-1})^2)
\end{align*}
$$

过程噪声 $\boldsymbol{Q}_k$ 和测量噪声 $\boldsymbol{R}_k$ 是随边界框尺寸动态变化的对角阵， $\sigma_p,\sigma_v,\sigma_m$ 是需要设定的参数。

## Camera Motion Compensation

相机本身会运动的场景中，图像中的边界框可能发生显著变化。相机静止时，也可能受到风引起的振动或漂移。在未知相机运动数据（导航、IMU）或者相机内参矩阵时，可以使用相邻帧之间图像配准，近似看作相机运动在图像上的投影。

使用 OpenCV 中的全局运动估计（GMC，global motion compensation），这种稀疏配准技术允许忽略场景中动态物体，从而更好地估计背景的运动：

1. 提取图像关键点（keypoints）。
2. 使用稀疏光流（sparse optical flow）进行特征跟踪，并对平移分量做阈值处理。
3. 使用 RANSAC（Random Sample Consensus 随机抽样一致）估计相邻帧之间的仿射变换 $\boldsymbol{A}_{k-1}^k\in\mathbb{R}^{2\times 3}$ .

$$
\boldsymbol{A}_{k-1}^k=[\boldsymbol{M}_{2\times 2}\mid \boldsymbol{T}_{2\times 1}]=
\begin{pmatrix}
a_{11} & a_{12} & a_{13} \\
a_{21} & a_{22} & a_{23}
\end{pmatrix}
$$

使用该仿射变换可以预测第 $k-1$ 帧的边界框在第 $k$ 帧中的位置。其中平移部分 $\boldsymbol{T}$ 只影响边界框的中心坐标，不影响宽度、高度和其变化率。而旋转部分 $\boldsymbol{M}$ 对状态变量和噪声都有影响，但由于是线性变换，该仿射矩阵也可以同时同等地作用于整个状态变量，因此我们构造矩阵：

$$
\tilde{\boldsymbol{M}}_{k-1}^k=\begin{pmatrix}
\boldsymbol{M} & 0 & 0 & 0 \\
0 & \boldsymbol{M} & 0 & 0 \\
0 & 0 & \boldsymbol{M} & 0 \\
0 & 0 & 0 & \boldsymbol{M} \\
\end{pmatrix}\in\mathbb{R}^{8\times 8},\quad
\tilde{\boldsymbol{T}}_{k-1}^k=\begin{pmatrix}
a_{12} \\ a_{13} \\ 0 \\ \vdots \\ 0
\end{pmatrix}\in\mathbb{R}^8
$$

假设我们有了原始 KF 预测的状态变量和协方差 $\hat{\boldsymbol{x}}_{k|k-1},\;\boldsymbol{P}_{k|k-1}$ ，补偿后的预测应该修正为：

$$
\begin{align*}
\hat{\boldsymbol{x}}'_{k|k-1}&=\tilde{\boldsymbol{M}}_{k-1}^k \hat{\boldsymbol{x}}_{k|k-1}+\tilde{\boldsymbol{T}}_{k-1}^k \\
\boldsymbol{P}'_{k|k-1}&=\tilde{\boldsymbol{M}}_{k-1}^k \boldsymbol{P}_{k|k-1} \left(\tilde{\boldsymbol{M}}_{k-1}^{k}\right)^T
\end{align*}
$$

此结果作为最终的预测值，再与当前检测/观测 $\boldsymbol{z}$ 做更新：

$$
\begin{align*}
\boldsymbol{K}_k &= \boldsymbol{P}'_{k|k-1} \boldsymbol{H}_k^T \left(\boldsymbol{H}_k \boldsymbol{P}'_{k|k-1} \boldsymbol{H}_k^T + \boldsymbol{R}_k\right)^{-1}
\\
\hat{\boldsymbol{x}}_{k|k} &= \hat{\boldsymbol{x}}'_{k|k-1} + \boldsymbol{K}_k \left(\boldsymbol{z}_k - \boldsymbol{H}_k \hat{\boldsymbol{x}}'_{k|k-1}\right)
\\
\boldsymbol{P}_{k|k} &= \left(\boldsymbol{I} - \boldsymbol{K}_k \boldsymbol{H}_k\right) \boldsymbol{P}'_{k|k-1}
\end{align*}
$$

相机高速运动场景下，上述运动补偿至关重要。相机缓慢运动时，可忽略对 $\boldsymbol{P}_{k|k-1}$ 的补偿。

## IoU-ReID Fusion

类似于 DeepSORT，首先采用 FastReID 库的 BoT-SBS 模型（以 ResNeSt-50 主干），提取 128-D 或 256-D 的 embedding 特征嵌入。

每一条轨迹（tracklet）维护一个**外观状态** $e_i^k$ ，表示第 $i$ 条轨迹在第 $k$ 帧的平均外观特征，并使用指数滑动平均（EMA，exponential moving average）进行更新：

$$
e_i^k=\alpha e_i^{k-1} +(1-\alpha)f_i^k
$$

其中 $f_i^k$ 为当前帧检测到的边界框对应的 embedding， $\alpha=0.9$ 为动量，表示新特征只占 10% 的权重。这样可以平滑 ReID 特征。

对每一条轨迹 $i$ 的平均外观特征 $e_i^k$ 和检测 $j$ 的特征 $f_j^k$ 计算余弦相似度 $d_{i,j}^{\cos}=1-e_i^k\cdot f_j^k$ .

> 不同于 DeepSORT 把 IoU 距离和特征的余弦距离的加权平均作为最终代价，BoT-SORT 采用了更保守的方法：取最小值。

首先进行两次阈值处理，得到新的余弦距离：

$$
\hat{d}_{i,j}^{\cos}=\begin{cases}
0.5 \cdot d_{i,j}^{\cos}, & \text{if } (d_{i,j}^{\cos}<\theta_{emb})\wedge (d_{i,j}^{iou}<\theta_{iou})\\
1, & \text{else}
\end{cases}
$$

若 IoU 和余弦距离都在阈值内，说明重合程度较高，赋予一个更小的距离 $0.5 \cdot d_{i,j}^{\cos}$ ，否则直接设为 1，表示不可能匹配。原文中设置参数 $\theta_{emb}=0.25,\;\theta_{iou}=0.5$ .

最终的代价取最小值 $c_{ij}=\min\{d_{i,j}^{iou},\;\hat{d}_{i,j}^{\cos}\}$ ，这样一来：

- 如果外观相似且位置相近，则使用更小的那个距离。
- 如果外观和位置有任何一个不可信，则把外观代价设为 1，相当于仅使用 IoU 位置距离，退化为纯 IoU 匹配。

---

接下来介绍几个 Joint Detection and Embedding 思路的代表算法：JDE、FairMOT 等。
