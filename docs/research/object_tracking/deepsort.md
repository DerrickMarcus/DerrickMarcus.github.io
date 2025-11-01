# DeepSORT

> [ICIP 2017] Simple Online and Realtime Tracking with a Deep Association Metric
>
> arXiv: <https://arxiv.org/abs/1703.07402>
>
> IEEE: <https://ieeexplore.ieee.org/document/8296962>
>
> Code: <https://github.com/nwojke/deep_sort>

前述的 SORT 只用几何信息（预测框与检测框的 IoU/位置）做关联，而没有用到目标的特征，在遮挡和交会时容易发生 ID switch。因此 DeepSORT 增加了外观特征（ReID embedding），使得相似目标在遮挡后仍保持 ID，并使用匹配级联（matching cascade）和门控马氏距离使得数据关联更加稳定。

## Track Handling and State Estimation

与 SORT 类似，构建常速度、线性观测的标准卡尔曼滤波模型。检测框的状态变量为 $\mathbf{x}=[u,v,\gamma,h,\dot{x},\dot{y},\dot{\gamma},\dot{h}]^T$ ，观测变量为 $\mathbf{z}=[u,v,\gamma,h]^T$ 。其中，此处使用宽高比 $\gamma$ 和高度 $h$ ，而非 SORT 使用的面积 $s$ .

对于每一个目标的跟踪轨迹 $k$ ，我们需要计数它从成功关联以来的帧数，也就是它连续关联失败的帧数，如果超过预设阈值 $A_{\max}$ ，认为它已经离开场景，终止该目标的跟踪。

对于无法与现有目标跟踪轨迹关联的检测，尝试分配新的 ID，如果连续 3 帧都能关联成功，就保留此目标。

## Assignment Problem

为了构建分配问题的代价、运用匈牙利算法，我们结合目标的**运行**信息和**外观**信息进行分析。

对于运动信息，使用卡尔曼预测状态和新到来的测量之间的平方马氏距离（Mahalanobis distance）：

$$
d^{(1)}(i,j)=(\mathbf{d}_j-\mathbf{y}_i)^T\mathbf{S}_i^{-1}(\mathbf{d}_j-\mathbf{y}_i)^T
$$

原文中是这么说的：

> ... where we denote the projection of the $i$-th track distribution into measurement space by $(\boldsymbol{y}_i,\boldsymbol{S}_i)$ and the $j$-th bounding box detection by $\boldsymbol{d}_j$ . The Mahalanobis distance takes state estimation uncertainty into account by measuring how many standard deviations the detection is away from the mean track location.

这里具体解释一下 $\mathbf{y}_i,\;\mathbf{S}_i$ 的含义。由于卡尔曼滤波中的状态变量是在状态空间中，我们需要通过观测矩阵 $\mathbf{H}$ 将预测的状态分布从状态空间映射到测量空间 $(\mathbf{x}_i,\mathbf{P}_i)\to(\mathbf{y}_i,\mathbf{S}_i)$ ，也就是：

$$
\mathbf{y}_i=\mathbf{H}\mathbf{x}_i,\quad \mathbf{S}_i=\mathbf{H}\mathbf{P}_i\mathbf{H}^T+\mathbf{R}
$$

因此， $\mathbf{y}_i$ 代表第 $i$ 个轨迹预测观测均值（即卡尔曼滤波预测的测量输出）， $\mathbf{S}_i$ 代表第 $i$ 个轨迹预测在测量空间中的协方差矩阵（即卡尔曼滤波的不确定性）。

这个马氏距离衡量了检测框距离预测分布的中心有多少个标准差。马氏距离在计算预测与检测之间差距时，会考虑状态估计的不确定性，比欧氏距离更加合理，能够动态适应“预测置信度”：

- 若某个预测位置很确定（协方差小），即预测框范围很“窄”，那么偏离一点点就意味着很大的马氏距离。
- 若某个预测位置很不确定（协方差大），允许更大误差。

为了排除不太可能的匹配，在 95% 置信度区间内，对马氏距离进行阈值处理：

$$
b^{(1)}(i,j)=\mathbb{I}[d^{(1)}(i,j)\leqslant t^{(1)}],\quad t^{(1)}=\chi_4^2(0.95)=9.4877
$$

> While the Mahalanobis distance is a suitable association metric when motion uncertainty is low, in our image-space problem formulation the predicted state distribution obtained from the Kalman filtering framework provides only a rough estimate of the object location. In particular, unaccounted camera motion can introduce rapid displacements in the image plane, making the Mahalanobis distance a rather uninformed metric for tracking through occlusions. Therefore, we integrate a second metric into the assignment problem.

这段话的意思是，目标运动的不确定性较低时，使用马氏距离较为合适。但是在图像坐标系中，卡尔曼滤波预测的状态往往知识一个粗略估计，当存在相机抖动、不规则运动时，目标会在图像上出现大幅度漂移，此时马氏距离的区分程度变差，难以在遮挡或者快速移动场景中有效工作。

因此我们引入外观度量（appearance metric）：

- 对每一个检测到的边界框 $\mathbf{d}_j$ ，经过一个预训练 CNN（ReID 网络）提取特征向量 $\mathbf{r}_j$ ，保证 L2 归一化 $\|\mathbf{r}_j\|=1$ .
- 对每条轨迹（track） $k$ 维护一个特征库 $\mathcal{R}_k=\{\mathbf{r}_k^{(i)}\}_{k=1}^{L_k}$ 保存最近的 $L_k=100$ 帧该目标的特征。保存多帧是为了即使目标被短暂遮挡，也能用旧特征重新识别回来。
- 对于第 $i$ 条轨迹、第 $j$ 条观测，计算特征 $\mathbf{r}_j$ 与轨迹历史特征集 $\mathcal{R}_i$ 中所有特征的余弦距离最小值，作为该匹配对的外观距离 $d^{(2)}(i,j)=\min\{1-\mathbf{r}_j\cdot\mathbf{r}_k^{(i)} \mid \mathbf{r}_k^{(i)}\in\mathcal{R}_i \}$ . （余弦距离越小，代表两个特征越相似。这里用1减去，是因为要求最小代价）
- 再使用阈值判断 $b^{(2)}(i,j)=\mathbb{I}[d^{(2)}(i,j)\leqslant t^{(2)}]$ .

<br>

马氏距离反映了运动位置信息，适合短期预测；余弦距离提供外观特征，适合长期保持、应对遮挡问题。最后使用它们的加权求和来构建一个配对的最终代价：

$$
c_{ij}=\lambda d^{(1)}(i,j)+(1-\lambda)d^{(2)}(i,j)
$$

同时，这两个距离必须**同时**在门控范围内 $b_{ij}=b_{ij}^{(1)}b_{ij}^{(2)}$ ，这个代价 $c_{ij}$ 才有效。

在实际测试中，如果存在明显的相机运动，设置 $\lambda=0$ 较为合适，虽然最终代价只用到了外观距离，但是马氏距离仍然用于排除不太可能的分配，仍然在起作用。

## Matching Cascade

> Instead of solving for measurement-to-track associations in a global assignment problem, we introduce a cascade that solves a series of subproblems. To motivate this approach, consider the following situation: When an object is occluded for a longer period of time, subsequent Kalman filter predictions increase the uncertainty associated with the object location. Consequently, probability mass spreads out in state space and the observation likelihood becomes less peaked. Intuitively, the association metric should account for this spread of probability mass by increasing the measurement-to-track distance. Counterintuitively, when two tracks compete for the same detection, the Mahalanobis distance favors larger uncertainty, because it effectively reduces the distance in standard deviations of any detection towards the projected track mean. This is an undesired behavior as it can lead to increased track fragmentations and unstable tracks. Therefore, we introduce a matching cascade that gives priority to more frequently seen objects to encode our notion of probability spread in the association likelihood.

文中指出这样一个问题：当一个目标被遮挡时间越来越长时，卡尔曼滤波预测的位置越来越不准确，预测误差会不断积累，协方差（不确定性）变大。也就是 $\mathbf{S}_i$ 变大、 $\mathbf{S}_i^{-1}$ 变小、马氏距离“虚假地”变小，算法产生“错觉”，误以为当前的检测框和旧的轨迹很匹配。

DeepSORT 提出的思路是，与其像 SORT 那样一次性在所有轨迹上做全局匹配，不如优先考虑最近更新过的轨迹（预测较准的），然后再考虑比较老的轨迹（预测不准的）。

伪代码为：

```yaml
Listing 1: Matching Cascade

Input:
  Track indices T = {1,...,N}
  Detection indices D = {1,...,M}
  Maximum age Amax

1: Compute cost matrix C = [c_ij] using Eq.5        # 综合代价矩阵
2: Compute gate matrix B = [b_ij] using Eq.6        # 可行匹配（门控）
3: Initialize set of matches M ← ∅
4: Initialize unmatched detections U ← D
5: for n ∈ {1,...,Amax} do
6:     Select tracks by age Tn ← {i ∈ T | a_i = n}
7:     [x_i, y_i] ← min_cost_matching(C, Tn, U)     # 对这些轨迹局部匈牙利匹配
8:     M ← M ∪ {(i,j)} | b_ij ≠ 0
9:     U ← U \ {j | ∃ i, (i,j) ∈ M}                 # 移除已匹配检测
10: end for
11: return M, U
```

具体的流程为：

1. 准备代价矩阵 $\mathbf{C}=(c_{ij})$ 和门控矩阵 $\mathbf{B}=(b_{ij})$ ， $b_{ij}=1$ 才表示该配对满足阈值条件。
2. $\mathcal{M}$ 保存已经配对的“轨迹预测框-检测框”， $\mathcal{U}$ 保存还未匹配的检测框索引。
3. 按轨迹的年龄 $n$ （距离上次匹配成功的帧数）从小到大逐级匹配
    - 取出轨迹中的“同龄者” $\mathcal{T}_n=\{i\in\mathcal{T}\mid a_i=n\}$ .
    - 局部匈牙利算法：最小化代价矩阵的子集 $\mathbf{C}[\mathcal{T}_n,\mathcal{U}]$ .
    - 把得到的匹配对加入集合 $\mathcal{M}$ ，把本次匹配到的检测框从 $\mathcal{U}$ 中移除。
4. 级联匹配后，再做一次 IoU 匹配，用于还未匹配的、“年龄=1”的轨迹，

级联匹配策略按轨迹“新鲜度”分层，从最新匹配的轨迹开始逐步匹配，避免被遮挡时间长、预测不准的轨迹抢占检测框。这一机制显著减少了 ID Switch，是 DeepSORT 的关键稳定性来源。

## Deep Appearance Descriptor

本方法的在线跟踪使用最近邻查询（nearest neighbor queries），无需额外的度量学习。使用 MARS 数据训练一个 CNN，得到具有较好区分度的特征嵌入。

网络的最后一层做 BatchNorm + L2 Norm，输出一个 128 维向量，将向量投影到单位超球面，消除尺度影响，且此时余弦距离=点积。
