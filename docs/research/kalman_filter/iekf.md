# Iterated Extended Kalman Filter

> The following content is referenced from:
>
> [进一步理解迭代扩展卡尔曼滤波（IEKF）-腾讯云开发者社区-腾讯云](https://cloud.tencent.com.cn/developer/article/2581697)
>
> [迭代扩展卡尔曼滤波(IEKF) - 知乎](https://zhuanlan.zhihu.com/p/141018958)

迭代扩展卡尔曼滤波（Iterated Extended Kalman Filter, IEKF）是对扩展卡尔曼滤波 EKF 的改进方法，用于解决非线性系统的状态估计问题。EKF 只做一次线性化和更新，而 IEKF 在更新步骤中反复迭代，每次迭代都基于一个新的、更准确的估计点进行线性化，直到收敛。它的核心思想是，既然 EKF 因为只在单个点（先验估计点）线性化而引入误差，那就在多个点上进行线性化，通过迭代来逼近真实状态，从而减少线性化误差。IEKF 的预测阶段和 EKF 相同，区别在于更新阶段。

EKF 存在的问题是，对非线性观测函数 $\boldsymbol{h}(\cdot)$ 只在先验估计 $\hat{\boldsymbol{x}}_{k|k-1}$ 处线性化一次。如果状态的先验估计距离状态的真值较远，或者 $\boldsymbol{h}(\cdot)$ 非线性化程度很高（泰勒展开高次项不可忽略），那么一次线性化会产生较大偏差。

IEKF 的思想是，在更新阶段反复迭代，使用最新的线性化点重新计算雅可比矩阵并更新状态，直到残差变小或增益收敛。这等价于在高斯分布先验下，对非线性测量做 Gauss-Newton 近似的最大后验（MAP）迭代。

## IEKF

IEKF 的滤波过程：

（1）Predict (same as EKF)

状态预测 $\hat{\boldsymbol{x}}_{k|k-1}=\boldsymbol{f}(\hat{\boldsymbol{x}}_{k-1|k-1},\boldsymbol{u}_{k-1})$

协方差预测 $\boldsymbol{P}_{k|k-1}=\boldsymbol{F}_{k-1}\boldsymbol{P}_{k-1|k-1}\boldsymbol{F}_{k-1}^T+\boldsymbol{W}_{k-1}\boldsymbol{Q}_{k-1}\boldsymbol{W}_{k-1}^T$

其中， $\boldsymbol{F}_{k-1} = \dfrac{\partial \boldsymbol{f}}{\partial \boldsymbol{x}} \bigg|_{\hat{\boldsymbol{x}}_{k-1|k-1}}$ 是状态转移雅可比矩阵， $\boldsymbol{Q}$ 是过程噪声协方差。

（2）Update with iteration

初始化 $\boldsymbol{x}^{(0)}=\hat{\boldsymbol{x}}_{k|k-1}$ ，循环 $i=0,1,\cdots$ 直到收敛或达到最大迭代次数：

- 计算当前迭代点 $\boldsymbol{x}^{(i)}$ 处的观测函数的雅可比矩阵 $\boldsymbol{H}^{(i)}=\dfrac{\partial \boldsymbol{h}}{\partial \boldsymbol{x}}\bigg|_{\boldsymbol{x}^{(i)}}$
- 计算迭代卡尔曼增益 $\boldsymbol{K}^{(i)}=\boldsymbol{P}_{k|k-1}\boldsymbol{H}^{(i)T}\left(\boldsymbol{H}^{(i)}\boldsymbol{P}_{k|k-1}\boldsymbol{H}^{(i)T}+\boldsymbol{R}_k\right)^{-1}$
- 状态更新 $\boldsymbol{x}^{(i+1)}=\boldsymbol{x}^{(i)}+\boldsymbol{K}^{(i)}\left[\boldsymbol{z}_k-\boldsymbol{h}(\boldsymbol{x}^{(i)})-\boldsymbol{H}^{(i)}\left(\hat{\boldsymbol{x}}_{k|k-1}-\boldsymbol{x}^{(i)}\right) \right]$
- 收敛条件 $\|\boldsymbol{x}^{(i+1)}-\boldsymbol{x}^{(i)}\|<\varepsilon$ 或 $i>N_{\max}$ ，否则 $i+1\gets i$ .

迭代完成后，最终的状态后验估计 $\hat{\boldsymbol{x}}_{k|k}=\boldsymbol{x}^{(i+1)}$ ，协方差后验估计 $\boldsymbol{P}_{k|k}=\left(\boldsymbol{I}-\boldsymbol{K}^{(i)}\boldsymbol{H}^{(i)}\right)\boldsymbol{P}_{k|k-1}$ .

## Gauss-Newton

> The following content is referenced from:
>
> [高斯-牛顿法(Guass-Newton Algorithm)与莱文贝格-马夸特方法(Levenberg–Marquardt algorithm)求解非线性最小二乘问题 | 会飞的大象](http://www.whudj.cn/?p=1122)

在 [Extended Kalman Filter](./ekf.md) 中我们提到过，最大后验估计问题等价于最小化损失函数 $L(\boldsymbol{x}_k)$ ：

$$
\hat{\boldsymbol{x}}_{k|k}^{\text{MAP}}=\arg\min_{\boldsymbol{x}_k}L(\boldsymbol{x}_k),\quad L(\boldsymbol{x}_k)=\frac{1}{2}\|\boldsymbol{x}_k-\hat{\boldsymbol{x}}_{k|k-1}\|^2_{\boldsymbol{P}_{k|k-1}^{-1}}+\frac{1}{2}\|\boldsymbol{z}_k-\boldsymbol{h}(\boldsymbol{x}_k)\|^2_{\boldsymbol{R}_k^{-1}}
$$

可以写成加权最小二乘的形式：

$$
L(\boldsymbol{x})=\min_{\boldsymbol{x}} \frac{1}{2}[\boldsymbol{r}(\boldsymbol{x})]^T\boldsymbol{W}\boldsymbol{r}(\boldsymbol{x}),\quad \boldsymbol{r}(\boldsymbol{x})=\begin{pmatrix}
\boldsymbol{x}-\hat{\boldsymbol{x}}_{k|k-1} \\
\boldsymbol{z}_k-\boldsymbol{h}(\boldsymbol{x})
\end{pmatrix},\quad
\boldsymbol{W}=\begin{pmatrix}
\boldsymbol{P}_{k|k-1}^{-1} & \boldsymbol{0} \\
\boldsymbol{0} & \boldsymbol{R}_k^{-1}
\end{pmatrix}
$$

若状态量维度 $\boldsymbol{x}:n_x\times 1$ ，观测量维度 $\boldsymbol{z}_k:n_z\times 1$ ，则 $\boldsymbol{r}(\boldsymbol{x}):(n_x+n_z)\times 1$ 为残差向量， $\boldsymbol{W}:(n_x+n_z)\times(n_x+n_z)$ 为权重/信息矩阵，用于给各个残差分量分配权重。通常取 $\boldsymbol{W}=\boldsymbol{R}^{-1}$ ，因为噪声越小、信息越可靠、权重越大。

这个加权最小二乘问题中，由于观测模型 $\boldsymbol{z}_k=\boldsymbol{h}(\boldsymbol{x}_k)$ 是非线性的，没有解析解，因此 IEKF 使用 Gauss-Newton 法来迭代求解这个非线性最小二乘问题：

1. 在每一次迭代中，非线性函数 $\boldsymbol{h}(\boldsymbol{x})$ 在当前状态 $\boldsymbol{x}^{(i)}$ 处线性化。
2. 线性化后，问题就转化为一个标准的线性加权最小二乘问题。
3. 通过重复线性化和求解线性最小二乘，状态估计 $\boldsymbol{x}$ 被逐步优化，直至收敛。

---

> 此处可回顾《数据与算法》课程中《基础数值算法-非线性方程与优化》。

我们讨论的问题是加权最小二乘问题，目标是最小化残差平方和（SSE）：

$$
\min_{\boldsymbol{x}} L(\boldsymbol{x})=\min_{\boldsymbol{x}} \frac{1}{2}\|\boldsymbol{r}(\boldsymbol{x})\|_{\boldsymbol{W}}^2=\min_{\boldsymbol{x}} \frac{1}{2}[\boldsymbol{r}(\boldsymbol{x})]^T\boldsymbol{W}\boldsymbol{r}(\boldsymbol{x}),\quad \boldsymbol{W}\succ 0
$$

其中 $\boldsymbol{x}\in\mathbb{R}^N,\;\boldsymbol{r}\in\mathbb{R}^m,\;\boldsymbol{W}\in\mathbb{R}^{m\times m}$ . 我们希望找到一个参数更新量 $\Delta \boldsymbol{x}$ 使得 $L(\boldsymbol{x}+\Delta\boldsymbol{x})$ 最小。

若对 $\boldsymbol{R}^{-1}$ 做 Cholesky 分解 $\boldsymbol{L}^T\boldsymbol{L}=\boldsymbol{R}^{-1},\;\boldsymbol{L}=\boldsymbol{R}^{-1/2}$ ，然后把残差左乘 $\boldsymbol{R}^{-1/2}$ 做白化 $\tilde{\boldsymbol{r}}=\boldsymbol{R}^{-1/2}\boldsymbol{r}$ ，目标变为 $\frac{1}{2}\|\tilde{\boldsymbol{r}}\|^2$ ，变换了残差函数，等价于取 $\boldsymbol{W}$ 为单位阵，更方便计算。

白化后堆叠残差，写为：

$$
\boldsymbol{r}(\boldsymbol{x})=\begin{pmatrix}
\boldsymbol{P}_{k|k-1}^{-1/2} (\boldsymbol{x}_k-\boldsymbol{x}_{k|k})\\
\boldsymbol{R}_k^{-1/2} (\boldsymbol{z}_k-h(\boldsymbol{x}_k))
\end{pmatrix}\in\mathbb{R}^{m+m},\quad \boldsymbol{W}=\boldsymbol{I}
$$

对于 Newton 法，通过二次近似目标函数来寻找极小值：

$$
L(\boldsymbol{x}+\Delta\boldsymbol{x})=L(\boldsymbol{x})+(\nabla L(\boldsymbol{x}))^T\Delta\boldsymbol{x}+\frac{1}{2}\Delta\boldsymbol{x}^T(\nabla^2 L(\boldsymbol{x}))\Delta\boldsymbol{x}
$$

令上式的梯度为 0，就能得到 Newton 法的迭代公式：

$$
\nabla^2 L(\boldsymbol{x}) \Delta\boldsymbol{x} = -\nabla L(\boldsymbol{x})
$$

牛顿法的问题在于 Hessian 矩阵的计算非常复杂，涉及到残差函数的二阶导数。Gauss-Newton 法是对 Newton 法的改进，用于（且仅用于）解决**非线性最小二乘问题**。它的优点是不需要直接计算 Hessian 矩阵，而是利用残差向量的一阶导数近似 Hessian 矩阵。

在当前参数点 $\boldsymbol{x}^{(i)}$ 附近，对残差函数 $\boldsymbol{r}(\boldsymbol{x})$ 做一阶泰勒展开：

$$
\boldsymbol{r}(\boldsymbol{x}^{(i)}+\Delta\boldsymbol{x})\approx \boldsymbol{r}(\boldsymbol{x}^{(i)})+\boldsymbol{J}_{\boldsymbol{r}}(\boldsymbol{x}^{(i)})\Delta\boldsymbol{x}
$$

代入最小化目标函数：

$$
\min_{\Delta\boldsymbol{x}}\|\boldsymbol{r}(\boldsymbol{x}^{(i)})+\boldsymbol{J}_{\boldsymbol{r}}(\boldsymbol{x}^{(i)})\Delta\boldsymbol{x}\|^2
$$

得到一个标准的线性最小二乘问题！直接写出正规方程：

$$
\left[\boldsymbol{J}_{\boldsymbol{r}}(\boldsymbol{x}^{(i)})\right]^T\boldsymbol{J}_{\boldsymbol{r}}(\boldsymbol{x}^{(i)})\Delta\boldsymbol{x}=-\left[\boldsymbol{J}_{\boldsymbol{r}}(\boldsymbol{x}^{(i)})\right]\boldsymbol{r}(\boldsymbol{x}^{(i)})
$$

---

带权重矩阵 $\boldsymbol{W}$ 的推导过程为：

记 $\boldsymbol{r}(\boldsymbol{x})$ 的 Jacobian 矩阵 $\boldsymbol{J}_{\boldsymbol{r}}(\boldsymbol{x})=\dfrac{\partial\boldsymbol{r}}{\partial\boldsymbol{x}}\in\mathbb{R}^{m\times n}$ ，每一行都是标量函数 $r_i(\boldsymbol{x})$ 的梯度（雅可比） $\nabla r_i(\boldsymbol{x})=\dfrac{\partial r_i}{\partial \boldsymbol{x}}=\left[\dfrac{\partial r_i}{\partial x_1},\cdots,\dfrac{\partial r_i}{\partial x_n}\right]\in\mathbb{R}^{1\times n}$ .

目标函数的梯度为：

> 若 $\boldsymbol{W}$ 对称，下面的公式直接成立，否则使用对称部分 $(\boldsymbol{W}+\boldsymbol{W}^T)/2$ .

$$
\nabla L(\boldsymbol{x})=\boldsymbol{J}_{\boldsymbol{r}}(\boldsymbol{x})^T\boldsymbol{W}\boldsymbol{r}(\boldsymbol{x})
$$

Hessian 矩阵为：

$$
\nabla^2 L(\boldsymbol{x})=\boldsymbol{J}_{\boldsymbol{r}}(\boldsymbol{x})^T\boldsymbol{W}\boldsymbol{J}_{\boldsymbol{r}}(\boldsymbol{x})+\sum_{i=1}^m \left[\boldsymbol{W}\boldsymbol{r}(\boldsymbol{x})\right]_i \nabla^2r_i(\boldsymbol{x})
$$

由于残差 $r_i(\boldsymbol{x})\approx 0$ ，因此忽略含二阶导数 $\nabla^2r_i(\boldsymbol{x})$ 的项，从而避免了计算 $\boldsymbol{r}(\boldsymbol{x})$ 的 Hessian 矩阵。因此目标函数的 Hessian 矩阵近似为：

$$
\nabla^2 L(\boldsymbol{x})\approx \boldsymbol{J}_{\boldsymbol{r}}(\boldsymbol{x})^T\boldsymbol{W}\boldsymbol{J}_{\boldsymbol{r}}(\boldsymbol{x})
$$

这里可以看到 Gauss-Newton 法与 Newton 法的不同，在于使用近似的 Hessian 矩阵降低计算复杂度，但是由于舍去了二阶项，仅适用于残差较小的情形。

带入 Newton 法迭代公式，就能得到 Gauss-Newton 法的**正规方程**：

$$
\left(\boldsymbol{J}_{\boldsymbol{r}}(\boldsymbol{x})^T\boldsymbol{W}\boldsymbol{J}_{\boldsymbol{r}}(\boldsymbol{x})\right)\Delta \boldsymbol{x}=-\boldsymbol{J}_{\boldsymbol{r}}(\boldsymbol{x})^T\boldsymbol{W}\boldsymbol{r}(\boldsymbol{x})
$$

更新方程为 $\boldsymbol{x}^{(i+1)}=\boldsymbol{x}^{(i)}-\left(\boldsymbol{J}_{\boldsymbol{r}}^T \boldsymbol{J}_{\boldsymbol{r}}\right)^{-1}\boldsymbol{J}_{\boldsymbol{r}}^T \boldsymbol{r}$ .

LM 算法：与 Newton 法一样，当初始值距离最小值较远时，Gauss-Newton 法并不能保证收敛。并且当 $\boldsymbol{J}_{\boldsymbol{r}}^T \boldsymbol{J}_{\boldsymbol{r}}$ 近似奇异的时候，Gauss-Newton 法也不能正确收敛。Levenberg-Marquart 算法是对上述缺点的改进。LM 方法是对梯度下降法与 Gauss-Newton 法进行线性组合以充分利用两种算法的优势。通过在 Hessian 矩阵中加入阻尼系数 $\lambda$ 来控制每一步迭代的步长以及方向：

$$
(\boldsymbol{H} + \lambda \boldsymbol{I})\varepsilon = -\boldsymbol{J}_{\boldsymbol{r}}^T \boldsymbol{r}
$$

- 当 $\lambda$ 增大时， $\boldsymbol{H} + \lambda \boldsymbol{I}$ 趋向于 $\lambda \boldsymbol{I}$ ，因此 $\varepsilon$ 趋向于 $-\frac{1}{\lambda} \boldsymbol{J}_{\boldsymbol{r}}^T \boldsymbol{r}$ ，也就是梯度下降法给出的迭代方向；
- 当 $\lambda$ 减小时，$\boldsymbol{H} + \lambda \boldsymbol{I}$ 趋向于 $\boldsymbol{H}$ ， $\varepsilon$ 趋向于 $-\boldsymbol{H}^{-1} \boldsymbol{J}_{\boldsymbol{r}}^T \boldsymbol{r}$ ，也就是 Gauss-Newton 法给出的方向。

---

对于 IEKF，使用 Gauss-Newton 求解的具体方法是，迭代过程中，在当前迭代点 $\boldsymbol{x}^{(i)}$ 对观测项做一次线性化，然后对增量 $\Delta \boldsymbol{x}$ 做最小二乘，求解 Gauss-Newton 正规方程。

首先，写出单时刻的 MAP 目标

$$
L(\boldsymbol{x}) = \frac{1}{2} \| \boldsymbol{x} - \hat{\boldsymbol{x}}_{k|k-1} \|^2_{\boldsymbol{P}_{k|k-1}^{-1}} + \frac{1}{2} \| \boldsymbol{z}_k - \boldsymbol{h}(\boldsymbol{x}) \|^2_{\boldsymbol{R}_k^{-1}}.
$$

在 $\boldsymbol{x}^{(i)}$ 处线性化测量：

$$
\boldsymbol{h}(\boldsymbol{x}) \approx \boldsymbol{h}(\boldsymbol{x}^{(i)}) + \boldsymbol{H}^{(i)}(\boldsymbol{x} - \boldsymbol{x}^{(i)}), \quad \boldsymbol{H}^{(i)} = \left. \frac{\partial \boldsymbol{h}}{\partial \boldsymbol{x}} \right|_{\boldsymbol{x}^{(i)}}
$$

得到**正规方程**：（中间推导过程略）

$$
\left(\boldsymbol{P}_{k|k-1}^{-1}+\boldsymbol{H}^{(i)T}\boldsymbol{R}_k^{-1}\boldsymbol{H}^{(i)}\right) \Delta\boldsymbol{x}=
\boldsymbol{P}_{k|k-1}^{-1}\left(\hat{\boldsymbol{x}}_{k|k-1}-\boldsymbol{x}^{(i)}\right)+\boldsymbol{H}^{(i)T}\boldsymbol{R}_k^{-1}\left(\boldsymbol{z}_k-\boldsymbol{h}(\boldsymbol{x}^{(i)})\right)
$$

每次迭代都在新的点 $\boldsymbol{x}^{(i)}$ 重新计算 $\boldsymbol{H}^{(i)}$ 与残差，求解上式得到 $\Delta \boldsymbol{x}$ ，更新 $\boldsymbol{x}^{(i+1)}=\boldsymbol{x}^{(i)}+\Delta\boldsymbol{x}$ ，直到收敛。

收敛后：

1. 卡尔曼增益为 $\boldsymbol{K}_k=\boldsymbol{P}_{k|k-1}\boldsymbol{H}^{(i)T}\left(\boldsymbol{H}^{(i)}\boldsymbol{P}_{k|k-1}\boldsymbol{H}^{(i)T}+\boldsymbol{R}_k\right)^{-1}$
2. 状态的后验估计为 $\hat{\boldsymbol{x}}_{k|k}=\boldsymbol{x}^{(i)}$
3. 协方差的后验估计为 $\boldsymbol{P}_{k|k}=\left( \boldsymbol{P}_{k|k-1}^{-1}+ \boldsymbol{H}^{(i)T}\boldsymbol{R}_k^{-1}\boldsymbol{H}^{(i)}\right)^{-1}$

---

总结：IEKF 有“卡尔曼增益求更新量”和“Gauss-Newton 正规方程求更新量”两种形式，他们在数学本质上是等价的。

对比：

| 特性     | EKF                                            | IEKF                                                         | UKF                                                          |
| :------- | :--------------------------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| 核心思想 | 一阶泰勒展开，在单个点（经验估计）线性化。     | 多次线性化，通过迭代线性化点逼近真实状态。                   | 无需求导，采用确定性采样（Sigma Point）来捕捉状态的均值和协方差。 |
| 精度     | 中等。对于中度非线性系统有效，线性化误差较大。 | 中等。通常比EKF精度更高，线性化误差显著减小。                | 高。通常与 IEKF 相当或更好，尤其对于强非线性系统。           |
| 计算成本 | 低。只需计算一次雅可比矩阵和更新。             | 中等。需要多次（通常3-5次）计算雅可比矩阵和更新。比 EKF 慢，但通常比 UKF 快。 | 高。需要传播 $2n+1$ 个 Sigma Point（ $n$ 为状态维数），计算成本随维数增长。 |
| 实现难度 | 简单。需要推导和计算雅可比矩阵。               | 中等。迭代循环需要小心处理收敛性。                           | 中等。无需推导雅可比矩阵，但需要设计Sigma Point 和权重。     |
| 适用场景 | 非线性程度不高，计算资源有限的系统。           | 高度非线性系统，且对精度要求高于计算效率的场景。观测模型非线性时特别有效。 | 高度非线性系统，特别是函数不可微或难以求导的情况。对状态分布的近似更好。 |
| 线性化点 | 固定的经验估计点。                             | 动态变化，从迭代至收敛。                                     | 无需线性化，直接对非线性函数进行传播。                       |
| 可靠性   | 一般。线性化误差可能导致发散。                 | 较好。更准确的线性化提高了稳定性。                           | 好。对非线性变换的统计特性捕捉得更准，更稳定。               |
