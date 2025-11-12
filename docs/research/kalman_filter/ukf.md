# Unscented Kalman Filter

> The following content is referenced from:
>
> [进一步理解无迹卡尔曼滤波（UKF）-腾讯云开发者社区-腾讯云](https://cloud.tencent.com.cn/developer/article/2581696)

无迹卡尔曼滤波（Unscented Kalman Filter, UKF）同样是用于处理非线性问题的卡尔曼滤波变体。它的核心是使用**无迹变换**（Unscented Transformation, UT），相比于 EKF 的粗略线性化更加精确、鲁棒，相比于 EKF 推导雅可比矩阵也更容易实现。在 SLAM、目标跟踪、自动驾驶等领域，UKF 已经很大程度上取代了 EKF。

EKF 通过对非线性函数做一阶泰勒展开，用雅可比矩阵近似，这会带来一系列问题。首先，线性化阶段了高阶项，对于高度非线性系统不够精确；其次，需要手动推导雅可比矩阵，复杂且易出错；最后，系统非线性较强时误差会累积。而 UKF 采取的思想，不是用一个粗略的线性函数近似非线性函数，而是在状态分布上选取一组确定性样本点（Sigma Points），这组点能够完全不捕获当前状态分布的均值和方差，把他们通过非线性函数传播和变换，映射到新的状态空间，再从变换后的点重新估计均值和方差。这样“采样—传播—重构”的过程就是无迹变换。这样能够实现不计算雅可比矩阵就能得到捕捉高精度的非线性信息。

一般问题中我们都假设模型是满足高斯分布的，也就是一个云团“。EKF 的做法是只关注云团的中心点（均值），假设整个云团都沿着中心点的切线方向移动。UKF 的做法则是分为三部：

1. 选点：根据当前高斯分布的均值和协方差，选择一组有代表性的点（Sigma Points）。这些点位于均值的不同方向上，距离的远近由协方差决定。
2. 传播：每一个 Sigma Point 都经过非线性函数 $f(\cdot)$ 变换。该过程是精确传播。
3. 重构：根据传播后的点重新计算出一个新的高斯分布。

## UT

首先需要确定 Sigma Point 的选择策略。对于一个满足多维高斯分布的状态向量 $\boldsymbol{x}\sim\mathcal{N}(\bar{\boldsymbol{x}},\boldsymbol{P})\in\mathbb{R}^n$ ，一共构造 $2n+1$ 个 Sigma Point。

第 1 个点为状态向量的均值：

$$
\chi_0=\bar{\boldsymbol{x}}
$$

接下来的 $2n$ 个点沿着 $n$ 个维度各自的正负方向选取：

$$
\chi_i=\bar{\boldsymbol{x}}+\left[ \sqrt{(n+\lambda)\boldsymbol{P}} \right]_i,\quad \chi_{i+n}=\bar{\boldsymbol{x}}-\left[ \sqrt{(n+\lambda)\boldsymbol{P}} \right]_i,
\quad i=1,\cdots,n
$$

然后计算各个 Sigma Point 的均值权重和协方差权重:

$$
\begin{align*}
W_0^{(m)}&=\frac{\lambda}{n+\lambda},\quad W_0^{(c)}=\frac{\lambda}{n+\lambda}+(1-\alpha^2+\beta) \\
W_i^{(m)}&=W_i^{(c)}=\frac{1}{2(n+\lambda)}, \quad i=1,\cdots,n
\end{align*}
$$

其中 $\sqrt{\cdot}$ 表示矩阵平方根，通常是 Cholesky 分解 $P=L^TL,\;L=\sqrt{P}$ ， $[\cdot]_i$ 表示矩阵的第 $i$ 列。

> 注：Cholesky 分解条件是被分解矩阵应为对称正定矩阵。若 $P$ 非正定可添加小的抖动项 $\varepsilon I$ .

$\lambda$ 是一个缩放因子，用于控制 Sigma Point 采样的覆盖范围，通常定义为 $\lambda=\alpha(n+\kappa)-n$ ，其中散布度 $\alpha\in(0,1]$ 为主要缩放参数，一般取很小值如 $10^{-3}$ ， $\kappa$ 是次要缩放参数， $\beta$ 是关于分布的先验知识的参数，可以优化协方差计算精度，对于高斯分布 $\beta=2$ 是最优的。

> 对于实际问题，一般使用高斯分布模型，参数通常设置为 $\alpha=10^{-3},\;\beta=2,\;\kappa=0$ .

将每个 Sigma Point 送入非线性函数 $\mathcal{Y}_i=f(\chi_i)$ ，重建均值与方差：

$$
\bar{\boldsymbol{y}}=\sum_i W_i^{(m)}\mathcal{Y}_i,\quad
\boldsymbol{P}_y=\sum_i W_i^{(c)}(\mathcal{Y}_i-\hat{\boldsymbol{y}})(\mathcal{Y}_i-\hat{\boldsymbol{y}})^T+\boldsymbol{R}
$$

## UKF

对于不同时刻 $k$ ，状态 $\boldsymbol{x}_k$ 具有系统过程噪声 $\boldsymbol{w}_k\sim\mathcal{N}(\boldsymbol{0},\boldsymbol{Q}_k)$ ，观测量 $\boldsymbol{z}_k$ 具有观测噪声 $\boldsymbol{v}_k\sim\mathcal{N}(\boldsymbol{0},\boldsymbol{R}_k)$ ，构成非线性系统：

$$
\begin{cases}
\boldsymbol{x}_k &= f(\boldsymbol{x}_{k-1},\boldsymbol{u}_{k-1})+\boldsymbol{w}_{k-1} \\
\boldsymbol{z}_k &= h(\boldsymbol{x}_k)+\boldsymbol{v}_k
\end{cases}
$$

UKF 的具体步骤为：

（1）基于 $k-1$ 时刻的后验估计的均值和协方差 $\hat{\boldsymbol{x}}_{k-1|k-1},\;\boldsymbol{P}_{k-1|k-1}$ ，构造一组 Sigma Points $\boldsymbol{\chi}_{k-1}=[\chi_{k-1}^{(i)}]$ .

$$
\boldsymbol{\chi}_{k-1}=\begin{pmatrix}
\hat{\boldsymbol{x}}_{k-1|k-1} & \hat{\boldsymbol{x}}_{k-1|k-1}+\sqrt{(n+\lambda)\boldsymbol{P}_{k-1|k-1}} & \hat{\boldsymbol{x}}_{k-1|k-1}-\sqrt{(n+\lambda)\boldsymbol{P}_{k-1|k-1}}
\end{pmatrix}\in\mathbb{R}^{n\times(2n+1)}
$$

通过状态转移函数 $f(\cdot)$ 传播 Sigma Points： $\chi_{k|k-1}^{(i)}=f\left(\chi_{k-1}^{(i)},\boldsymbol{u}_k\right)$ ，这一步不是近似，没有误差。

（2）Predict

预测状态均值，对传播后的点加权求和：

$$
\hat{\boldsymbol{x}}_{k|k-1}=\sum_i W_i^{(m)}\chi_{k|k-1}^{(i)}
$$

预测状态协方差，对传播后的协方差加权求和并加上过程噪声：

$$
\boldsymbol{P}_{k|k-1}=\sum_i W_i^{(c)}\left(\chi_{k|k-1}^{(i)}-\hat{\boldsymbol{x}}_{k|k-1}\right) \left(\chi_{k|k-1}^{(i)}-\hat{\boldsymbol{x}}_{k|k-1}\right)^T+\boldsymbol{Q}_{k-1}
$$

（3）Update

*Optional*：使用预测的先验分布 $\mathcal{N}\left(\hat{\boldsymbol{x}}_{k|k-1},\;\boldsymbol{P}_{k|k-1}\right)$ 重新生成一组 Sigma Points，一般会更加准确。

实际的观测量为 $\boldsymbol{z}_k$ . 将 Sigma Points 通过观测模型传播 $\mathcal{Z}_k^{(i)}=h\left(\chi_{k|k-1}^{(i)}\right)$ .

观测量的均值和协方差也是加权求和形式，分别为：

$$
\hat{\boldsymbol{z}}_k=\sum_i W_i^{(m)}\mathcal{Z}_k,\quad
\boldsymbol{P}_{z}=\sum_i W_i^{(c)}\left(\mathcal{Z}_k^{(i)}-\hat{\boldsymbol{z}}_k\right) \left(\mathcal{Z}_k^{(i)}-\hat{\boldsymbol{z}}_k\right) ^T+\boldsymbol{R}_{k}
$$

状态量-观测量互协方差为：

$$
\boldsymbol{P}_{xz}=\sum_i W_i^{(c)}\left(\chi_{k|k-1}^{(i)}-\hat{\boldsymbol{x}}_{k|k-1}\right) \left(\mathcal{Z}_k^{(i)}-\hat{\boldsymbol{z}}_k\right)^T
$$

计算卡尔曼增益 $\boldsymbol{K}_k=\boldsymbol{P}_{xz}\boldsymbol{P}_z^{-1}$ .

更新状态均值和协方差：

$$
\hat{\boldsymbol{x}}_{k|k-1}=\hat{\boldsymbol{x}}_{k-1|k-1}+\boldsymbol{K}(\boldsymbol{z}_k-\hat{\boldsymbol{z}}_k),\quad \boldsymbol{P}_{k|k}=\boldsymbol{P}_{k|k-1}-\boldsymbol{K}_k\boldsymbol{P}_z\boldsymbol{K}_k^T
$$

---

EKF、UKF 的对比：

| 特性     | EKF                                                          | UKF                                                          |
| -------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 核心方法 | 一阶泰勒展开（局部线性化）                                   | 无迹变换（采样逼近）                                         |
| 精度     | 一阶精度，线性化误差大，特别是对于强非线性系统               | 二阶精度，能更准确地捕获非线性变换后的均值和协方差，精度高于 EKF |
| 计算量   | 较低（但需要计算雅可比矩阵）                                 | 较高，需要传播 2L+1 个点，但对现代计算机通常可接受           |
| 实现难度 | 数学推导复杂，需要手动推导并雅可比矩阵 $\boldsymbol{F},\boldsymbol{H}$，容易出错 | 实现简单，只需提供非线性函数 $f(\cdot),h(\cdot)$ 的黑箱实现，无需求导 |
| 鲁棒性   | 对模型误差和强非线性敏感，容易发散                           | 更鲁棒，对非线性系统表现稳定，更不易发散                     |
| 适用场景 | 非线性程度较低、比较平滑的系统                               | 强非线性系统（如剧烈机动的目标跟踪）                         |
