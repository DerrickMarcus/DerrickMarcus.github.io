# Extended Kalman Filter

先前我们讨论的普通卡尔曼滤波，是为**线性系统**设计的最优估计算法，它假设系统满足以下线性高斯模型：

$$
\begin{cases}
\boldsymbol{x}_n &=\boldsymbol{F}\boldsymbol{x}_{n-1}+\boldsymbol{G}\boldsymbol{u}_{n-1}+\boldsymbol{\omega}_{n-1} ,\quad &\boldsymbol{w}_{n-1}\sim\mathcal{N}(\boldsymbol{0},\boldsymbol{Q}_{n-1}) \\
\boldsymbol{z}_n &= \boldsymbol{H}\boldsymbol{x}_n+\boldsymbol{v}_n ,\quad &\boldsymbol{v}_{n}\sim\mathcal{N}(\boldsymbol{0},\boldsymbol{R}_{n})
\end{cases}
$$

由此，KF 可以递推计算出每一时刻状态的最优估计（最小方差意义下的最优）。

但是现实系统往往是非线性的，例如机器人位姿计算、惯性导航、传感器的非线性测量，可能涉及到三角函数、平方、开方等非线性运算，因此需要使用扩展卡尔曼滤波（Extended Kalman Filter, EKF）来处理。

EKF 的核心思想是，用泰勒展开将非线性系统近似为局部线性系统（保留一阶项，忽略二阶以上的项），然后在每一个时刻应用普通卡尔曼滤波。

## EKF

<br>

非线性系统的模型为：

$$
\begin{cases}
\boldsymbol{x}_n &=\boldsymbol{f}(\boldsymbol{x}_{n-1},\boldsymbol{u}_{n-1})+\boldsymbol{\omega}_{n-1},\quad \boldsymbol{\omega}_{n-1}\sim\mathcal{N}(0,\boldsymbol{Q}_{n-1}) \\
\boldsymbol{z}_n &= \boldsymbol{h}(\boldsymbol{x}_n)+\boldsymbol{v}_n,\quad \boldsymbol{v}_n\sim\mathcal{N}(0,\boldsymbol{R}_n)
\end{cases}
$$

其中 $\boldsymbol{f}(\cdot)$ 为状态转移的非线性函数， $\boldsymbol{h}(\cdot)$ 为观测模型的非线性函数。噪声均为高斯白噪声。

在 $n-1$ 时刻，状态后验估计为 $\boldsymbol{x}_{n-1}\sim\mathcal{N}(\hat{\boldsymbol{x}}_{n-1|n-1},\boldsymbol{P}_{n-1})$ .

对非线性函数做泰勒展开：

$$
\begin{align*}
\boldsymbol{f}(\boldsymbol{x}_{n-1}) &=\boldsymbol{f}(\hat{\boldsymbol{x}}_{n-1|n-1},\boldsymbol{u}_{n-1})+\boldsymbol{F}_n(\boldsymbol{x}_n-\hat{\boldsymbol{x}}_{n-1|n-1})+o(\boldsymbol{x}_n-\hat{\boldsymbol{x}}_{n-1|n-1}) \\
\boldsymbol{h}(\boldsymbol{x}_n) &=\boldsymbol{h}(\hat{\boldsymbol{x}}_{n|n-1})+\boldsymbol{H}_n(\boldsymbol{x}_n-\hat{\boldsymbol{x}}_{n|n-1})+o(\boldsymbol{x}_n-\hat{\boldsymbol{x}}_{n|n-1})
\end{align*}
$$

计算估计点处的 Jacobian 矩阵：

$$
\boldsymbol{F}_n=\frac{\partial \boldsymbol{f}}{\partial \boldsymbol{x}}\bigg|_{\boldsymbol{x}=\hat{\boldsymbol{x}}_{n-1|n-1},\boldsymbol{u}=\boldsymbol{u}_{n-1}},\quad \boldsymbol{H}_n=\frac{\partial \boldsymbol{h}}{\partial \boldsymbol{x}}\bigg|_{\boldsymbol{x}=\hat{\boldsymbol{x}}_{n|n-1}}
$$

对噪声同样有：

$$
\boldsymbol{W}_{n-1}=\frac{\partial \boldsymbol{f}}{\partial \boldsymbol{\omega}_n}\bigg|_{\boldsymbol{x}=\hat{\boldsymbol{x}}_{n-1|n-1},\boldsymbol{u}=\boldsymbol{u}_{n-1}},\quad
\boldsymbol{V}_n=\frac{\partial \boldsymbol{h}}{\partial \boldsymbol{v}_n}
$$

> 注意，这里看似 $\boldsymbol{f}(\cdot),\;\boldsymbol{h}(\cdot)$ 与噪声无关，是直接和噪声相加的关系，使用 $\boldsymbol{f}(\cdot)$ 对 $\boldsymbol{w}$ 求偏导无意义。实际上是 $\boldsymbol{x}_n =\boldsymbol{f}(\boldsymbol{x}_{n-1},\boldsymbol{u}_{n-1})+\boldsymbol{W}_{n-1}\boldsymbol{\omega}_{n-1},\;\boldsymbol{W}_{n-1}=\dfrac{\partial \boldsymbol{x}_n}{\partial \boldsymbol{w}_{n-1}}$ ，而 $\boldsymbol{W}_n$ 实际上代表了系统对噪声的作用矩阵。只不过是对于一般问题，我们都简写为加性高斯白噪声，此时 $\boldsymbol{W}=\boldsymbol{V}=\boldsymbol{I}$ ，也就是可以忽略。
>
> 那么我们还可能会问，既然高斯分布的线性变换仍然是高斯分布，那么为什么不把 $\boldsymbol{Ww}_n$ 写作一个新的高斯噪声 $\boldsymbol{w}'_n$ ？这是因为有些时候，噪声只作用于部分维度的状态，或者与时间有关，此时 $\boldsymbol{W}\neq \boldsymbol{I}$ .
>
> 例如一维匀速度模型，状态 $[p,v]^T$ ，加速度具有白噪声 $w_a$ ：
>
> $$
> \begin{cases}
> p_k=p_{k-1}+v_{k-1}\Delta t+\frac{1}{2}w_a\Delta t^2 \\
> v_k=v_{k-1}+w_a \Delta t
> \end{cases} \implies
> \boldsymbol{W}=\begin{pmatrix}
> \frac{1}{2}\Delta t^2 \\ \Delta t
> \end{pmatrix},\quad \boldsymbol{P}_{k|k-1}=\cdots+\boldsymbol{W}\sigma_a^2 \boldsymbol{W}^T
> $$
>
> 这里噪声 $w_a$ 是一个标量，但是通过 $\boldsymbol{W}$ 映射为对协方差的贡献，或者说是将低维噪声作用到高维状态，将低维噪声映射到各状态分量（位置、速度）的协方差结构里。

则扩展卡尔曼滤波的方程为：

Predict：

1. 状态预测 $\hat{\boldsymbol{x}}_{n|n-1}=\boldsymbol{f}(\hat{\boldsymbol{x}}_{n-1|n-1},\boldsymbol{u}_{n-1})$
2. 协方差预测 $\boldsymbol{P}_{n|n-1}=\boldsymbol{F}_{n-1}\boldsymbol{P}_{n-1|n-1}\boldsymbol{F}_{n-1}^T+\boldsymbol{W}_{n-1}\boldsymbol{Q}_{n-1}\boldsymbol{W}_{n-1}^T$

Update：

1. 预测状态投影到观测空间 $\hat{\boldsymbol{z}}_{k|k-1}=\boldsymbol{h}(\hat{\boldsymbol{x}}_{n|n-1})$
2. 创新协方差 $\boldsymbol{S}_n=\boldsymbol{H}_n\boldsymbol{P}_{n|n-1}\boldsymbol{H}_n^T+\boldsymbol{V}_n\boldsymbol{R}_n\boldsymbol{V}_n^T$
3. 卡尔曼增益 $\boldsymbol{K}_n=\boldsymbol{P}_{n|n-1}\boldsymbol{H}_n^T\boldsymbol{S}_n^{-1}=\boldsymbol{P}_{n|n-1}\boldsymbol{H}_n^T\left(\boldsymbol{H}_n\boldsymbol{P}_{n|n-1}\boldsymbol{H}_n^T+\boldsymbol{V}_n\boldsymbol{R}_n\boldsymbol{V}_n^T \right)^{-1}$
4. 状态更新 $\hat{\boldsymbol{x}}_{n|n}=\hat{\boldsymbol{x}}_{n|n-1}+\boldsymbol{K}_n\left( \boldsymbol{z}_n-\boldsymbol{h}(\hat{\boldsymbol{x}}_{n|n-1}) \right)$
5. 协方差更新 $\boldsymbol{P}_{n|n} = (\boldsymbol{I}-\boldsymbol{K}_n \boldsymbol{H}_n)\boldsymbol{P}_{n|n-1}(\boldsymbol{I}-\boldsymbol{K}_n \boldsymbol{H}_n)^T+ \boldsymbol{K}_n\boldsymbol{R}_n\boldsymbol{K}_n^T$ ，简洁形式 $\boldsymbol{P}_{n|n} = (\boldsymbol{I}-\boldsymbol{K}_n \boldsymbol{H}_n)\boldsymbol{P}_{n|n-1}$

若协方差 $\boldsymbol{P}$ 非正定，可以对角线上加小抖动 $\varepsilon \boldsymbol{I}$ ，保持正定。

EKF 实际上是把非线性观测在先验估计 $\hat{\boldsymbol{x}}_{n|n-1}$ 处线性化：

$$
\boldsymbol{z}_n\approx \boldsymbol{h}(\hat{\boldsymbol{x}}_{n|n-1})+\boldsymbol{H}_n(\boldsymbol{x}_n-\hat{\boldsymbol{x}}_{n|n-1})+\boldsymbol{v}_k
$$

将当前时刻的观测近似为线性高斯模型，然后直接套用线性 KF 的最小均方误差（MMSE）公式，就能得到上面的 $\boldsymbol{K}_n,\;\hat{\boldsymbol{x}}_{n|n},\;\boldsymbol{P}_{n|n}$ .

EKF 相比于普通的 KF，适用范围更广，更贴近真实系统，滤波效果更好，但形式较为复杂，尤其是复杂函数可能难以写出显式导数和雅可比矩阵。

| 特性       | KF                                  | EKF                                                        |
| :--------- | :---------------------------------- | :--------------------------------------------------------- |
| 适用系统   | 严格线性系统                        | 平滑的弱非线性系统                                         |
| 模型要求   | 线性模型                            | 支持非线性模型                                             |
| 核心操作   | 直接使用模型矩阵 $F,H$ 进行线性运算 | 需要对非线性模型 $f,h$ 进行线性化，求雅可比矩阵            |
| 最优性     | 在线性高斯假设下是最优估计          | 是次优估计，精度取决于线性化近似的准确性                   |
| 计算复杂度 | 较低                                | 较高，因为每个预测和更新周期都需要实时计算雅可比矩阵       |
| 实现难度   | 简单，一旦确定 $F,H,Q,R$ 即可       | 复杂，需要推导非线性函数的偏导数以得到，代码实现也更复杂   |
| 鲁棒性     | 对线性系统鲁棒                      | 对模型误差和初始误差更敏感，在强非线性或近似不佳时容易发散 |

## Appendix

> 之前在 [Linear Kalman Filter](./lkf.md) 中我们说到，KF 是线性高斯模型下直接给出后验高斯的均值与协方差，因为此时 MMSE 等价于 MAP。而 EKF 的本质是，在非线性模型下，在局部近似为线性高斯，给出局部后验高斯的条件均值（是对 MMSE 的近似）。

若当前时刻 $k$ 的单次观测为 $\boldsymbol{z}_k$ ，当前时刻以前所有观测组成的观测集为 $\mathcal{Z}_k$ ，则 $\mathcal{Z}_k=(\mathcal{Z}_{k-1},\boldsymbol{z}_k)$ ，根据 Bayes 定理：

$$
p(\boldsymbol{x}_k\mid\mathcal{Z}_k)=p(\boldsymbol{x}_k\mid\mathcal{Z}_{k-1}, \boldsymbol{z}_k)=\frac{p(\boldsymbol{z}_k,\boldsymbol{x}_k\mid \mathcal{Z}_{k-1})}{p(\boldsymbol{z}_k\mid \mathcal{Z}_{k-1})}
=\frac{p(\boldsymbol{z}_k\mid\boldsymbol{x}_k,\mathcal{Z}_{k-1})\;p(\boldsymbol{x}_k\mid \mathcal{Z}_{k-1})}{p(\boldsymbol{z}_k\mid \mathcal{Z}_{k-1})}
=\frac{p(\boldsymbol{z}_k\mid\boldsymbol{x}_k)\;p(\boldsymbol{x}_k\mid \mathcal{Z}_{k-1})}{p(\boldsymbol{z}_k\mid \mathcal{Z}_{k-1})}
$$

最后一步是因为，在标准状态空间中，假设给定当前状态 $\boldsymbol{x}_k$ ，当前观测 $\boldsymbol{z}_k$ 与过去的观测 $\mathcal{Z}_{k-1}$ 独立，因此 $p(\boldsymbol{z}_k\mid\boldsymbol{x}_k,\mathcal{Z}_{k-1})=p(\boldsymbol{z}_k\mid\boldsymbol{x}_k)$ ，也就是似然概率。上式也就可以表示为“后验 = 先验 ✖ 似然”。

由于过程噪声、观测噪声均为高斯噪声，且先验分布也为高斯分布 $\mathcal{N}(\hat{\boldsymbol{x}}_{k|k-1},\boldsymbol{P}_{k|k-1})$ ，我们求解的后验估计 $p(\boldsymbol{x}_k\mid\mathcal{Z}_k)$ 具有以下性质：

$$
\begin{align*}
p(\boldsymbol{x}_k\mid\mathcal{Z}_k) &\propto p(\boldsymbol{z}_k\mid\boldsymbol{x}_k)\;p(\boldsymbol{x}_k\mid \mathcal{Z}_{k-1})\\
&\propto \exp\left(-\frac{1}{2}(\boldsymbol{z}_k-h(\boldsymbol{x}_k))^T\boldsymbol{R}_k^{-1}(\boldsymbol{z}_k-h(\boldsymbol{x}_k))-\frac{1}{2}(\boldsymbol{x}_k-\hat{\boldsymbol{x}}_{k|k-1})^T\boldsymbol{P}_{k|k-1}^{-1}(\boldsymbol{x}_k-\hat{\boldsymbol{x}}_{k|k-1}) \right)
\end{align*}
$$

上式中第一项为**似然**项，是在已知真实状态 $\boldsymbol{x}_k$ 时观测到 $\boldsymbol{z}_k$ 的概率，由测量噪声模型给出；第二项为**先验**项，是先验分布 $\boldsymbol{x}_k\mid\mathcal{Z}_{k-1}\sim\mathcal{N}(\hat{\boldsymbol{x}}_{k|k-1},\boldsymbol{P}_{k|k-1})$ 的概率密度函数。

卡尔曼滤波旨在最大化这个**后验** $\hat{\boldsymbol{x}}=\arg\max_{\boldsymbol{x}_k} p(\boldsymbol{x}_k\mid \mathcal{Z}_k)$ ，则该最大后验估计问题等价于最小化损失函数 $L(\boldsymbol{x}_k)$ ：

$$
\hat{\boldsymbol{x}}_{k|k}^{\text{MAP}}=\arg\min_{\boldsymbol{x}_k}L(\boldsymbol{x}_k),\quad L(\boldsymbol{x}_k)=\frac{1}{2}\|\boldsymbol{x}_k-\hat{\boldsymbol{x}}_{k|k-1}\|^2_{\boldsymbol{P}_{k|k-1}^{-1}}+\frac{1}{2}\|\boldsymbol{z}_k-h(\boldsymbol{x}_k)\|^2_{\boldsymbol{R}_k^{-1}}
$$

可以写成加权最小二乘的形式：

$$
L(\boldsymbol{x})=\min_{\boldsymbol{x}} \frac{1}{2}[\boldsymbol{r}(\boldsymbol{x})]^T\boldsymbol{W}\boldsymbol{r}(\boldsymbol{x}),\quad \boldsymbol{r}(\boldsymbol{x})=\begin{pmatrix}
\boldsymbol{x}-\hat{\boldsymbol{x}}_{k|k-1} \\
\boldsymbol{z}_k-h(\boldsymbol{x})
\end{pmatrix},\quad
\boldsymbol{W}=\begin{pmatrix}
\boldsymbol{P}_{k|k-1}^{-1} & \boldsymbol{0} \\
\boldsymbol{0} & \boldsymbol{R}_k^{-1}
\end{pmatrix}
$$

若状态量维度 $\boldsymbol{x}:n_x\times 1$ ，观测量维度 $\boldsymbol{z}_k:n_z\times 1$ ，则 $\boldsymbol{r}(\boldsymbol{x}):(n_x+n_z)\times 1$ 为残差向量， $\boldsymbol{W}:(n_x+n_z)\times(n_x+n_z)$ 为权重/信息矩阵，用于给各个残差分量分配权重。通常取 $\boldsymbol{W}=\boldsymbol{R}^{-1}$ ，因为噪声越小、信息越可靠、权重越大。

这个加权最小二乘问题中，由于观测模型 $\boldsymbol{z}_k=\boldsymbol{h}(\boldsymbol{x}_k)$ 是非线性的，没有解析解。EKF 在局部线性化，将非线性最小二乘转换为线性最小二乘。
