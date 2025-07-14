# Chapter 5 Gaussian Process

> 高斯过程是随机过程中较为简单的一种，只需要均值和协方差就可以完全确定，性质简单、易于研究，且实际生活中很多问题都可以使用高斯过程建模，不仅可以大大简化分析问题，也具有较好的适用性。
>
> 高斯过程的特征函数中，自变量没有选取 $\boldsymbol{t}$ ，是因为随机过程的参数已经占用了 $t$ ，为避免混淆，自变量选取为 $\boldsymbol{\omega}$ ，但注意不是频域的常见频率表示 $\omega$ ，这里的 $\boldsymbol{\omega}$ 反而具有时域的意义。

高斯过程的定义：随机过程 $\{X(t)\}$ 满足 $\forall n,\;\forall t_1,\cdots,t_n\in T$ ，随机向量 $\left(X(t_1),X(t_2)\cdots,X(t_n)\right)^T$ 均服从 $n$ 元高斯分布，则 $\{X(t)\}$ 为 高斯过程。

上面的定义也是高斯过程最显著的性质，因此对高斯过程性质的研究，首先要从 *多元高斯分布* 开始。

## 多元高斯分布

服从 $n$ 元高斯分布的随机向量 $\boldsymbol{X}=\left(X_1,X_2\cdots,X_n\right)^T$ 的概率密度函数 PDF 为：

$$
f_{\boldsymbol{X}}(\boldsymbol{x})=\frac{1}{(2\pi)^{n/2}\sqrt{|\boldsymbol{\Sigma}|}}\exp\left(-\frac{1}{2}(\boldsymbol{x}-\boldsymbol{\mu})^T\boldsymbol{\Sigma}^{-1}(\boldsymbol{x}-\boldsymbol{\mu})\right)
$$

均值和协方差矩阵：

$$
\begin{gather*}
\boldsymbol{\mu}=\mathrm{E}[\boldsymbol{X}]=\begin{pmatrix}
\mathrm{E}[X_1] \\ \vdots \\ \mathrm{E}[X_n]
\end{pmatrix},\quad
\boldsymbol{\Sigma}=\mathrm{Cov(\boldsymbol{X})}=\mathrm{E}\left[ (\boldsymbol{X}-\boldsymbol{\mu})(\boldsymbol{X}-\boldsymbol{\mu})^T \right] \\
\mu_i=\mathrm{E}[X_i],\quad \sigma_{ij}=\mathrm{E}[(X_i-\mu_i)(X_j-\mu_j)]
\end{gather*}
$$

高斯向量 $\boldsymbol{X}=\left(X_1,X_2\cdots,X_n\right)^T$ 的特征函数（概率密度函数的傅里叶反变换）：

$$
\phi:\mathbb{R}^n\to \mathbb{C},\quad
\phi_{\boldsymbol{X}}(\boldsymbol{\omega})=\mathrm{E}[\mathrm{e}^{\mathrm{j}\boldsymbol{\omega}^T\boldsymbol{X}}]=\exp\left(\mathrm{j}\boldsymbol{\omega}^T\boldsymbol{\mu}-\frac{1}{2}\boldsymbol{\omega}^T\boldsymbol{\Sigma}\boldsymbol{\omega}\right)
$$

其中 $\boldsymbol{X}=\left(X_1,X_2\cdots,X_n\right)^T$ 各分量相互独立的 充要条件 为 协方差矩阵 $\boldsymbol{\Sigma}$ 是对角阵（非对角线位置为0），此时各分量解耦，特征函数可写为求积形式：

$$
\phi_{\boldsymbol{X}}(\boldsymbol{\omega})=\prod_{i=1}^n\phi_{X_i}(\omega_i)=\prod_{i=1}^n\exp\left(\mathrm{j}\mu_i\omega_i-\frac{1}{2}\sigma_i^2\omega_i^2\right)
$$

经过线性变换，特征函数变为：

$$
\phi_{\boldsymbol{A}\boldsymbol{X}+\boldsymbol{b}}(\boldsymbol{\omega})=\mathrm{e}^{\mathrm{j}\boldsymbol{\omega}^T\boldsymbol{b}}\phi_{\boldsymbol{X}}(\boldsymbol{A}^T\boldsymbol{\omega})
$$

如果存在高阶矩 $\mathrm{E}[X_1^{k_1}X_2^{k_2}\cdots X_n^{k_n}]<+\infty$ ，则：

$$
\mathrm{E}[X_1^{k_1}X_2^{k_2}\cdots X_n^{k_n}]=(-\mathrm{j})^{k_1+\cdots+k_n}\frac{\partial^{k_1+\cdots+k_n} \phi_{\boldsymbol{X}}(\omega)}{\partial \omega_1^{k_1}\cdots \partial \omega_n^{k_n}}\bigg|_{\boldsymbol{\omega}=\boldsymbol{0}}
$$

特别地，对于 <span style="color:red">零均值4维高斯分布</span> $\boldsymbol{X}=(X_1,X_2,X_3,X_4)^T$ 有：

$$
\mathrm{E}[X_1X_2X_3X_4]=\mathrm{E}[X_1X_2]\cdot\mathrm{E}[X_3X_4]+\mathrm{E}[X_1X_3]\cdot\mathrm{E}[X_2X_4]+\mathrm{E}[X_1X_4]\cdot\mathrm{E}[X_2X_3]
$$

与三角函数有关的特征函数计算技巧：

$$
\begin{gather*}
\mathrm{E}[\cos X]=\frac{1}{2}\mathrm{E}[\mathrm{e}^{\mathrm{j}X}+\mathrm{e}^{-\mathrm{j}X}]=\frac{\phi_X(1)+\phi_X(-1)}{2} \\
\mathrm{E}[\sin X]=\frac{1}{2\mathrm{j}}\mathrm{E}[\mathrm{e}^{\mathrm{j}X}-\mathrm{e}^{-\mathrm{j}X}]=\frac{\phi_X(1)-\phi_X(-1)}{2\mathrm{j}}
\end{gather*}
$$

## 高斯过程的边缘分布

联合高斯分布 $\boldsymbol{}{X}=(X_1, X_2, \dots, X_n)^T$ ，其边缘分布仍为高斯分布，且均值和协方差矩阵对应原分布中的相应位置上的值。例如：

$$
\begin{pmatrix}
X_1 \\
X_2 \\
X_3 \\
X_4
\end{pmatrix}
\sim
\begin{pmatrix}
\mu_1 \\
\mu_2 \\
\mu_3 \\
\mu_4
\end{pmatrix},
\begin{pmatrix}
\sigma_{11} & \sigma_{12} & \sigma_{13} & \sigma_{14} \\
\sigma_{21} & \sigma_{22} & \sigma_{23} & \sigma_{24} \\
\sigma_{31} & \sigma_{32} & \sigma_{33} & \sigma_{34} \\
\sigma_{41} & \sigma_{42} & \sigma_{43} & \sigma_{44}
\end{pmatrix}
$$

则

$$
\begin{pmatrix}
X_1 \\
X_3
\end{pmatrix}
\sim
\begin{pmatrix}
\mu_1 \\
\mu_3
\end{pmatrix},
\begin{pmatrix}
\sigma_{11} & \sigma_{13} \\
\sigma_{31} & \sigma_{33}
\end{pmatrix}
$$

## 高斯过程的线性变换

对于 $n$ 元高斯随机变量 $\boldsymbol{X} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ ，其线性变换 $\boldsymbol{Y} = \boldsymbol{A}\boldsymbol{X} + \boldsymbol{b}$ 仍服从高斯分布，且均值为 $\boldsymbol{A}\boldsymbol{\mu} + \boldsymbol{b}$ ，协方差矩阵为 $\boldsymbol{A}\boldsymbol{\Sigma} \boldsymbol{A}^T$ .

## 高斯过程的条件分布

对于 $n$ 元高斯随机变量 $\boldsymbol{X} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ ，将其分为两个子向量 $\boldsymbol{X}_1,\boldsymbol{X}_2$ ：

$$
\boldsymbol{X}=\begin{pmatrix}
\boldsymbol{X}_1 \\ \boldsymbol{X}_2
\end{pmatrix}, \quad
\boldsymbol{\mu}=\begin{pmatrix}
\boldsymbol{\mu}_1 \\ \boldsymbol{\mu}_2
\end{pmatrix}, \quad
\boldsymbol{\Sigma}=\begin{pmatrix}
\boldsymbol{\Sigma}_1 & \boldsymbol{\Sigma}_{12} \\
\boldsymbol{\Sigma}_{21} & \boldsymbol{\Sigma}_2
\end{pmatrix}
$$

则条件分布仍为联合高斯分布 $\boldsymbol{X}_1|\boldsymbol{X}_1\sim\mathcal{N}(\boldsymbol{\mu}_{1|2},\boldsymbol{\Sigma}_{1|2})$ ，其均值和协方差分别为：

$$
\boldsymbol{\mu}_{1|2}=\boldsymbol{\mu}_1+\boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_2^{-1}(\boldsymbol{X}_2-\boldsymbol{\mu}_2),\quad \boldsymbol{\Sigma}_{1|2}=\boldsymbol{\Sigma}_1-\boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_2^{-1}\boldsymbol{\Sigma}_{21}
$$

注意到 条件均值 $\boldsymbol{\mu}_{1|2}$ 与条件取值 $\boldsymbol{X}_2$ 呈线性关系，而 条件协方差矩阵 $\boldsymbol{\Sigma}_{1|2}$ 与条件取值 $\boldsymbol{X}_2$ 无关。对于联合高斯分布来说，一部分取值的固定不会影响其余部分的不确定性，但这在其他分布中不一定成立。

<br>

退化到二维情况，得到：

$$
\begin{gather*}
\mathrm{E}(X|Y) = \mu_X + \frac{\mathrm{Cov}(X, Y)}{\mathrm{Var}(Y)}(Y - \mu_Y) = \mu_X + \rho_{XY}\sqrt{\frac{\mathrm{Var}(X)}{\mathrm{Var}(Y)}}(Y - \mu_Y) \\
\mathrm{Var}(X|Y) = \mathrm{Var}(X)(1 - \rho_{XY}^2) = \mathrm{Var}(X) - \frac{\mathrm{Cov}^2(X, Y)}{\mathrm{Var}(Y)}
\end{gather*}
$$

其中 $\rho_{XY}$ 为 $X,Y$ 的相关系数， $\mathrm{Cov}(X, Y)$ 为 $X,Y$ 的协方差， $\rho_{XY}=\dfrac{\mathrm{Cov}(X, Y)}{\sqrt{\mathrm{Var}(X)\mathrm{Var}(Y)}}$ .
