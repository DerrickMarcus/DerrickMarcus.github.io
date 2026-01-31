# 2 高斯过程

!!! abstract
    高斯过程是随机过程中较为简单的一种，只需要均值和协方差就可以完全确定，性质简单、易于研究，且实际生活中很多问题都可以使用高斯过程建模，不仅可以大大简化分析问题，也具有较好的适用性。

    高斯过程的特征函数中，自变量没有选取 $\boldsymbol{t}$ ，是因为随机过程的参数已经占用了 $t$ ，为避免混淆，自变量选取为 $\boldsymbol{\omega}$ ，但注意不是频域的常见频率表示 $\omega$ ，这里的 $\boldsymbol{\omega}$ 反而具有时域的意义。

高斯过程的定义：随机过程 $\{X(t)\}$ 满足 $\forall n,\;\forall t_1,\cdots,t_n\in T$ ，随机向量 $\left(X(t_1),X(t_2)\cdots,X(t_n)\right)^T$ 均服从 $n$ 元高斯分布，则 $\{X(t)\}$ 为高斯过程。

上面的定义也是高斯过程最显著的性质，因此对高斯过程性质的研究，首先要从**多元高斯分布**开始。

## 多元高斯分布

服从 $n$ 元高斯分布的随机向量 $\boldsymbol{X}=\left(X_1,X_2\cdots,X_n\right)^T\sim\mathcal{N}(\boldsymbol{\mu},\boldsymbol{\Sigma})$ ，概率密度函数 PDF 为：

$$
\begin{align*}
f_{\boldsymbol{X}}(\boldsymbol{x})&=\frac{1}{(2\pi)^{n/2}\sqrt{|\boldsymbol{\Sigma}|}}\exp\left(-\frac{1}{2}(\boldsymbol{x}-\boldsymbol{\mu})^T\boldsymbol{\Sigma}^{-1}(\boldsymbol{x}-\boldsymbol{\mu})\right) \\
\boldsymbol{x}&=(x_1,\cdots,x_n)^T,\quad \boldsymbol{\mu}\in\mathbb{R}^n,\quad \boldsymbol{\Sigma}\in\mathbb{R}^{n\times n}
\end{align*}
$$

均值和协方差矩阵：

$$
\begin{gather*}
\boldsymbol{\mu}=\mathrm{E}(\boldsymbol{X})=\begin{pmatrix}
\mathrm{E}(X_1) \\ \vdots \\ \mathrm{E}(X_n)
\end{pmatrix},\quad
\boldsymbol{\Sigma}=\mathrm{Cov(\boldsymbol{X})}=\mathrm{E}\left( (\boldsymbol{X}-\boldsymbol{\mu})(\boldsymbol{X}-\boldsymbol{\mu})^T \right) \\
\mu_i=\mathrm{E}(X_i),\quad \sigma_{ij}=\mathrm{E}\left((X_i-\mu_i)(X_j-\mu_j)\right)
\end{gather*}
$$

高斯向量 $\boldsymbol{X}=\left(X_1,X_2\cdots,X_n\right)^T$ 的特征函数（概率密度函数的傅里叶反变换）：

$$
\phi:\mathbb{R}^n\to \mathbb{C},\quad
\phi_{\boldsymbol{X}}(\boldsymbol{\omega})=\mathrm{E}(\mathrm{e}^{\mathrm{j}\boldsymbol{\omega}^T\boldsymbol{X}})=\exp\left(\mathrm{j}\boldsymbol{\omega}^T\boldsymbol{\mu}-\frac{1}{2}\boldsymbol{\omega}^T\boldsymbol{\Sigma}\boldsymbol{\omega}\right)
$$

其中 $\boldsymbol{X}=\left(X_1,X_2\cdots,X_n\right)^T$ 各分量相互独立的**充要条件**为协方差矩阵 $\boldsymbol{\Sigma}$ 是对角阵（非对角线位置为 0），此时各分量解耦，特征函数可写为求积形式：

$$
\phi_{\boldsymbol{X}}(\boldsymbol{\omega})=\prod_{i=1}^n\phi_{X_i}(\omega_i)=\prod_{i=1}^n\exp\left(\mathrm{j}\mu_i\omega_i-\frac{1}{2}\sigma_i^2\omega_i^2\right)
$$

经过线性变换，特征函数变为：

$$
\phi_{\boldsymbol{A}\boldsymbol{X}+\boldsymbol{b}}(\boldsymbol{\omega})=\mathrm{e}^{\mathrm{j}\boldsymbol{\omega}^T\boldsymbol{b}}\phi_{\boldsymbol{X}}(\boldsymbol{A}^T\boldsymbol{\omega})
$$

多元高斯分布的高阶矩由均值向量和协方差矩阵决定。

事实上，如果随机向量 $\boldsymbol{X}\in\mathbb{R}^n$ 的特征函数满足 $\phi_{\boldsymbol{X}}(\boldsymbol{\omega})=\exp\left(\mathrm{j}\boldsymbol{\omega}^T\boldsymbol{\mu}-\frac{1}{2}\boldsymbol{\omega}^T\boldsymbol{\Sigma}\boldsymbol{\omega}\right)$ 的形式，其中 $\boldsymbol{\mu}\in\mathbb{R}^n$ 且 $\boldsymbol{\Sigma}$ 非负定，那么 $\boldsymbol{X}$ **一定是** $n$ **元高斯向量**。

如果存在**高阶混合矩** $\mathrm{E}\left(X_1^{k_1}X_2^{k_2}\cdots X_n^{k_n}\right)<+\infty$ ，则：

$$
\mathrm{E}\left(X_1^{k_1}X_2^{k_2}\cdots X_n^{k_n}\right)=(-\mathrm{j})^{k_1+\cdots+k_n}\frac{\partial^{k_1+\cdots+k_n} \phi_{\boldsymbol{X}}(\omega)}{\partial \omega_1^{k_1}\cdots \partial \omega_n^{k_n}}\bigg|_{\boldsymbol{\omega}=\boldsymbol{0}}
$$

特别地，对于**零均值4维高斯分布** $\boldsymbol{X}=(X_1,X_2,X_3,X_4)^T$ ，四阶混合原点矩为：

$$
\mathrm{E}(X_1X_2X_3X_4)=\mathrm{E}(X_1X_2)\cdot\mathrm{E}(X_3X_4)+\mathrm{E}(X_1X_3)\cdot\mathrm{E}(X_2X_4)+\mathrm{E}(X_1X_4)\cdot\mathrm{E}(X_2X_3)
$$

与三角函数有关的特征函数计算技巧：

$$
\begin{align*}
\mathrm{E}(\cos X)&=\frac{1}{2}\mathrm{E}\left(\mathrm{e}^{\mathrm{j}X}+\mathrm{e}^{-\mathrm{j}X}\right)=\frac{\phi_X(1)+\phi_X(-1)}{2} \\
\mathrm{E}(\sin X)&=\frac{1}{2\mathrm{j}}\mathrm{E}\left(\mathrm{e}^{\mathrm{j}X}-\mathrm{e}^{-\mathrm{j}X}\right)=\frac{\phi_X(1)-\phi_X(-1)}{2\mathrm{j}}
\end{align*}
$$

## 线性变换

对于 $n$ 元高斯随机变量 $\boldsymbol{X} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ ，其线性变换 $\boldsymbol{Y} = \boldsymbol{A}\boldsymbol{X} + \boldsymbol{b}\sim\mathcal{N}\left(\boldsymbol{A}\boldsymbol{\mu} + \boldsymbol{b},\boldsymbol{A}\boldsymbol{\Sigma} \boldsymbol{A}^T\right)$ 仍为多元高斯分布，且均值为 $\boldsymbol{A}\boldsymbol{\mu} + \boldsymbol{b}$ ，协方差矩阵为 $\boldsymbol{A}\boldsymbol{\Sigma} \boldsymbol{A}^T$ .

$$
\boldsymbol{\mu}_Y=\boldsymbol{A}\boldsymbol{\mu}_X + \boldsymbol{b},\quad
\boldsymbol{\Sigma}_Y=\boldsymbol{A}\boldsymbol{\Sigma}_X \boldsymbol{A}^T
$$

## 边缘分布

联合高斯分布 $\boldsymbol{}{X}=(X_1, X_2, \dots, X_n)^T$ ，其边缘分布仍为高斯分布，且均值和协方差矩阵对应原分布均值和协方差矩阵中相应位置上的值。例如：

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

则：

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

## 独立性

重要结论：**对于高斯变量，“不相关”等价于“独立”**。

设 $\boldsymbol{X}=(\boldsymbol{X}_1,\boldsymbol{X}_2)^T$ 服从 $n$ 元高斯分布，均值为 $\boldsymbol{\mu}=(\boldsymbol{\mu}_1,\boldsymbol{\mu}_2)^T$ ，协方差矩阵为：

$$
\boldsymbol{\Sigma}=\begin{pmatrix}
\boldsymbol{\Sigma}_1 & \boldsymbol{\Sigma}_{12} \\
\boldsymbol{\Sigma}_{21} & \boldsymbol{\Sigma}_2
\end{pmatrix}
$$

则 $\boldsymbol{X}$ 的两个子向量 $\boldsymbol{X}_1,\boldsymbol{X}_2$ 相互独立的充要条件为 $\boldsymbol{\Sigma}_{12}=\boldsymbol{0}$ 零矩阵，此时 $\boldsymbol{\Sigma}$ 为分块对角阵。

推论： $n$ 元高斯向量 $\boldsymbol{X}=(X_1,\cdots,X_n)^T$ 各个分量相互独立的充要条件为各分量之间协方差为 $\sigma_{ij}=0$ ，即 $\boldsymbol{\Sigma}$ 为对角阵。

## 条件分布

结论：多元高斯的任意两个分量之间的条件分布也是高斯分布。

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

则条件分布仍为联合高斯分布 $\boldsymbol{X}_1|\boldsymbol{X}_2\sim\mathcal{N}(\boldsymbol{\mu}_{1|2},\boldsymbol{\Sigma}_{1|2})$ ，其条件均值和条件协方差分别为：

$$
\boldsymbol{\mu}_{1|2}=\boldsymbol{\mu}_1+\boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_2^{-1}(\boldsymbol{X}_2-\boldsymbol{\mu}_2),\quad \boldsymbol{\Sigma}_{1|2}=\boldsymbol{\Sigma}_1-\boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_2^{-1}\boldsymbol{\Sigma}_{21}
$$

注意到条件均值 $\boldsymbol{\mu}_{1|2}$ 与条件取值 $\boldsymbol{X}_2$ 呈线性关系，而条件协方差矩阵 $\boldsymbol{\Sigma}_{1|2}$ **与条件取值** $\boldsymbol{X}_2$ **无关**。对于联合高斯分布来说，一部分取值的固定不会影响其余部分的不确定性，但这在其他分布中不一定成立。

<br>

退化到**二维情况**，得到：

$$
\begin{align*}
\mathrm{E}(X|Y) &= \mu_X + \frac{\mathrm{Cov}(X, Y)}{\mathrm{Var}(Y)}(Y - \mu_Y) = \mu_X + \rho_{XY}\sqrt{\frac{\mathrm{Var}(X)}{\mathrm{Var}(Y)}}(Y - \mu_Y) \\
\mathrm{Var}(X|Y) &= \mathrm{Var}(X)(1 - \rho_{XY}^2) = \mathrm{Var}(X) - \frac{\mathrm{Cov}^2(X, Y)}{\mathrm{Var}(Y)}
\end{align*}
$$

其中 $\rho_{XY}$ 为 $X,Y$ 的相关系数， $\mathrm{Cov}(X, Y)$ 为 $X,Y$ 的协方差， $\rho_{XY}=\dfrac{\mathrm{Cov}(X, Y)}{\sqrt{\mathrm{Var}(X)\mathrm{Var}(Y)}}$ .

---

补充内容：Price 定理

二元高斯分布 $(X_1,X_2)\sim\mathcal{N}(0,0,\sigma_1^2,\sigma_2^2,\rho)$ ，且 $g(x,y)$ 为一非线性函数，则：

$$
\frac{\partial\mathrm{E}(g(X_1,X_2))}{\partial \rho}=\sigma_1 \sigma_2 \mathrm{E}\left( \frac{\partial^2g(X_1,X_2)}{\partial X_1 \partial X_2} \right)
$$
