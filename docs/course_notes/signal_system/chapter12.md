# 12 系统的状态变量分析

状态/状态变量/状态矢量：一个动态系统的状态是表示系统的一组最少变量，只需知道 $t=t_0$ 时刻这组变量和 $t\geqslant t_0$ 时刻以后的输入， 就能确定系统在 $t\geqslant t_0$​ 时刻以后的行为。

引入状态变量的目的：使用中间变量表示每相邻阶输入输出的关系，可以把一元 $N$ 阶方程转换为 $N$ 元一阶方程，每一阶都用状态变量表示，相邻阶的中间变量之间是一阶关系。

算子 $p$ 是微分运算，算子 $\dfrac{1}{p}$​ 是积分运算。算子表达式就是关于积分和微分环节的组合。

## 12.1 连续时间系统状态方程的建立

状态方程与输出方程分别为：

$$
\begin{align*}
\dot{\mathbf{\lambda}}(t)&=\mathbf{A\lambda}(t)+\mathbf{Be}(t) \\
\mathbf{r}(t)&=\mathbf{C\lambda}(t)+\mathbf{De}(t)
\end{align*}
$$

对于 LTI 系统， $\mathbf{A},\mathbf{B},\mathbf{C},\mathbf{D}$ 矩阵是常数；对于时变系统 $\mathbf{A},\mathbf{B},\mathbf{C},\mathbf{D}$ 矩阵是时间的函数。

<br>

状态方程的建立方法：

（1）由电路图建立（电路课的重点，非本课重点）。

（2）由系统输入输出方程或信号流图建立状态方程。对于与给定的系统，流图的形式可以不同，状态变量的选择不唯一， $\mathbf{A},\mathbf{B},\mathbf{C},\mathbf{D}$ 矩阵也可能不唯一。

![2024春信号与系统26第二十四讲12.1-12.3_15](https://cdn.jsdelivr.net/gh/DerrickMarcus/picgo_image/images/ch12_img1.png)

由算子表达式分解或系统函数建立状态方程。将表达式部分分式展开为 $H(p)=\displaystyle\sum\dfrac{\beta_i}{p+\alpha_i}$​ 的基本单元，由这些基本单元串联、并联、级联组装而成。

基本单元：

![2024春信号与系统26第二十四讲12.1-12.3_20](https://cdn.jsdelivr.net/gh/DerrickMarcus/picgo_image/images/ch12_img2.png)

## 12.2 连续时间系统状态方程的求解

### Laplace 变换解法

该方法是比较容易的方法。

对给定状态方程：

$$
\begin{align*}
\dot{\mathbf{\lambda}}(t) &= \mathbf{A}\mathbf{\lambda}(t) + \mathbf{B}\mathbf{e}(t)\\
\mathbf{r}(t) &= \mathbf{C}\mathbf{\lambda}(t) + \mathbf{D}\mathbf{e}(t)
\end{align*}
$$

两侧做拉氏变换，将微分变成乘法：

$$
\begin{align*}
s\mathbf{\Lambda}(s) - \mathbf{\lambda}(0_-) &= \mathbf{A}\mathbf{\Lambda}(s) + \mathbf{B}\mathbf{E}(s)\\
\mathbf{R}(s) &= \mathbf{C}\mathbf{\Lambda}(s) + \mathbf{D}\mathbf{E}(s)
\end{align*}
$$

由 $s$ 域状态方程解出状态变量：

$$
\mathbf{\Lambda}(s) = (s\mathbf{I} - \mathbf{A})^{-1}\mathbf{\lambda}(0_-) + (s\mathbf{I} - \mathbf{A})^{-1}\mathbf{B}\mathbf{E}(s)
$$

代入输出方程得到结果：

$$
\mathbf{R}(s) = \mathbf{C}(s\mathbf{I} - \mathbf{A})^{-1}\mathbf{\lambda}(0_-) + [\mathbf{C}(s\mathbf{I} - \mathbf{A})^{-1}\mathbf{B} + \mathbf{D}]\mathbf{E}(s)
$$

定义特征矩阵 $\mathbf{\Gamma}(s) = (s\mathbf{I} - \mathbf{A})^{-1}$ ，重写状态方程和输出方程：

$$
\begin{align*}
\mathbf{\Lambda}(s) &= \mathbf{\Gamma}(s)\mathbf{\lambda}(0_-) + \mathbf{\Gamma}(s)\mathbf{B}\mathbf{E}(s)
\\
\mathbf{R}(s) &= \mathbf{C}\mathbf{\Gamma}(s)\mathbf{\lambda}(0_-) + [\mathbf{C}\mathbf{\Gamma}(s)\mathbf{B} + \mathbf{D}]\mathbf{E}(s)
\end{align*}
$$

逆变换得到：

$$
\begin{align*}
\mathbf{\lambda}(t) &= \underbrace{\gamma(t)\mathbf{\lambda}(0_-)}_{\text{零输入解}} + \underbrace{\gamma(t)\mathbf{B} * \mathbf{e}(t)}_{\text{零状态解}} \\
\mathbf{r}(t) &= \underbrace{\mathbf{C}\gamma(t)\mathbf{\lambda}(0_-)}_{\text{零输入响应}} + \underbrace{[\mathbf{C}\gamma(t)\mathbf{B} + \mathbf{D}\delta(t)] * \mathbf{e}(t)}_{\text{零状态响应}}
\end{align*}
$$

其中 $\gamma(t) = \mathcal{L}^{-1}\{\mathbf{\Gamma}(s)\} = \mathcal{L}^{-1}\{(s\mathbf{I} - \mathbf{A})^{-1}\}$

### 时域解法

推导过程比较繁琐，因此略去。

对比时域解法和拉氏变换域解法的结果：

$$
\begin{align*}
\mathbf{\lambda}(t) &= e^{\mathbf{A}t}\mathbf{\lambda}(0_-) + e^{\mathbf{A}t}\mathbf{B} * \mathbf{e}(t) \quad \text{时域解法}\\
&= \underbrace{\gamma(t)\mathbf{\lambda}(0_-)}_{\text{零输入解}} + \underbrace{\gamma(t)\mathbf{B} * \mathbf{e}(t)}_{\text{零状态解}} \quad \text{变换域解法}\\
\mathbf{r}(t) &= \mathbf{C}e^{\mathbf{A}t}\mathbf{\lambda}(0_-) + [\mathbf{C}e^{\mathbf{A}t}\mathbf{B} + \mathbf{D}\delta(t)] * \mathbf{e}(t) \quad \text{时域解法}\\
&= \underbrace{\mathbf{C}\gamma(t)\mathbf{\lambda}(0_-)}_{\text{零输入响应}} + \underbrace{[\mathbf{C}\gamma(t)\mathbf{B} + \mathbf{D}\delta(t)] * \mathbf{e}(t)}_{\text{零状态响应}} \quad \text{变换域解法}
\end{align*}
$$

可发现特征矩阵和状态转移矩阵是一对拉氏变换对:

$$
\mathcal{L}\{e^{\mathbf{A}t}\} = \mathcal{L}\{\gamma(t)\} = \mathbf{\Gamma}(s) = (s\mathbf{I} - \mathbf{A})^{-1}
$$

时域法关键是求 $\mathrm{e}^{\mathbf{A}t}$ ，拉氏变换法关键是求 $(s\mathbf{I} - \mathbf{A})^{-1}$ . 两者本质相同，但前者要用凯莱-哈密顿定理，比较繁琐，后者相对容易。

## 12.3 根据状态方程求转移函数

考察完全解的变换式：

$$
\mathbf{R}(s) = \mathbf{C}(s\mathbf{I} - \mathbf{A})^{-1}\boldsymbol{\lambda}(0_-) + [\mathbf{C}(s\mathbf{I} - \mathbf{A})^{-1}\mathbf{B} + \mathbf{D}]\mathbf{E}(s)
$$

零状态时有：

$$
\mathbf{R}(s) = [\mathbf{C}(s\mathbf{I} - \mathbf{A})^{-1}\mathbf{B} + \mathbf{D}]\mathbf{E}(s) = \mathbf{H}(s)\mathbf{E}(s)
$$

即：

$$
\mathbf{H}(s) = \mathbf{C}(s\mathbf{I} - \mathbf{A})^{-1}\mathbf{B} + \mathbf{D}
$$

假设单输入单输出，有：

$$
\begin{align*}
\mathbf{H}(s) &= \mathbf{C}_{1\times k}(s\mathbf{I}_{k\times k} - \mathbf{A}_{k\times k})^{-1}\mathbf{B}_{k\times 1} + \mathbf{D}\\
&= \frac{\mathbf{C}_{1\times k}\text{adj}(s\mathbf{I} - \mathbf{A})\mathbf{B}_{k\times 1} + \mathbf{D}}{|s\mathbf{I} - \mathbf{A}|}
\end{align*}
$$

$|s\mathbf{I} - \mathbf{A}| = 0$ 之根就是 $\mathbf{H}(s)$ 的极点，即 $\mathbf{A}$ 的特征值。

## 12.4 离散时间系统状态方程的建立

同连续时间系统的形式，用差分代替微分。

$$
\begin{align*}
\mathbf{\lambda}(n+1)&=\mathbf{A\lambda}(n)+\mathbf{Bx}(n) \\
\mathbf{y}(n)&=\mathbf{C\lambda}(n)+\mathbf{Dx}(n)
\end{align*}
$$

建立方法：

- 由定义建立。
- 由系统框图或信号流图建立。

## 12.5 离散时间系统状态方程的求解

### 时域迭代法

如果 $n_0 = 0$ ，则有：

$$
\begin{align*}
\lambda(n) &= \mathbf{A}^n \lambda(0) u(n) \left[ \sum_{i=0}^{n-1} \mathbf{A}^{n-1-i} \mathbf{B} \mathbf{x}(i) \right] u(n - 1) \\
&= \mathbf{A}^n \lambda(0) u(n) \mathbf{A}^{n-1} \mathbf{B} u(n - 1) \mathbf{x}(n)
\end{align*}
$$

代入输出方程：

$$
\begin{align*}
\mathbf{y}(n) &= \mathbf{C}\lambda(n) + \mathbf{D}\mathbf{x}(n)\\
\mathbf{y}(n) &= \underbrace{\mathbf{C}\mathbf{A}^n \lambda(0) u(n)}_{\text{零输入响应}} + \underbrace{\left[ \mathbf{C}\mathbf{A}^{n-1} \mathbf{B} u(n-1) + \mathbf{D}\delta(n) \right] \mathbf{x}(n)}_{\text{零状态响应}}
\end{align*}
$$

考察零状态响应，得到：

$$
\mathbf{h}(n) = \mathbf{C}\mathbf{A}^{n-1} \mathbf{B} u(n - 1) + \mathbf{D}\delta(n)
$$

定义离散状态转移矩阵 $\mathbf{A}^n$ ，类似于连续系统的 $\mathrm{e}^{\mathbf{A}t}$ .

### z 变换法

已知状态方程和输出方程：

$$
\begin{cases}
\mathbf{\lambda}(n+1) = \mathbf{A}\mathbf{\lambda}(n) + \mathbf{B}\mathbf{x}(n) \\
\mathbf{y}(n) = \mathbf{C}\mathbf{\lambda}(n) + \mathbf{D}\mathbf{x}(n)
\end{cases}
$$

两侧同取 $z$ 变换得到：

$$
\begin{cases}
z\mathbf{\lambda}(z) - z\mathbf{\lambda}(0) = \mathbf{A}\mathbf{\Lambda}(z) + \mathbf{B}\mathbf{X}(z) \\
\mathbf{Y}(z) = \mathbf{C}\mathbf{\Lambda}(z) + \mathbf{D}\mathbf{X}(z)
\end{cases}
$$

由 $z$ 域状态方程解出状态变量：

$$
\mathbf{\Lambda}(z) = (z\mathbf{I} - \mathbf{A})^{-1}z\mathbf{\lambda}(0) + (z\mathbf{I} - \mathbf{A})^{-1}\mathbf{B}\mathbf{X}(z)
$$

两侧同取逆 $z$ 变换解出：

$$
\mathbf{\lambda}(n) = \mathcal{Z}^{-1}\left[(z\mathbf{I} - \mathbf{A})^{-1}z\right]\mathbf{\lambda}(0) + \mathcal{Z}^{-1}\left[(z\mathbf{I} - \mathbf{A})^{-1}\right]\mathbf{B} * \mathbf{x}(n)
$$

注意和连续情况拉氏变换解在形式上的区别。

对照 $z$ 变换解和时域解：

$$
\begin{align*}
\lambda(n) &= \mathcal{Z}^{-1}\left[(z\mathbf{I} - \mathbf{A})^{-1}z\right]\lambda(0) + \mathcal{Z}^{-1}\left[(z\mathbf{I} - \mathbf{A})^{-1}\right]\mathbf{B}* \mathbf{x}(n) \\
&= \underbrace{\mathbf{A}^n\lambda(0)u(n)}_{\text{零输入解}} + \underbrace{\mathbf{A}^{n-1}\mathbf{B}u(n - 1) * \mathbf{x}(n)}_{\text{零状态解}}\\
\mathbf{y}(n) &= \mathbf{C}\mathcal{Z}^{-1}\left[(z\mathbf{I} - \mathbf{A})^{-1}z\right]\lambda(0) + \left\{\mathbf{C}\mathcal{Z}^{-1}\left[(z\mathbf{I} - \mathbf{A})^{-1}\right]\mathbf{B} + \mathbf{D}\delta(n)\right\} _\mathbf{x}(n) \\
&= \underbrace{\mathbf{C}\mathbf{A}^n\lambda(0)u(n)}_{\text{零输入响应}} + \underbrace{\left[\mathbf{C}\mathbf{A}^{n-1}\mathbf{B}u(n - 1) + \mathbf{D}\delta(n)\right] * \mathbf{x}(n)}_{\text{零状态响应}}
\end{align*}
$$

得到对应关系：

$$
\begin{align*}
\mathbf{A}^nu(n) &= \mathcal{Z}^{-1}\left[(z\mathbf{I} - \mathbf{A})^{-1}z\right] \\
\mathbf{A}^{n-1}u(n - 1) &= \mathcal{Z}^{-1}\left[(z\mathbf{I} - \mathbf{A})^{-1}\right]
\end{align*}
$$

对比连续时间系统的 $\mathcal{L}^{-1}\left\{(s\mathbf{I} - \mathbf{A})^{-1}\right\} = \mathrm{e}^{\mathbf{A}t}$ ，用类似方法可求得 $\mathbf{H}(z) = \mathbf{C}(z\mathbf{I} - \mathbf{A})^{-1}\mathbf{B} + \mathbf{D}$ .

## 12.6 状态矢量的线性变换

选择不同的状态矢量可以得到不同的 $\mathbf{A},\mathbf{B},\mathbf{C},\mathbf{D}$ 矩阵，各状态矢量之间存在某种约束，矩阵 $\mathbf{A},\mathbf{B},\mathbf{C},\mathbf{D}$ 之间存在某种变换关系。

连续时间系统的 BIBO 稳定性：

- 系统转移函数为 $\mathbf{H}(s)=\mathbf{C}(s\mathbf{I}-\mathbf{A})^{-1}\mathbf{B}+\mathbf{D}$ .
- $|s\mathbf{I}-\mathbf{A}|=0$ 的根， $\mathbf{A}$ 的特征根，就是 $\mathbf{H}(s)$ 的极点。
- 若上述的根/极点在 $s$ 平面左半平面，则系统稳定。

离散时间系统的 BIBO 稳定性：

- 系统转移函数为 $\mathbf{H}(z)=\mathbf{C}(z\mathbf{I}-\mathbf{A})^{-1}\mathbf{B}+\mathbf{D}$ .
- $|z\mathbf{I}-\mathbf{A}|=0$ 的根， $\mathbf{A}$ 的特征根，就是 $\mathbf{H}(z)$ 的极点。
- 若上述的根/极点在 $z$ 平面左半平面，则系统稳定。

## 12.7 系统的可控性和可观性

可控性（Controllability）：给定起始状态，可以找到容许的输入量 (控制矢量)，在有限时间内把系统的所有状态引向零状态。如果可做到这点，则称系统完全可控。

可观性（Observability）：给定输入 (控制) 后，能在有限时间内根据系统输出唯一地确定系统的起始状态。如果可做到这点，则称系统完全可观。

利用可控阵和可观阵判定：

若

$$
\mathbf{M} = \left[ \mathbf{B} \quad \mathbf{A}\mathbf{B} \quad \cdots \quad \mathbf{A}^{k-1}\mathbf{B} \right]
$$

为**行满秩**，即 $\operatorname{rank}(\mathbf{M}) = k$ ，则系统完全可控。

若

$$
\mathbf{N} = \begin{bmatrix} \mathbf{C} \\ \mathbf{C}\mathbf{A} \\ \vdots \\ \mathbf{C}\mathbf{A}^{k-1} \end{bmatrix}
$$

为**列满秩**，即 $\operatorname{rank}(\mathbf{N}) = k$ ，则系统完全可观。

可控性只和 $\mathbf{A},\mathbf{B}$ 有关，可观性只和 $\mathbf{A},\mathbf{C}$ 有关。

<br>

零极点相消现象：

$H(s)$ 原有 $k$ 个极点，有项消失（即零极点对消）后则极点不到 $k$ 个（降阶）；
零极点相消部分是不可控、不可观的部分，因而转移函数的描述方式只反映了系统中可观可控的部分，不能反映不可观、不可控的部分。所以用转移函数描述系统是不全面的，而用**状态方程和输出方程描述系统更全面、详尽**。
