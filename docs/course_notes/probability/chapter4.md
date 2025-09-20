# 4 二阶矩过程和谱分析

!!! abstract
    本章讨论的主要内容为随机过程的基本概念和数字特征，二阶矩过程的性质（均方极限，均方连续，均方可导），随机过程的平稳性分析、相关性，宽平稳过程的谱分析等。

写在前面：当学习完信号与系统，并了解了一点随机过程后，我们应当对信号有一定的认识。以下内容摘自[数字通信 第2章 数字通信的数学基础 - 西安电子科技大学](https://web.xidian.edu.cn/yjsun/files/20140410_170612.pdf)：

> 信号就是一个以时间作为自变量的函数，例如 $x(t)$ ，或一个离散时间序列，例如 $\{x(n)\}$ ，其取值一般是实数或复数。如果是 $N$ 维函数，可表示为矢量函数 $[\mathbf{x}_1(t) ,\mathbf{x}_2(t) , \cdots , \mathbf{x}_N(t)]^T$ ，它在任一时刻都是一个 $N$ 维矢量。
>
> 如果 $x(t)$ 或 $\{x(n)\}$ 在任一时刻的值都是确知量，那么这个信号就是确知信号；如果 $x(t)$ 或 $\{x(n)\}$ 在任一时刻的值是随机变量或随机向量，那么这个信号就是随机信号；随机信号是一种随机过程，离散随机过程也称随机序列。
>
> 如果某个函数中的参数都是已知的，例如函数 $a \sin(\omega t + \varphi)$ 中 $a,\omega,\varphi$ 都已知，那么它所表示的信号就是确知信号。从一个随机过程中获取一个现实，例如记录得到的一段信号波形，也是一个确知信号，尽管它因各态历经性而可能隐含了该随机过程的一些统计特性。

## 随机过程的基本概念

> 随机过程是概率论的一个自然延伸。
>
> 概率论研究的是随机现象和随机变量，随机变量是静态的、不随外部条件变化的。而随机过程研究的是随着某些参数（时间，空间，频率等）变化的随机现象和随机变量。随机过程可以看作是随机变量从有限维到无限维的自然延伸，是一组无穷多个、相互有关的随机变量。[^1]

（随机过程的定义） 随机过程是一组依赖于参数 $t$ 的随机变量 $\{X(t),t\in T\}$ . $T$ 称为参数集/指标集，参数 $t$ 称为指标。

根据参数集 $T$ 的性质，可以分为离散时间随机过程 $\{X(n)\}$ 或 $\{X_n\}$ 和连续时间随机过程 $\{X(t)\}$ 或 $\{X_t\}$ .

随机过程 $X(t)$ 可以理解为随时间变化的、随机变量的函数。当时间 $t$ 取定时，随机过程就是一个随机变量。

<br>

（随机过程的二元函数观） 随机过程也可以看作是一个二元函数 $X(\omega,t):\varOmega\times T\to\mathbb{R}$ ，其中 $\varOmega$ 为样本空间， $T$ 为指标集。

固定 $\omega$ ，则 $X(\omega,t)$ 随着 $t\in T$ 变化，表示随机过程的一次实现，称为一个**样本轨道**。

固定一个时刻 $t$ ，表示一个一元随机变量 $X(t)$ ，任取 $n$ 个时刻 $t_1,\cdots,t_n$ 则 $(X(t_1),\cdots,X(t_n))$ 表示 $n$ 元随机变量。

<br>

（状态空间，状态） 所有时刻的 $\{X(t),t\in T\}$ 的可能取值的全体称为状态空间 $\mathcal{S}$ . $\mathcal{S}$ 中的元素称为状态。例如 $X(t)=x\in \mathcal{S}$ 代表随机过程在 $t$ 时刻处于状态 $x$ .

根据随机过程的时间和状态是连续或离散的，分为4类：

1. 连续时间连续状态，例如连续时间随相正弦波 $X(t)=A\cos(\omega t+\varTheta)$ .
2. 连续时间离散状态，例如进入超市的顾客数 $\{N(t),t\geqslant 0\}$ .
3. 离散时间连续状态，例如离散时间随相正弦波 $X_n=A\cos(\omega n+\varTheta),\;n\in\mathbb{Z},\;\varTheta\in[-\pi,\pi]$ .
4. 离散时间离散状态，例如无穷次抛硬币实验 $\{X_n,\;n=1,2,\cdots\}$ .

<br>

（随机过程的有限维分布族） $\forall n\in\mathbb{N}^+,\;\forall t_1,\cdots,t_n\in T$ ，联合分布：

$$
F_{X(t_1),\cdots,X(t_n)}(x_1,\cdots,x_n)=P(X(t_1)\leqslant x_1,\cdots,X(t_n)\leqslant x_n)
$$

的全体称为随机过程的有限维分布族，可以看作是随机过程的累积分布函数 CDF，通过求导可以得到概率密度函数 PDF（连续情况）。

如果各个变量独立，等价于 PDF 或 CDF 可分解：

$$
\begin{gather*}
F_{X(t_1),\cdots,X(t_n)}(x_1,\cdots,x_n)=F_{X(t_1)}(x_1)\cdots F_{X(t_n)}(x_n) \\
f_{X(t_1),\cdots,X(t_n)}(x_1,\cdots,x_n)=f_{X(t_1)}(x_1)\cdots F_{X(t_n)}(x_n)
\end{gather*}
$$

## 随机过程的数字特征

> 与随机变量相比，随机过程仅仅是多了一个时间参数，因此均值、方差等描述随机变量的特征，用来描述随机过程时，只是作用在随机性上，均与时间无关。也就是说，随机过程的数字特征仍然是关于时间参数的函数。
>
> 对于时间的作用，我们感兴趣的是随机过程在不同时间点上随机变量之间的关系，也就是相关性。
>
> 考虑最一般的情况，我们研究的是复随机过程，因此相关函数等都需要加共轭符号。当然，实际中我们研究的大多都是实随机过程。

均值函数（一阶原点矩） $\mu_X(t)=\mathrm{E}[X(t)]$ .

方差函数（二阶中心矩） $\mathrm{Var}_X(t)=\mathrm{E}\left[ |X(t)-\mu_X(t)|^2\right]=R_X(t,t)-|\mu_X(t)|^2=C_X(t,t)$ .

自相关函数 $R_X(t,s)=\mathrm{E}\left[ X(t) \overline{X(s)} \right]$ .

自协方差函数 $C_X(t,s)=\mathrm{E}\left[ \left(X(t)-\mu_X(t)\right)\overline{\left(X(s)-\mu_X(s)\right)} \right]=R_X(t,s)-\mu_X(t)\overline{\mu_X(s)}$ . 若为**0均值**随机过程，则 $C_X(t,s)=R_X(t,s)$ .

互相关函数 $R_{XY}(t,s)=\mathrm{E}\left[ X(t) \overline{Y(s)} \right]$ .

互协方差函数 $C_{XY}(t,s)=\mathrm{E}\left[ \left(X(t)-\mu_X(t)\right)\overline{\left(Y(s)-\mu_Y(s)\right)} \right]=R_{XY}(t,s)-\mu_X(t)\overline{\mu_Y(s)}$ . 若为**0均值**随机过程，则 $C_{XY}(t,s)=R_{XY}(t,s)$ .

上式均具有共轭性质，例如 $R_{XY}(t,s)=\overline{R_{YX}(s,t)}$ .

<br>

两个随机过程和的方差 $\mathrm{Var}(X(t)\pm Y(s))=\mathrm{Var}(X(t))+\mathrm{Var}(Y(s))\pm 2C_{XY}(t,s)$ .

相关系数 $\rho_{XY}(t,s)=\dfrac{C_{XY}(t,s)}{\sqrt{\mathrm{Var}(X(t))\mathrm{Var}(Y(s))}}$ .

独立 $f_{X(t)Y(s)}(x,y)=f_{X(t)}(x)f_{Y(s)}(y)$ .

不相关 $R_{XY}(t,s)=\mathrm{E}\left[ X(t) \overline{Y(s)} \right]=\mathrm{E}[X(t)]\mathrm{E}[\overline{Y(s)}]=\mu_X(t)\overline{\mu_Y(s)}$ . or $C_{XY}(t,s)=\rho_{XY}(t,s)=0$ .

<br>

（向量随机过程） 同一参数集 $T$ 上的多个随机过程，可记为向量随机过程 $\boldsymbol{X}(\omega,t):\varOmega\times T\to\mathbb{R}^d$ ，则有：

$$
\begin{gather*}
\boldsymbol{\mu}_{\boldsymbol{X}}(t)=\mathrm{E}[\boldsymbol{X}(t)] \\
\boldsymbol{R}_{\boldsymbol{X}}(t,s)=\mathrm{E}[\boldsymbol{X}(t)\boldsymbol{X}(s)^T] \\
\boldsymbol{C}_{\boldsymbol{X}}(t,s)=\boldsymbol{R}_{\boldsymbol{X}}(t,s)-\boldsymbol{\mu}_{\boldsymbol{X}}(t)\boldsymbol{\mu}_{\boldsymbol{X}}(s)^T
\end{gather*}
$$

## 二阶矩过程

若 $\forall t\in T$ ，随机变量 $X(t)$ 的均值和方差都存在，则称 $X(t)$ 为二阶矩过程。其等价定义是 $\mathrm{E}\left[|X(t)|^2\right]<+\infty$ 有限。常见随机过程均为二阶矩过程。

根据定义，二阶矩过程的均值和方差都存在。通过内积空间和 Cauchy-Schwarz 不等式可以推导出，二阶矩过程的自相关函数、自协方差函数、互相关函数、互协方差函数等其他数字特征也都存在。

二阶矩过程的自相关函数具有以下性质：

（1）共轭对称性

连续情形 $R_X(t,s)=R_X^*(s,t)$ .

离散情形：采样得到的随机变量序列 $\boldsymbol{X}=[X(t_1),\cdots,X(t_n)]^T$ ，自相关矩阵 $\boldsymbol{R}_{\boldsymbol{X}}=\mathrm{E}\left(\boldsymbol{X}\boldsymbol{X}^H\right)$ 是共轭对称矩阵（Hermite 矩阵）。

（2）非负定性

自相关矩阵 $\boldsymbol{R}_{\boldsymbol{X}}=\mathrm{E}\left(\boldsymbol{X}\boldsymbol{X}^H\right)$ 是非负定矩阵。

对于任意 $n$ 维确定性向量 $\boldsymbol{\alpha}=[\alpha_1,\cdots,\alpha_n]^T$ 和采样得到的随机变量序列 $\boldsymbol{X}=[X(t_1),\cdots,X(t_n)]^T$ 有：

$$
\boldsymbol{\alpha}^H\boldsymbol{R}_{\boldsymbol{X}}\boldsymbol{\alpha}=\mathrm{E}\left(\boldsymbol{\alpha}^H\boldsymbol{X}\boldsymbol{X}^H\boldsymbol{\alpha}\right)=\mathrm{E}\left[\left(\sum_{i=1}^n\alpha_i^* X(t_i)\right)\left(\sum_{i=1}^n\alpha_i^* X(t_i)\right)^*\right]\geqslant 0
$$

## 平稳过程

若随机过程的统计特性不随时间参数的平移而改变，称其具有平稳性。

### 严平稳 Strict Stationary

随机过程 $\{X(t),\;t\in T\}$ ，对于 $n\geqslant 1,\;t_1,\cdots,t_n\in T$ 和 $\tau\in \mathbb{R},\;t_1+\tau,\cdots t_n+\tau\in T$ ，若多维随机变量 $(X(t_1),\cdots,X(t_n))$ 和 $(X(t_1+\tau),\cdots,X(t_n+\tau))$ 分布完全相同，则 $X(t)$ 为严平稳随机过程(SSS)。

$$
F_{t_1,\cdots,t_n}(x_1,\cdots,x_n)=F_{t_1+\tau,\cdots,t_n+\tau}(x_1,\cdots,x_n)
$$

上式的含义是：严平稳过程的任意有限维分布不随时间改变。

<br>

一维分布函数 $F_X(x;t)=F_X(x;t+\tau)=F_X(x;0)$ ，任意两个时刻分布相同，也即严平稳过程一维分布是确定的，与时间无关。

二维分布函数 $F_X(x_1,x_2;t_1,t_2)=F_X(x_1,x_2;t_1+\tau,t_2+\tau)=F(x_1,x_2;t_1-t_2,0)$ ，即严平稳过程的二维分布之和只和时间差有关（相对时间），与绝对时间无关。

严平稳过程的均值和方差为常数，自相关函数和协方差函数仅为时间差 $t_1-t_2$ 的函数。

严平稳过程通常是十分稳定的信号，多数性质都不随时间改变，例如功率谱密度为常数的白噪声信号。

!!! tip
    严平稳过程的高阶矩与时间无关，可作为判断一个过程是否为严平稳的必要条件。

    对于 $\varphi:\mathbb{R}\to \mathbb{R},\;\mathrm{E}[\varphi(X(t))]=\displaystyle\int \varphi(x)\mathrm{d}F_{X(t)}(x)=\displaystyle\int \varphi(x)\mathrm{d}F_{X(0)}(x)=\mathrm{E}[\varphi(X(0))]$ ，令 $\varphi(x)=x^n$ ，高阶矩与时间无关。

### 宽平稳 Wide-sense Stationary

> 严平稳对随机过程的要求太高，且研究起来性质过于简单，无法传递有效信息。宽平稳放宽了对分布的要求，是对严平稳的推广。
>
> 因此，严平稳过程必然满足宽平稳，但是宽平稳过程不一定是严平稳的。

若二阶矩过程 $\{X(t)\}$ 的均值为常数，且自相关函数仅与时间差 $\tau=t-s$ 有关，则为宽平稳随机过程(WSS)。

$$
\begin{gather*}
\mathrm{E}[X(t)]=\mu_X,\quad  R_X(t,s)=R_X(t-s)=R_X(\tau) \\
R_X(\tau)=\mathrm{E}\left[X(t+\tau)\overline{X(t)}\right]
\end{gather*}
$$

由此可推出，宽平稳过程的自协方差函数也仅与时间差有关：

$$
C_X(t,s)=R_X(t,s)-\mu_X(t)\mu_X(s) \implies C_X(\tau)=R_X(\tau)-\mu_X^2
$$

宽平稳过程的方差也为常数：

$$
\sigma_X^2=[R_X(0)]^2-\mu_X^2=C_X(0)
$$

对于两个不同的随机过程，可定义联合宽平稳： $\{X(t)\},\;\{Y(t)\}$ 的互相关函数只与时间差有关

$$
R_{XY}(t,s)=R_{XY}(t+\tau,s+\tau),\quad \forall t,s,\tau \in T
$$

宽平稳过程的性质：

（1）自相关函数和自协方差函数的共轭对称性 $R_X(\tau)=\overline{R_X(-\tau)},\;C_X(\tau)=\overline{C_X(\tau)}$ .

（2）自相关函数在零点处取最大值 $R_X(0)\geqslant |R_X(\tau)|,\; R_X(0)\geqslant \mu_X^2$ . 自协方差函数也在零点取最大值 $C_X(0)=\mathrm{Var}(X(t))\geqslant|C_X(\tau)|$ .

!!! tip
    由于宽平稳过程的均值为常数，我们通常令 $X(t)-\mu_X$ 为新的 $X(t)$ ，变为零均值，且不改变其他信息，更方便研究。之后我们的讨论的大多都是 零均值宽平稳随机过程。

!!! note
    多数宽平稳过程都不是严平稳的，但高斯过程是个例外。**宽平稳的高斯过程一定是严平稳的**，因为高斯过程的任意有限维分布函数仅由均值和协方差函数决定。

    有关高斯过程的讨论将在 [Chapter 5 Gaussian Process 高斯过程](./chapter5.md) 中进行。

### 相关系数与相关时间

为衡量相隔时间为 $\tau$ 的两个随机变量 $X(t+\tau),X(t)$ 之间的**线性相关程度**，引入相关系数（归一化协方差函数，标准协方差函数） $r_X(\tau)=\dfrac{C_X(\tau)}{C_X(0)}$ .

若 $r_X(\tau)=\pm 1$ ，代表 $X(t+\tau),X(t)$ 完全线性相关（正相关，负相关）；若 $r_X(\tau)=0$ 则 $X(t+\tau),X(t)$ 线性不相关。

为衡量当时间差 $\tau$ 达到多大时， $X(t+\tau),X(t)$ 的相关程度可以忽略，引入相关时间 $\tau_0=\displaystyle\int_0^{+\infty}r_X(\tau)\mathrm{d}\tau$ .

若 $\tau_0$ 较小，说明 $r_X(\tau)$ 随时间增大而迅速衰减，可认为该过程随时间起伏变化剧烈。

## 宽平稳过程的的谱分析

> 在之前的《信号与系统》课程中，我们研究的都是确定性信号。而真实信号都是随机的，不存在完全确定的信号。随机过程的目标是研究随机信号。

宽平稳由于具有时间相关性，时间差可以作为一个独立的变量拿出来，进行谱分析。不满足宽平稳的过程不具有谱。

宽平稳过程的谱密度函数，定义为自相关函数的傅里叶变换（**维纳-欣钦定理**）：

$$
\begin{align*}
S_X(\omega)&=\int_{-\infty}^{+\infty}R_X(\tau)\mathrm{e}^{-\mathrm{j}\omega \tau}\mathrm{d}\tau \\
S_X(\omega)&=|X(\omega)|^2\geqslant 0  \\
R_X(\tau)&=\frac{1}{2\pi}\int_{-\infty}^{+\infty}S_X(\omega)\mathrm{e}^{\mathrm{j}\omega \tau}\mathrm{d}\omega
\end{align*}
$$

以频率 $f$ 为变量的形式：

$$
\begin{align*}
S_X(f)&=\int_{-\infty}^{+\infty}R_X(\tau)\mathrm{e}^{-\mathrm{j}2\pi f \tau}\mathrm{d}\tau \\
R_X(\tau)&=\int_{-\infty}^{+\infty}S_X(f)\mathrm{e}^{\mathrm{j}2\pi f \tau}\mathrm{d}f
\end{align*}
$$

$R_X(0)=\dfrac{1}{2\pi}\displaystyle\int_{-\infty}^{+\infty}S_X(\omega)\mathrm{d}\omega=\displaystyle\int_{-\infty}^{+\infty}S_X(f)\mathrm{d}f$ 称为 随机过程 $X(t)$ 的**功率**。

!!! tip
    显然，功率谱密度为**正实数**，具有实际的物理意义。功率谱密度表示随机过程在不同角频率上的平均功率。

    对于宽平稳的**实随机过程**，自相关函数和功率谱密度均为偶函数。

线谱过程：宽平稳的 $X(t)=\displaystyle\sum_{k=1}^n X_k\exp(\mathrm{j}\omega_k t)$ ， $n$ 为确定值，且 $\mathrm{E}(X_k)=0,\mathrm{Var}(X_k)=\sigma_k^2$ ，当 $i\neq j$ 时 $X_i,X_j$ 不相关。则有：

$$
\begin{align*}
R_X(\tau)&=\sum_{k=1}^n\sigma_k^2 \exp(\mathrm{j}\omega_k\tau) \\
S_X(\omega)&=2\pi \sum_{k=1}^n\sigma_k^2\delta(\omega-\omega_k)
\end{align*}
$$

对于离散随机过程，谱密度为自相关函数的 DTFT ：

$$
S_X(\omega)=\sum_{n=-\infty}^{+\infty}R_X[n]\mathrm{e}^{-\mathrm{j}\omega n}
$$

常见随机过程的自相关函数和谱密度：

<table class="md-typeset">
    <thead style="font-weight:bold;">
        <tr>
            <th>Autocorrelation Function</th>
            <th>Power Spectral Density</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><span class="arithmatex">
            $$
            R(\tau) = \mathrm{e}^{-\alpha|\tau|},\;\alpha > 0
            $$
            </span></td>
            <td><span class="arithmatex">
            $$
            S(\omega) = \frac{2\alpha}{\omega^2 + \alpha^2}
            $$
            </span></td>
        </tr>
        <tr>
            <td><span class="arithmatex">
            $$
            R(\tau)=\cos(\omega_0 \tau)
            $$
            </span></td>
            <td><span class="arithmatex">
            $$
            S(\omega)=\pi(\delta(\omega-\omega_0)+\delta(\omega+\omega_0))
            $$
            </span></td>
        </tr>
        <tr>
            <td><span class="arithmatex">
            $$
            R(\tau)=\dfrac{\sin(\omega_0\tau)}{\omega_0\tau}
            $$
            </span></td>
            <td><span class="arithmatex">
            $$
            S(\omega)=\begin{cases} \dfrac{\pi}{\omega_0},& |\omega|\leqslant\omega_0 \\ 0,& |\omega|>\omega_0 \end{cases}
            $$
            </span></td>
        </tr>
        <tr>
            <td><span class="arithmatex">
            $$R(\tau)=\begin{cases} 1-\frac{2|\tau|}{T}, & |\tau|\leqslant \frac{T}{2} \\ 0,
                &|\tau|>\frac{T}{2}\end{cases}
            $$
            </span></td>
            <td><span class="arithmatex">
            $$
            S(\omega)=\dfrac{8\sin^2\left(\dfrac{\omega T}{4}\right)}{\omega^2T}
            $$
            </span></td>
        </tr>
        <tr>
            <td><span class="arithmatex">
            $$
            R(\tau)=\mathrm{e}^{-\alpha|\tau|}\cos(\omega_0\tau),\,\alpha>0
            $$
            </span></td>
            <td><span class="arithmatex">
            $$
            S(\omega)=\dfrac{\alpha}{(\omega-\omega_0)^2+\alpha^2}+\dfrac{\alpha}{(\omega+\omega_0)^2+\alpha^2}
            $$
            </span></td>
        </tr>
        <tr>
            <td><span class="arithmatex">
            $$
            R(\tau)=\mathrm{e}^{-\alpha\tau^2},\,\alpha>0
            $$
            </span></td>
            <td><span class="arithmatex">
            $$
            S(\omega)=\sqrt{\dfrac{\pi}{\alpha}}\mathrm{e}^{-\omega^2/4\alpha}
            $$
            </span></td>
        </tr>
        <tr>
            <td><span class="arithmatex">
            $$
            R(\tau)=\mathrm{e}^{-\alpha\tau^2}\cos(\beta\tau),\,\alpha>0
            $$
            </span></td>
            <td><span class="arithmatex">
            $$
            S(\omega)=\dfrac{1}{2}\sqrt{\dfrac{\pi}{\alpha}}\left(\mathrm{e}^{-(\omega-\beta)^2/4\alpha}+\mathrm{e}^{-(\omega+\beta)^2/4\alpha}\right)
            $$
            </span></td>
        </tr>
        <tr>
            <td><span class="arithmatex">
            $$
            R(\tau)=\dfrac{2\sin\left(\frac{\Delta\omega}{2}\tau\right)}{\pi\tau}\cos(\omega_0\tau)
            $$
            </span></td>
            <td><span class="arithmatex">
            $$
            S(\omega)=\begin{cases} 1,&\omega_0-\dfrac{\Delta\omega}{2}\leqslant |\omega| \leqslant
                \omega_0+\dfrac{\Delta\omega}{2} \\ 0, & \text{else}\end{cases}
            $$
            </span></td>
        </tr>
    </tbody>
</table>

### 白噪声

白噪声的谱密度函数为常数，在整个频域内均匀分布（理想情况），该过程在各个频率分量上强度相等。自相关函数为冲激函数：

$$
S_X(\omega)=n_0,\quad R_X(\tau)=n_0\delta(\tau)
$$

白噪声的“白“描述的是频域特征，也就是随机过程的时间相关性。根据冲激函数的性质，不同时间得到的自相关函数值为0，也就是白噪声**任意两个时刻都不相关**。这一性质与某一时刻的分布无关，因此白噪声可以有很多种，最常见的高斯白噪声，即一系列互不相关的高斯分布按时间排列。

理想的白噪声不存在，因为实际信号带宽不可能无限大。实际应用中，在我们感兴趣的频率范围内，谱密度为常数的信号就可以认为是白噪声。

### 宽平稳过程通过 LTI 系统

宽平稳过程 $\{X(t)\}$ 通过 冲激响应为 $h(t)$ 的 LTI 系统，输出 $\{Y(t)\}$ 也是随机过程。

经过推导有：

$$
\begin{gather*}
R_Y(t,s)=\int_{-\infty}^{+\infty}h^*(-x)R_{YX}(t-s-x)\mathrm{d}x \\
R_{YX}(v)=\mathrm{E}[Y(s+v)X^*(s)]=\int_{-\infty}^{+\infty}h(v-x)R_{X}(x)\mathrm{d}x
\end{gather*}
$$

因此，当宽平稳过程通过线性时不变系统时：

1. 输出与输入联合宽平稳。
2. 输出也为宽平稳过程。

$$
\begin{align*}
R_Y(\tau)&=R_X(\tau)*h(\tau)*h(-\tau) \\
S_Y(\omega)&=S_X(\omega)|H(\omega)|^2
\end{align*}
$$

若随机过程 $\{X(t)\},\{Y(t)\}$ 联合宽平稳，互谱密度定义为互相关函数的傅里叶变换：

$$
\begin{align*}
S_{XY}(\omega)&=\mathcal{F}[R_{XY}(\tau)]=\int_{-\infty}^{+\infty}R_{XY}(\tau)\mathrm{e}^{-\mathrm{j}\omega \tau}\mathrm{d}\tau \\
S_{YX}(\omega)&=\mathcal{F}[R_{YX}(\tau)]=\int_{-\infty}^{+\infty}R_{YX}(\tau)\mathrm{e}^{-\mathrm{j}\omega \tau}\mathrm{d}\tau \\
S_{XY}(\omega)&=\overline{S_{YX}(\omega)}
\end{align*}
$$

总结：

时域：

$$
\begin{gather*}
X(t) \to \boxed{h(t)} \to Y(t) \\
R_X(\tau) \to \boxed{h(t)} \to R_{YX}(\tau) \to \boxed{\overline{h(-t)}} \to R_Y(\tau) \\ \\
Y(t)=h(t)*X(t) \\
R_{YX}(\tau)=h(\tau)*R_X(\tau), \quad R_Y(\tau)=\overline{h(-\tau)}*h(\tau)*R_X(\tau)
\end{gather*}
$$

频域：

$$
\begin{gather*}
S_X(\omega) \to \boxed{H(\omega)} \to S_{YX}(\omega) \to \boxed{\overline{H(\omega)}} \to S_Y(\omega) \\ \\
\hat{Y}(\omega)=H(\omega)\hat{X}(\omega) \\
S_{YX}(\omega)=H(\omega)S_X(\omega), \quad S_Y(\omega)=|H(\omega)|^2S_X(\omega)
\end{gather*}
$$

如果 $\forall t,s$ 都有 $X(t),Y(s)$ 不相关，即互相关函数 $R_{XY}(\tau)=0,\;\forall \tau$ ，等价于 $S(\omega)=0,\;\forall \omega$ .

<br>

若 $\{X(t)\},\{Y(t)\}$ 均为宽平稳且联合宽平稳，则 $\{Z(t)=X(t)+Y(t)\}$ 的自相关函数和谱密度为：

$$
\begin{align*}
R_Z(t)&=\mathrm{E}\left[(X(t+\tau)+Y(t+\tau))\overline{(X(t)+Y(t))}\right] \\
&=R_X(\tau)+R_{XY}(\tau)+R_{YX}(\tau)+R_Y(\tau) \\
S_Z(\omega)&=S_X(\omega)+S_{XY}(\omega)+S_{YX}(\omega)+S_Y(\omega)
\end{align*}
$$

!!! note
    虽然互相关函数和互谱密度是傅里叶变换对的关系，但是互谱密度不具有描述随机信号随频率分布的意义。

## 增量过程

增量过程是一种典型的非平稳的二阶矩过程。

（正交增量过程） 若 $\forall t_1<t_2\leqslant t_3<t_4$ ，二阶矩过程 $\{X(t)\}$ 满足 $\mathrm{E}\left[(X(t_2)-X(t_1))\overline{(X(t_4)-X(t_3))}\right]=0$ ，称为 正交增量过程。

（定理） 设 $\{X(t)\}$ 在起始时刻归 0，即 $X(0)=0$ ，则 $\{X(t)\}$ 为正交增量过程的充分必要条件为自相关函数 $R_X(t,s)=F(\min\{t,s\})$ ，其中 $F(\cdot)$ 为单调不减函数。

<br>

（独立增量过程） 若 $\forall t_1<t_2\leqslant t_3<t_4$ ，二阶矩过程 $\{X(t)\}$ 满足 $X(t_2)-X(t_1)$ 和 $X(t_4)-X(t_3)$ 相互独立，称为独立增量过程。

<br>

（平稳增量过程） 若 $\forall t_1,t_2$ ，二阶矩过程 $\{X(t)\}$ 满足 $X(t_2)-X(t_1)$ 的概率分布仅仅取决于 $t_2-t_1$ ，称为平稳增量过程。

增量过程的一个典型例子是随机游走。

!!! note
    泊松过程是典型的增量过程，具体讨论参考 [Chapter 7 Poisson Process 泊松过程](./chapter7.md)。

## 二阶矩过程的连续、导数和积分

（均方极限/均方收敛）设随机变量序列 $\{X_n,n\in\mathbb{N}\}$ 满足 $\mathrm{E}\left(|X_n|^2\right)<+\infty$ ，随机变量 $X$ 满足 $\mathrm{E}\left(|X|^2\right)<+\infty$ ，若 $\displaystyle\lim_{n\to+\infty}\mathrm{E}\left(|X_n-X|^2\right)=0$ ，称 $X_n$ 的均方极限为 $X$ ，也称 $\{X_n,n\in\mathbb{N}\}$ 均方收敛于 $X$ ，记为 $X_n\stackrel{m.s.}{\longrightarrow}X$ .

（柯西准则） 设随机变量序列 $\{X_n,n\in\mathbb{N}\}$ 满足 $\mathrm{E}\left(|X_n|^2\right)<+\infty$ ，随机变量 $X$ 满足 $\mathrm{E}\left(|X|^2\right)<+\infty$ ，则 $X_n\stackrel{m.s.}{\longrightarrow}X$ 的充要条件为 $\mathrm{E}\left(|X_n-X_m|^2\right)\to 0,\;m,n\to\infty$ .

<br>

（均方连续） 对于二阶矩过程 $\{X(t)\}$ ，若 $t\to t_0$ 时 $X(t)\stackrel{m.s.}{\longrightarrow}X(t_0)$ ，也即 $\mathrm{E}\left(|X(t)-X(t_0)|^2\right)\to 0$ ，则 $\{X(t)\}$ 在 $t_0$ 点均方连续。

均方连续的性质可以完全由自相关函数确定，以下几个命题等价：

1. $\forall t_0\in T,\;R_X(t,s)$ 在 $(t_0,t_0)$ 点连续。
2. $X(t)$ 在 $T\times T$ 上连续。
3. $X(t)$ 在 $\mathbb{R}$ 上均方连续。

<br>

（均方导数） 若 $\dfrac{X(t_0+h)-X(t_0)}{h}\stackrel{m.s.}{\longrightarrow}Y(t_0),\;t_0\in T,\;h\to 0$ ，则称 $X()$ 均方意义下的导数为 $Y(t)$ .

（均方导数判定定理） 若 $\dfrac{\partial^2 R_X(t,s)}{\partial t\partial s}$ 在 $(t_0,t_0)$ 处**存在且连续**，则 $X(t)$ 在 $t_0$ 处存在均方导数。

均方导数的性质：

$$
\begin{gather*}
\mathrm{E}[X'(t)]=\frac{\mathrm{d}}{\mathrm{d}t}\mathrm{E}[X(t)],\quad \mathrm{E}\left[X'(t)\overline{X(s)}\right]=\frac{\partial}{\partial t}R_X(t,s) \\
\mathrm{E}\left[X(t)\overline{X'(s)}\right]=\frac{\partial}{\partial s}R_X(t,s),\quad \mathrm{E}\left[X'(t)\overline{X'(s)} \right]=\frac{\partial^2}{\partial t\partial s}R_X(t,s)
\end{gather*}
$$

对于宽平稳过程，由于自相关函数只有一个变量，可以简写为：

$$
\mathrm{E}\left[ X^{(m)}(t)\overline{X^{(n)}(s)} \right]=(-1)^n R_X^{(m+n)}(\tau),\quad \tau=t-s
$$

[^1]: 欧志坚，李刚. *概率论与随机过程*［M］. 北京：清华大学出版社，2022.
