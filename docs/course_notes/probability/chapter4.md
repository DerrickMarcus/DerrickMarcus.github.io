# Chapter 4 Second-order Processes and Spectral Analysis

!!! abstract
    本章的主要内容为随机过程的平稳性分析、相关性，二阶矩过程的性质（均方极限，均方连续，均方可导），宽平稳过程的谱分析等。

## 随机过程基本概念

> 随机过程是概率论的一个自然延伸。
>
> 概率论研究的是随机现象和随机变量，随机变量是静态的、不随外部条件变化的。而随机过程研究的是随着某些参数（时间，空间，频率等）变化的随机现象和随机变量。随机过程可以看作是随机变量从有限维到无限维的自然延伸，是一组无穷多个、相互有关的随机变量[^1]。

随机过程的定义：随机过程是一组依赖于参数 $t$ 的随机变量 $\{X(t),t\in T\}$ . $T$ 称为参数集/指标集，参数 $t$ 称为指标。

根据参数集 $T$ 的性质，可以分为 离散时间随机过程 $\{X(n)\}$ 或 $\{X_n\}$ ，和 连续时间随机过程 $\{X(t)\}$ 或 $\{X_t\}$ .

随机过程 $X(t)$ 可以理解为随时间变化的、随机变量的函数。当时间 $t$ 取定时，随机过程就是一个随机变量。

<br>

随机过程也可以看作是一个二元函数 $X(\omega,t):\varOmega\times T\to\mathbb{R}$ ，其中 $\varOmega$ 为样本空间， $T$ 为指标集。

固定 $\omega$ ，则 $X(\omega,t)$ 随着 $t\in T$ 变化，表示随机过程的一次实现，称为一个**样本轨道**。

固定一个时刻 $t$ ，表示一个一元随机变量 $X(t)$ ，任取 $n$ 个时刻 $t_1,\cdots,t_n$ 则 $(X(t_1),\cdots,X(t_n))$ 表示 $n$ 元随机变量。

<br>

与随机变量相比，随机过程多了时间的概念，自然我们感兴趣的是随机过程在不同时间点上随机变量之间的关系，也就是相关性。

> 考虑最一般的情况，我们研究的是复随机过程，因此相关函数等都需要加共轭符号。当然，实际中我们研究的大多都是实随机过程。

自相关函数 $R_X(t,s)=\mathrm{E}\left[ X(t) \overline{X(s)} \right]$ .

自协方差函数 $C_X(t,s)=\mathrm{E}\left[ \left(X(t)-\mu_X(t)\right)\overline{\left(X(s)-\mu_X(s)\right)} \right]=R_X(t,s)-\mu_X(t)\overline{\mu_X(s)}$ . 若为 **0均值** 随机过程，则 $C_X(t,s)=R_X(t,s)$ .

互相关函数 $R_{XY}(t,s)=\mathrm{E}\left[ X(t) \overline{Y(s)} \right]$ .

互协方差函数 $C_{XY}(t,s)=\mathrm{E}\left[ \left(X(t)-\mu_X(t)\right)\overline{\left(Y(s)-\mu_Y(s)\right)} \right]=R_{XY}(t,s)-\mu_X(t)\overline{\mu_Y(s)}$ . 若为 **0均值** 随机过程，则 $C_{XY}(t,s)=R_{XY}(t,s)$ .

上式均具有共轭性质，例如 $R_{XY}(t,s)=\overline{R_{YX}(s,t)}$ .

<br>

方差 $\mathrm{Var}(X(t))=\mathrm{E}\left[ |X(t)-\mu_X(t)|^2\right]=R_X(t,t)-|\mu_X(t)|^2=C_X(t,t)$ .

两个随机过程和的方差 $\mathrm{Var}(X(t)\pm Y(s))=\mathrm{Var}(X(t))+\mathrm{Var}(Y(s))\pm 2C_{XY}(t,s)$ .

相关系数 $\rho_{XY}(t,s)=\dfrac{C_{XY}(t,s)}{\sqrt{\mathrm{Var}(X(t))\mathrm{Var}(Y(s))}}$ .

独立 $f_{X(t)Y(s)}(x,y)=f_{X(t)}(x)f_{Y(s)}(y)$ .

不相关 $R_{XY}(t,s)=\mathrm{E}\left[ X(t) \overline{Y(s)} \right]=\mathrm{E}[X(t)]\mathrm{E}[\overline{Y(s)}]=\mu_X(t)\overline{\mu_Y(s)}$ . or $C_{XY}(t,s)=\rho_{XY}(t,s)=0$ .

## 二阶矩过程

若 $\forall t\in T$ ，随机变量 $X(t)$ 的均值和方差都存在，则称 $X(t)$ 为 二阶矩过程。其等价定义是 $\mathrm{E}\left[|X(t)|^2\right]<+\infty$ 有限。常见随机过程均为二阶矩过程。

### 严平稳 Strict Stationary

随机过程 $\{X(t),\;t\in T\}$ ，对于 $n\geqslant 1,\;t_1,\cdots,t_n\in T$ 和 $\tau\in \mathbb{R},\;t_1+\tau,\cdots t_n+\tau\in T$ ，若多维随机变量 $(X(t_1),\cdots,X(t_n))$ 和 $(X(t_1+\tau),\cdots,X(t_n+\tau))$ 分布完全相同，则 $X(t)$ 为严平稳随机过程(SSS)。

$$
F_{t_1,\cdots,t_n}(x_1,\cdots,x_n)=F_{t_1+\tau,\cdots,t_n+\tau}(x_1,\cdots,x_n)
$$

上式的含义是：严平稳过程的任意有限维分布不随时间改变。

令 $n=1\Rightarrow\forall t,s,\;F(t;x)=F(s;x)$ ，任意两个时刻分布相同，也即严平稳过程一维分布是确定的。

令 $n=2\Rightarrow F(t,s;x_1,x_2)=F(t+\tau,s+\tau;x_1,x_2)=F(t-s,0;x_1,x_2)$ ，也即严平稳过程的二维分布之和只和时间差有关（相对时间），与绝对时间无关。

严平稳过程通常是十分稳定的信号，多数性质都不随时间改变，例如功率谱密度为常数的白噪声信号。

!!! tip
    严平稳过程的高阶矩与时间无关，可作为判断一个过程是否为严平稳的必要条件。

    对于 $\varphi:\mathbb{R}\to \mathbb{R},\;\mathrm{E}[\varphi(X(t))]=\displaystyle\int \varphi(x)\mathrm{d}F_{X(t)}(x)=\displaystyle\int \varphi(x)\mathrm{d}F_{X(0)}(x)=\mathrm{E}[\varphi(X(0))]$ ，令 $\varphi(x)=x^n$ ，高阶矩与时间无关。

### 宽平稳 Wide-sense Stationary

> 严平稳对随机过程的要求太高，且研究起来性质过于简单，无法传递有效信息。宽平稳放宽了对分布的要求，是对严平稳的推广。

若二阶矩过程 $\{X(t)\}$ 的均值为常数，且自相关函数仅与时间差 $\tau=t-s$ 有关，则为 宽平稳随机过程(WSS)。

$$
\begin{gather*}
\mathrm{E}[X(t)]=\mu_X,\quad  R_X(t,s)=R_X(t-s)=R_X(\tau) \\
R_X(\tau)=\mathrm{E}\left[X(t+\tau)\overline{X(t)}\right]
\end{gather*}
$$

宽平稳的自相关函数具有共轭反对称性质 $R_X(\tau)=\overline{R_X(-\tau)}$ . 自相关函数在零点处取得最大值 $R_X(0)\geqslant |R_X(\tau)|,\; R_X(0)\geqslant \mu_X^2$ .

由此可推出，宽平稳过程的方差也为常数：

$$
\sigma_X^2=\mathrm{Var}(X(t))=\mathrm{E}[X^2(t)]-\mu_X^2=[R_X(0)]^2-\mu_X^2
$$

宽平稳过程的自协方差函数也仅与时间差有关：

$$
C_X(t,s)=R_X(t,s)-\mu_X(t)\mu_X(s) \implies C_X(\tau)=R_X(\tau)-\mu_X^2
$$

也具有共轭反对称性 $C_X(\tau)=\overline{C_X(\tau)}$ .

<br>

对于两个不同的随机过程，可定义 联合宽平稳： $\{X(t)\},\;\{Y(t)\}$ 的互相关函数只与时间差有关

$$
R_{XY}(t,s)=R_{XY}(t+\tau,s+\tau),\quad \forall t,s,\tau \in T
$$

由于宽平稳过程的均值为常数，我们通常令 $X(t)-\mu_X$ 为新的 $X(t)$ ，变为零均值，且不改变其他信息，更方便研究。之后我们的讨论的大多都是 零均值宽平稳随机过程。

### 二阶矩过程的导数

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

## 宽平稳过程的的谱分析

> 在之前的《信号与系统》课程中，我们研究的都是确定性信号。而真实信号都是随机的，不存在完全确定的信号。随机过程的目标是研究随机信号。

宽平稳由于具有时间相关性，时间差可以作为一个独立的变量拿出来，进行谱分析。不满足宽平稳的过程不具有谱。

宽平稳过程的谱密度函数，定义为 自相关函数的傅里叶变换：

$$
S_X(\omega)=\int_{-\infty}^{+\infty}R_X(\tau)e^{-\mathrm{j}\omega \tau}\mathrm{d}\tau
$$

上式可由 维纳-欣钦 定理得出。

对于离散随机过程，谱密度为 自相关函数的 DTFT ：

$$
S_X(\omega)=\sum_{n=-\infty}^{+\infty}R_X[n]e^{-\mathrm{j}\omega n}
$$

<br>

常见随机过程的自相关函数和谱密度：

<table>
    <thead>
        <tr>
            <th>Autocorrelation Function</th>
            <th>Power Spectral Density</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>$$R(\tau) = \mathrm{e}^{-\alpha|\tau|},\ \alpha > 0$$</td>
            <td>$$S(\omega) = \frac{2\alpha}{\omega^2 + \alpha^2}$$</td>
        </tr>
        <tr>
            <td>$$R(\tau)=\cos(\omega_0 \tau)$$</td>
            <td>$$S(\omega)=\pi(\delta(\omega-\omega_0)+\delta(\omega+\omega_0))$$</td>
        </tr>
        <tr>
            <td>$$R(\tau)=\dfrac{\sin(\omega_0\tau)}{\omega_0\tau}$$</td>
            <td>$$S(\omega)=\begin{cases} \dfrac{\pi}{\omega_0},& |\omega|\leqslant\omega_0 \\ 0,& |\omega|>\omega_0
                \end{cases}$$</td>
        </tr>
        <tr>
            <td>$$R(\tau)=\begin{cases} 1-\frac{2|\tau|}{T}, & |\tau|\leqslant \frac{T}{2} \\ 0,
                &|\tau|>\frac{T}{2}\end{cases}$$</td>
            <td>$$S(\omega)=\dfrac{8\sin^2\left(\dfrac{\omega T}{4}\right)}{\omega^2T}$$</td>
        </tr>
        <tr>
            <td>$$R(\tau)=\mathrm{e}^{-\alpha|\tau|}\cos(\omega_0\tau),\,\alpha>0$$</td>
            <td>$$S(\omega)=\dfrac{\alpha}{(\omega-\omega_0)^2+\alpha^2}+\dfrac{\alpha}{(\omega+\omega_0)^2+\alpha^2}$$
            </td>
        </tr>
        <tr>
            <td>$$R(\tau)=\mathrm{e}^{-\alpha\tau^2},\,\alpha>0$$</td>
            <td>$$S(\omega)=\sqrt{\dfrac{\pi}{\alpha}}\mathrm{e}^{-\omega^2/4\alpha}$$</td>
        </tr>
        <tr>
            <td>$$R(\tau)=\mathrm{e}^{-\alpha\tau^2}\cos(\beta\tau),\,\alpha>0$$</td>
            <td>$$S(\omega)=\dfrac{1}{2}\sqrt{\dfrac{\pi}{\alpha}}\left(\mathrm{e}^{-(\omega-\beta)^2/4\alpha}+\mathrm{e}^{-(\omega+\beta)^2/4\alpha}\right)$$
            </td>
        </tr>
        <tr>
            <td>$$R(\tau)=\dfrac{2\sin\left(\frac{\Delta\omega}{2}\tau\right)}{\pi\tau}\cos(\omega_0\tau)$$</td>
            <td>$$S(\omega)=\begin{cases} 1,&\omega_0-\dfrac{\Delta\omega}{2}\leqslant |\omega| \leqslant
                \omega_0+\dfrac{\Delta\omega}{2} \\ 0, & \text{else}\end{cases}$$</td>
        </tr>
    </tbody>
</table>

!!! tip
    实际信号的功率谱密度为**正实数**，这样功率才有实际的物理意义。

### 白噪声

白噪声的谱密度函数为常数，在整个频域内均匀分布（理想情况），自相关函数为冲激函数：

$$
S_X(\omega)=n_0,\quad R_X(\tau)=n_0\delta(\tau)
$$

白噪声的“白“描述的是频域特征，也就是随机过程的时间相关性。根据冲激函数的性质，不同时间得到的自相关函数值为0，也就是白噪声 **任意两个时刻都不相关**。这一性质与某一时刻的分布无关，因此白噪声可以有很多种，最常见的高斯白噪声，即一系列互不相关的高斯分布按时间排列。

### 宽平稳过程通过 LTI 系统

宽平稳过程通过 $\{X(t)\}$ 通过 冲激响应为 $h(t)$ 的 LTI 系统，输出 $\{Y(t)\}$ 也是随机过程。

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

<br>

随机过程 $\{X(t)\},\;\{Y(t)\}$ 联合宽平稳，互谱密度 定义为 互相关函数的傅里叶变换：

$$
S_{XY}(\omega)=\int_{-\infty}^{+\infty}R_{XY}(\tau)e^{-\mathrm{j}\omega \tau}\mathrm{d}\tau
$$

如果 $\forall t,s$ 都有 $X(t),Y(s)$ 不相关，即互相关函数 $R_{XY}(\tau)=0,\;\forall \tau$ ，等价于 $S(\omega)=0,\;\forall \omega$ .

若 $\{X(t)\},\;\{Y(t)\}$ 均为宽平稳且联合宽平稳，则 $\{Z(t)=X(t)+Y(t)\}$ 的自相关函数和谱密度为：

$$
\begin{align*}
R_Z(t)&=\mathrm{E}\left[(X(t+\tau)+Y(t+\tau))\overline{(X(t)+Y(t))}\right] \\
&=R_X(\tau)+R_{XY}(\tau)+R_{YX}(\tau)+R_Y(\tau) \\
S_Z(\omega)&=S_X(\omega)+S_{XY}(\omega)+S_{YX}(\omega)+S_Y(\omega)
\end{align*}
$$

!!! note
    虽然 互相关函数 和 互谱密度 是傅里叶变换对的关系，但是 互谱密度 不具有描述随机信号随频率分布的意义。

[^1]: 欧志坚，李刚. *概率论与随机过程*［M］. 北京：清华大学出版社，2022.
