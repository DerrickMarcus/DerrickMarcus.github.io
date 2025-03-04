# 第6章 信号的矢量空间

## 1. 基本概念

1-范数：信号的作用强度（信号作用对时间的累积）

2-范数：信号的能量

$\infty$-范数：信号的幅度/峰值

对于能量无限大的信号，功率定义为：
$$
\lim_{T\to\infty}\frac{1}{T} \int_{-T/2}^{T/2}|x(t)|^2\,\mathrm{d}t
$$
功率的平方根：方均根值（rms）

平均值/直流分量：
$$
\lim_{T\to\infty}\frac{1}{T} \int_{-T/2}^{T/2}x(t)\,\mathrm{d}t
$$
内积性质的例子：
$$
\left\langle x,y \right\rangle = \int_{-\infty}^{\infty} x(t)y^*(t)\,\mathrm{d}t
\\
\left\langle x,y \right\rangle = \left\langle y,x \right\rangle ^*
\\
\left\langle x,x \right\rangle = ||x||_2^2
\\
|\left\langle x,y \right\rangle|^2 \leqslant \left\langle x,x \right\rangle \cdot \left\langle y,y \right\rangle
$$

## 2. 正交函数分解

对于两个矢量 $x,y$ ，用 $cy$ 逼近 $x$ ，误差最小的为：
$$
c=\frac{\left\langle x,y \right\rangle}{\left\langle y,y \right\rangle}
$$
推广到函数，在区间 $t_1<t<t_2$ 内用 $f_2(t)$ 近似表示 $f_1(t)$ ：$f_1(t)\approx c_{12}f_2(t)$​

方均误差（误差的方均值）：
$$
\overline{\epsilon^2}=\frac{1}{t_2-t_1} \int_{t_1}^{t_2}[f_1(t)-c_{12}f_2(t)]^2\,\mathrm{d}t
$$
方均误差最小（对  $c_{12}$ 的二阶导为0），有：
$$
\begin{align}
c_{12}=\frac{\left\langle f_1(t),f_2(t) \right\rangle}{\left\langle f_2(t),f_2(t) \right\rangle}
=\frac{\int_{t_1}^{t_2} f_1(t)f_2(t)\,\mathrm{d}t}{\int_{t_1}^{t_2} f_2^2(t)\,\mathrm{d}t}
\end{align}
$$
正交的等价条件为：
$$
\int_{t_1}^{t_2} f_1(t)f_2(t)\,\mathrm{d}t=0
$$
对于复变量函数，上面条件变为：
$$
\begin{align}
c_{12}=\frac{\left\langle f_1(t),f_2(t) \right\rangle}{\left\langle f_2(t),f_2(t) \right\rangle}
=\frac{\int_{t_1}^{t_2} f_1(t)f_2^*(t)\,\mathrm{d}t}{\int_{t_1}^{t_2} f_2(t)f_2^*(t)\,\mathrm{d}t}
\end{align}
$$

## 3. 完备正交函数集、帕塞瓦尔定理

帕塞瓦尔方程：
$$
\int_{t_1}^{t_2} f^2(t)\,\mathrm{d}t=\sum_{r=1}^{\infty}c_r^2K_r
$$
一个信号的功率等于它在完备正交函数集中各分量的功率之和。这体现了正交函数的范数不变性 / 内积不变性。

## 4. 相关

### 功率信号与能量信号

对于能量信号，能量有限：
$$
\begin{align}
E &= \int_{-\infty}^{\infty} |f(t)|^2\,\mathrm{d}t
\\
&= \int_{-\infty}^{\infty} f^2(t)\,\mathrm{d}t\quad(\text{real\ variable})
\end{align}
$$
对于功率信号，能量无限，平均功率有限：
$$
\begin{align}
P &= \lim_{T\to\infty} \int_{-T/2}^{T/2}|f(t)|^2 \,\mathrm{d}t
\\
&= \lim_{T\to\infty} \int_{-T/2}^{T/2} f^2(t) \,\mathrm{d}t\quad(\text{real\ variable})
\end{align}
$$
有些信号既不属于能量信号，也不属于功率信号，比如 $f(t)=e^t$​ 。

对于能量信号，有相关系数（类似于矢量的夹角）：
$$
\rho_{12}=\frac{\left\langle f_1,f_2 \right\rangle}{||f_1||_2 \cdot ||f_2||_2}
$$

### 相关函数

复变量表示的相关函数，其中 $\tau$ 为两个信号的时差。实函数将共轭去掉即可。

对于能量信号：
$$
\begin{align}
R_{12}(\tau) &= \int_{-\infty}^{\infty} f_1(t)f_2^*(t-\tau)\,\mathrm{d}t
= \int_{-\infty}^{\infty} f_1(t+\tau)f_2^*(t)\,\mathrm{d}t
\\
R(\tau) &= \int_{-\infty}^{\infty} f(t)f^*(t-\tau)\,\mathrm{d}t
= \int_{-\infty}^{\infty} f(t+\tau)f^*(t)\,\mathrm{d}t
\end{align}
$$
对于功率信号：
$$
\begin{align}
R_{12}(\tau) &= \lim_{T\to\infty} \int_{-T/2}^{T/2} f_1(t)f_2^*(t-\tau) \,\mathrm{d}t
=\lim_{T\to\infty} \int_{-T/2}^{T/2} f_1(t+\tau)f_2^*(t-) \,\mathrm{d}t
\\
R(\tau) &= \lim_{T\to\infty} \int_{-T/2}^{T/2} f(t)f^*(t-\tau) \,\mathrm{d}t
=\lim_{T\to\infty} \int_{-T/2}^{T/2} f(t+\tau)f^*(t) \,\mathrm{d}t
\end{align}
$$
性质——共轭反对称： $R_{12}(\tau)=R_{21}^*(-\tau),\quad R(\tau)=R^*(-\tau)$ 。

特别地，对于实函数有： $R_{12}(\tau)=R_{21}(-\tau),\quad R(\tau)=R(-\tau)$ ，自相关函数为偶函数。

周期信号的自相关函数也为周期函数，且周期相同。

![2024春信号与系统19第十七讲6.6-6.11_15](D:\ASUS\Pictures\Saved Pictures\2024春信号与系统19第十七讲6.6-6.11_15.png)

### 与卷积的比较

相关与卷积的关系：卷积要“反褶 + 移位 + 积分”，相关仅为“移位 + 积分”，不用反褶。
$$
R_{12}(\tau) = f_1(t)*f_2(t)
\\
R(\tau) = f(t)*f(-t)
$$
对于实偶函数，卷积和相关结果相同。

### 相关定理

若有 $\mathscr{F}[f_1(t)]=F_1(\omega),\mathscr{F}[f_2(t)]=F_2(\omega)$ ，则有：
$$
\mathscr{F}[R_{12}(\tau)] = F_1(\omega)\cdot F_2^*(\omega)
\\
\mathscr{F}[R(\tau)] = F(\omega)\cdot F^*(\omega)=|F(\omega)|^2
$$
若 $f_2(t)$ 为实偶函数， $F_2(\omega)$​ 为实函数，则与卷积定理结果相同。

### 能量谱和功率谱

能谱：

能量信号，自相关函数在0处的取值 = 时域 $f^2(t)$ 覆盖的面积 = 频域 $|F_1(f)|^2$ 覆盖的面积，时域能量等于频域能量：
$$
R(0)=\int_{-\infty}^{\infty}f^2(t)\,\mathrm{d}t
=\frac{1}{2\pi}\int_{-\infty}^{\infty}|F(\omega)|^2\,\mathrm{d}\omega
=\int_{-\infty}^{\infty}|F_1(f)|^2\,\mathrm{d}f
$$
定义能量谱密度/能谱，反映了单位带宽内的能量：
$$
\mathscr{E}(\omega) = |F(\omega)|^2
\\
\implies \mathscr{E}(\omega) = \mathscr{F}[R(\tau)]
\\
E = \frac{1}{2\pi} \int_{-\infty}^{\infty}\mathscr{E}(\omega)\,\mathrm{d}\omega
=\int_{-\infty}^{\infty}\mathscr{E}_1(\omega)\,\mathrm{d}\omega
$$
自相关函数和能谱函数是一对 Fourier 变换对。

功率谱：
$$
\mathscr{P}(\omega) = \lim_{T\to\infty}\frac{|F_T(\omega)|^2}{T}
\\
\implies \mathscr{P}(\omega) = \mathscr{F}[R(\tau)]
\\
P = \frac{1}{2\pi}\int_{-\infty}^{\infty}\mathscr{P}(\omega)\,\mathrm{d}\omega
$$
自相关函数和功率谱函数是一对 Fourier 变换对。（维纳-欣钦关系）

注意：能量信号和功率信号的自相关函数单位不同， 因此能量谱和功率谱的单位也不同，相差时间单位。

### 线性系统的自相关函数与能量谱、功率谱

在频域，我们有：

能量信号： $\mathscr{E}_r(\omega)=|H(\mathrm{j}\omega)|^2 \cdot \mathscr{E}_e(\omega)$

功率信号： $\mathscr{P}_r(\omega)=|H(\mathrm{j}\omega)|^2 \cdot \mathscr{P}_e(\omega)$

变换到时域，二者共同有： $R_r(\tau)=R_e(\tau)*R_h(\tau)$

## 5. 匹配滤波器

使有用信号 $s(t)$ 增强，抑制噪声 $n(t)$ 。信号和噪声同时进入滤波器，如果在某段时间内信号 $s(t)$ 存在，滤波器的输出在相应的瞬间出现强大的峰值。

有用信号 $s(t)$ 持续时间有限，为 $0 \sim T$ ，则匹配滤波器的冲激响应为 $h(t)=ks(t_m-t)$

考虑到系统因果可实现（$t_m\geqslant T$）+ 观察时间 $t_m$ 尽可能小（$t_m=T$） + 取系数 $k$ 为1，得到：$h(t)=s(T-t)$ 。

输出为 $s_o(t)=s(t)*h(t)=s(t)*s(T-t)=R_{ss}(t_T)$ 。

匹配滤波器相当于对 $s(t)$ 进行自相关运算，在 $t=T$ 的时刻取得自相关函数的峰值，峰值大小等于信号 $s(t)$​ 的能量 E，且仅与能量有关，与波形无关。

自相关函数的峰值在原点取到： $R_{ss}(0)\geq R_{ss}(\tau)$ ，并且等于信号的能量，有时差之后取值会减少。
