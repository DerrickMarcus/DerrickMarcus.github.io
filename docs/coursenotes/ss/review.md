# 《信号与系统》期末复习

Yanxu Chen, June 6th, 2024

参考：

《信号与系统》课件，2024春，谷源涛

《信号与系统》，郑君里

感谢：李国林（~~Emperor of Electronic Circuit~~），吴昊（~~Emperor of Mathematics~~）



目录：

[TOC]



## 第5章 傅里叶变换应用于通信系统（后半部分）

### 1. 系统的物理可实现性与佩利-维纳准则

时域判断：因果性

频域判断：

首先有平方可积条件：
$$
\int_{-\infty}^{\infty}|H(\mathrm{j}\omega)|^2\,\mathrm{d}\omega<\infty
$$
根据帕塞瓦尔定理：
$$
\int_{-\infty}^{\infty}|h(t)|^2\,\mathrm{d}t=
\frac{1}{2\pi}\int_{-\infty}^{\infty}|H(\mathrm{j}\omega)|^2\,\mathrm{d}\omega
$$
佩利-维纳准则（必要不充分条件）：
$$
\int_{-\infty}^{\infty}\frac{|\ln|H(\mathrm{j}\omega)||}{1+\omega^2}\,\mathrm{d}\omega<\infty
$$
不满足 PW 准则的幅度函数，响应比冲激激励先出现，违反了因果性。

PW 准则不允许 $|H(\mathrm{j}\omega)|$ 在某一频带内恒为0（理想滤波器不可能实现），但允许在一些不连续的点为0。

对数函数  $\ln()$ 限制了 $|H(\mathrm{j}\omega)|\to 0$ 的衰减速度。

PW 准则不是物理可实现的充分条件，因为对相频特性 $\varphi(\omega)$ 没有要求。如果已知一个满足 PW 准则的 $|H(\mathrm{j}\omega)|$ ，可以搭配一个适当的相频特性函数  $\varphi(\omega)$ ，使系统物理可实现。

实际上只有多项式函数和双曲函数满足 PW 准则。



### 2. 希尔伯特变换研究系统的约束特性

对于一个因果稳定的系统有：
$$
\begin{align}
H(\mathrm{j}\omega) &= R(\omega) + \mathrm{j}X(\omega)
\\
\implies R(\omega) &= \frac{1}{\pi} \int_{-\infty}^{\infty} \frac{X(\lambda)}{\omega-\lambda}\,\mathrm{d}\lambda
\\
\implies X(\omega) &= -\frac{1}{\pi} \int_{-\infty}^{\infty} \frac{R(\lambda)}{\omega-\lambda}\,\mathrm{d}\lambda
\end{align}
$$


实部是虚部的 Hilbert 变换，虚部是实部的 Hilbert 逆变换。

可见实部与虚部相互约束，二者可以互相确定，两者不能任意给定。

另外，对于一个最小相移函数， $\ln|H(\mathrm{j}\omega)|$ 与 $\varphi(\omega)$ 也相互约束。



### 3. 调制与解调

#### SC-AM（抑制载波调幅）

调制信号（基带信号） $g(t)$ ，频谱 $G(\omega)$ 占据有限频带 $-\omega_m\sim \omega_m$ 。

已调信号 $f(t)=g(t)\cos(\omega_0t)$ ，把 $g(t)$ 频谱搬移到 $\pm\omega_0$ 上，且分成两部分，各占1/2。

解调：从已调信号 $f(t)$ 恢复出基带信号 $g(t)$ 需要用本地载波 $\cos(\omega_0t)$ ，使频谱 $F(\omega)$ 左右移动，经过低通滤波器（带宽大于 $\omega_m$，小于 $2\omega_0-\omega_m$）之后取出 $G(\omega)$ ，但能量变为原来的一半。
$$
\begin{align}
g_0(t) &= f(t)\cos(\omega_0 t)=g(t)\cos^2(\omega_0 t)=\frac{1}{2}g(t)(1+\cos(2\omega_0 t))
\\
\implies G_0(\omega) &= \frac{1}{2}G(\omega)+\frac{1}{4}[G(\omega+2\omega_0)+G(\omega-2\omega_0)]
\end{align}
$$
注意：$f(t)$ 的频域 $F(\omega)=\frac{1}{2}G(\omega-\omega_0)+\frac{1}{2}G(\omega+\omega_0)$ 不含载波的频谱 $\delta(\omega)$ 。

缺点：解调使用的本地载波需要与发送端相同，接收机较为复杂。

![2024春信号与系统16第十四讲5.5-5.7_22](D:\ASUS\Pictures\Saved Pictures\2024春信号与系统16第十四讲5.5-5.7_22.png)

#### AM（调幅）

不需要本地载波，接收机简单，适用于日常使用，但是发射功率大，价格较贵。卫星上有应用。
$$
f(t)=[A+g(t)]\cos(\omega_0 t)
$$
其中 $K=1/A$ 为调制深度。

$A$ 足够大时，$f(t)$ 的波形包络就是 $A+g(t)$​ ，使用包络检测器（二极管、电容、电阻）即可恢复。

![2024春信号与系统16第十四讲5.5-5.7_26](D:\ASUS\Pictures\Saved Pictures\2024春信号与系统16第十四讲5.5-5.7_26.png)

![2024春信号与系统16第十四讲5.5-5.7_27](D:\ASUS\Pictures\Saved Pictures\2024春信号与系统16第十四讲5.5-5.7_27.png)

#### SSB（单边带）

从中间切开。

为节省频带，只发半个边带，由于频移特性，在收端能恢复。多用于短波通信、跳频电台等。

优点是节省频带，多容纳电台。但“陡峭的”边带滤波器不易制作，所以适用于信号中无直流成分且缺少一段低频成分，此时对边带滤波器的要求放宽。

![2024春信号与系统16第十四讲5.5-5.7_29](D:\ASUS\Pictures\Saved Pictures\2024春信号与系统16第十四讲5.5-5.7_29.png)

#### VSB（残留边带）

斜着切。

![2024春信号与系统16第十四讲5.5-5.7_32](D:\ASUS\Pictures\Saved Pictures\2024春信号与系统16第十四讲5.5-5.7_32.png)

![2024春信号与系统16第十四讲5.5-5.7_33](D:\ASUS\Pictures\Saved Pictures\2024春信号与系统16第十四讲5.5-5.7_33.png)

#### FM，PM（调频与调相）

FM（调频），直接作用于相位： $f(t)=A\cos(\omega_c t+g(t))$ 。

PM（调相），直接作用于频率： $f(t)=A\cos(\omega_c t+\int_{-\infty}^{t}g(\tau)\,\mathrm{d}\tau)$ 。

本质都是调相。

解调过程：对 $f(t)$ 求导，进行包络检波。

![2024春信号与系统16第十四讲5.5-5.7_35](D:\ASUS\Pictures\Saved Pictures\2024春信号与系统16第十四讲5.5-5.7_35.png)

#### FDM（频分复用）

对于不同的信号，调制和解调使用不同的 $\cos(\omega_n t)$ ，在频域上占用不同的频率区间，互不干扰。

另有：时分复用，不同的信号占用不同的时间区间，互不干扰。依据为抽样定理，相当于在时域抽样，频域周期延拓，满足奈奎斯特抽样频率的前提下可以分离信号。实际传送的信号并非冲激抽样，可以占有一小段时间。



### 4. 从抽样信号恢复连续时间信号

1. 冲激抽样信号恢复连续时间信号（常规）：时域抽样——频域周期延拓——低通滤波器——恢复。

2. 零阶抽样保持：

   脉冲信号 $p(t)$ 对信号 $f(t)$ 抽样时，保持样本值到下一次抽样为止，抽样输出信号 $f_s(t)$ 呈阶梯形状。

3. 一阶抽样保持：

   使用一个冲激响应为三角型脉冲的 LTI 系统，使得由抽样信号 $f_s(t)$ 经过该系统形成的三角脉冲叠加恢复出 $f(t)$ 。

零阶抽样保持和一阶抽样保持都是对第一种方法的逼近。

上述三种方法都要求信号 $f(t)$ 频带受限，且抽样频率满足抽样定理要求。

具体内容可以看书。

![2024春信号与系统17第十五讲5.9-5.12_07](D:\ASUS\Pictures\Saved Pictures\2024春信号与系统17第十五讲5.9-5.12_07.png)

![2024春信号与系统17第十五讲5.9-5.12_14](D:\ASUS\Pictures\Saved Pictures\2024春信号与系统17第十五讲5.9-5.12_14.png)

![2024春信号与系统17第十五讲5.9-5.12_19](D:\ASUS\Pictures\Saved Pictures\2024春信号与系统17第十五讲5.9-5.12_19.png)



## 第6章 信号的矢量空间

### 1. 基本概念

~~范数，赋范空间~~

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
~~内积，内积空间~~

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

### 2. 正交函数分解

~~正交矢量，正交函数~~

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

### 3. 完备正交函数集、帕塞瓦尔定理

帕塞瓦尔方程：
$$
\int_{t_1}^{t_2} f^2(t)\,\mathrm{d}t=\sum_{r=1}^{\infty}c_r^2K_r
$$
一个信号的功率等于它在完备正交函数集中各分量的功率之和。这体现了正交函数的范数不变性 / 内积不变性。





### 4. 相关

#### 功率信号与能量信号

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

#### 相关函数

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



#### 与卷积的比较

相关与卷积的关系：卷积要“反褶 + 移位 + 积分”，相关仅为“移位 + 积分”，不用反褶。
$$
R_{12}(\tau) = f_1(t)*f_2(t)
\\
R(\tau) = f(t)*f(-t)
$$
对于实偶函数，卷积和相关结果相同。



#### 相关定理

若有 $ \mathscr{F}[f_1(t)]=F_1(\omega),\ \mathscr{F}[f_2(t)]=F_2(\omega) $ ，则有：
$$
\mathscr{F}[R_{12}(\tau)] = F_1(\omega)\cdot F_2^*(\omega)
\\
\mathscr{F}[R(\tau)] = F(\omega)\cdot F^*(\omega)=|F(\omega)|^2
$$

若 $f_2(t)$ 为实偶函数， $F_2(\omega)$​ 为实函数，则与卷积定理结果相同。



#### 能量谱和功率谱

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



#### 线性系统的自相关函数与能量谱、功率谱

在频域，我们有：

能量信号： $\mathscr{E}_r(\omega)=|H(\mathrm{j}\omega)|^2 \cdot \mathscr{E}_e(\omega)$

功率信号： $\mathscr{P}_r(\omega)=|H(\mathrm{j}\omega)|^2 \cdot \mathscr{P}_e(\omega)$

变换到时域，二者共同有： $R_r(\tau)=R_e(\tau)*R_h(\tau)$



### 5. 匹配滤波器

使有用信号 $s(t)$ 增强，抑制噪声 $n(t)$ 。信号和噪声同时进入滤波器，如果在某段时间内信号 $s(t)$ 存在，滤波器的输出在相应的瞬间出现强大的峰值。

有用信号 $s(t)$ 持续时间有限，为 $0 \sim T$ ，则匹配滤波器的冲激响应为 $h(t)=ks(t_m-t)$

考虑到系统因果可实现（$t_m\geqslant T$）+ 观察时间 $t_m$ 尽可能小（$t_m=T$） + 取系数 $k$ 为1，得到：$h(t)=s(T-t)$ 。

输出为 $s_o(t)=s(t)*h(t)=s(t)*s(T-t)=R_{ss}(t_T)$ 。

匹配滤波器相当于对 $s(t)$ 进行自相关运算，在 $t=T$ 的时刻取得自相关函数的峰值，峰值大小等于信号 $s(t)$​ 的能量 E，且仅与能量有关，与波形无关。

> 自相关函数的峰值在原点取到： $R_{ss}(0)\geq R_{ss}(\tau)$ ，并且等于信号的能量，有时差之后取值会减少。



## 第7章 离散时间系统的时域分析

### 1. 序列和离散时间系统的数学模型

离散时间信号——序列 $x(n)$ 。对应某序号 $n$​ 的函数值称为样值。

序列之间加减乘除、延时（右移）、左移、反褶、尺度倍乘（波形压缩、扩展）、差分、累加。

$x(an)$ 波形压缩， $x(\frac{x}{n})$​ 波形扩展，这时可能要按规律去除某些点或补足0值。

![2024春信号与系统20第十八讲7.1-7.7_12](D:\ASUS\Pictures\Saved Pictures\2024春信号与系统20第十八讲7.1-7.7_12.png)

![2024春信号与系统20第十八讲7.1-7.7_13](D:\ASUS\Pictures\Saved Pictures\2024春信号与系统20第十八讲7.1-7.7_13.png)

典型序列：

1. 单位样值信号： $\delta(n)$ 。
2. 单位阶跃序列： $u(n)$ 。
3. 矩形序列： $R_N(n)=u(n)-u(n-N)$  ，有 $0 \sim N-1$ 的 $N$ 个值，或 $R_N(n-m)=u(n-m)-u(n-m-N)$ ，有  $m \sim m+N-1$ 的 $N$ 个值。
4. 斜变序列： $x(n)=nu(n)$ ，显然有 $u(n)=x(n)-x(n-1)$ 。
5. 指数序列： $x(n)=a_n u(n)$ 。​
6. 正弦序列：$x(n)=\sin(n\omega_0)$ ，不一定有周期 $T$ ，但是有频率 $\omega_0$ 。

$\delta(n),\ u(n),\ nu(n)$ 仍然有差分关系： $\delta(n)=u(n)-u(n-1),\ u(n)=nu(n)-(n-1)u(n-1)$ 。

系统方框图： $1/E$ 代表单位延时。

![2024春信号与系统20第十八讲7.1-7.7_20](D:\ASUS\Pictures\Saved Pictures\2024春信号与系统20第十八讲7.1-7.7_20.png)



### 2. 常系数线性差分方程

差分方程：

阶数=未知序列变量序号的极差

前向差分方程，表现为 $x(n-\cdots),\ y(n-\cdots)$ 。

后向差分方程，表现为 $x(n+\cdots),\ y(n+\cdots)$ 。



差分方程的解法：

1. 迭代法。
2. 时域分解法，齐次解 + 特解，对应自由响应 + 强迫响应。
3. 求零输入响应 + 零状态响应，用求齐次解的方法（激励置为0）得到零输入响应，用卷积和（或边界条件全置0）求零状态响应。
4. z 变换法（下一章）。



![2024春信号与系统20第十八讲7.1-7.7_32](D:\ASUS\Pictures\Saved Pictures\2024春信号与系统20第十八讲7.1-7.7_32.png)

![2024春信号与系统20第十八讲7.1-7.7_33](D:\ASUS\Pictures\Saved Pictures\2024春信号与系统20第十八讲7.1-7.7_33.png)

![2024春信号与系统20第十八讲7.1-7.7_34](D:\ASUS\Pictures\Saved Pictures\2024春信号与系统20第十八讲7.1-7.7_34.png)



### 3. 单位样值响应、卷积和

单位样值响应 $h(n)$ ：因为只在0处去非0值可以通过迭代求出。

对于离散的 LTI 系统：因果 $\iff h(n)=h(n)u(n)$ 单边，稳定 $\iff \sum_{m=-\infty}^{\infty}|h(n)|\leqslant M$​ 绝对可和。

利用单位样值响应+卷积和求系统响应：
$$
y(n)=x(n)*h(n)=\sum_{m=-\infty}^{\infty} x(m)h(n-m)
$$
有限长序列可以快速求卷积：对位相乘求和，再不断移动。

性质：交换，分配，结合，筛选（与冲激序列卷积）。

**查表：课本 P34，表 7-1 常见序列的卷积和**

解卷积 / 反卷积：矩阵运算（这肯定不会考吧...但是第8章的课后习题 8-20 介绍了一种很简单的方法）。



## 第8章 z 变换、离散时间系统的 z 域分析

### 1. z 变换定义

抽样信号的 Laplace 变换：
$$
X_s(s)=\sum_{n=0}^{\infty} x(nT)e^{-snT}
\\
z=e^{sT},\quad s=\frac{1}{T}\ln z
\\
T=1 \implies X(z)=\sum_{n=0}^{\infty} x(n)z^{-n}
$$
也可以由洛朗级数引出（~~又到了最喜欢的复变~~）。

![2024春信号与系统22第二十讲8.1-8.5_07](D:\ASUS\Pictures\Saved Pictures\2024春信号与系统22第二十讲8.1-8.5_07.png)



单边 z 变换：
$$
X(z)=\mathscr{Z}[x(n)]=\sum_{n=0}^{\infty} x(n)z^{-n}
$$
双边 z 变换：
$$
X(z)=\mathscr{Z}[x(n)]=\sum_{n=-\infty}^{\infty} x(n)z^{-n}
$$
z 为复变量。因果信号的单边与双边 z 变换相同。

**查表：附录5 序列的 z 变换表**



### 2. z 变换的收敛域 ROC

不同序列的 z 变换可能相同。如 $a^nu(n), \ -a^nu(-n-1)$ 。

每一个 z 变换式要标注 ROC。在 ROC 内 z 变换函数解析，因此 ROC 内不会有极点。有理函数以 ROC 以极点为边界。

收敛的充分条件是绝对可和： $\sum_{n=-\infty}^{\infty} x(n)z^{-n}<\infty$ ，回忆级数收敛的判定方法——比值判定和根植判定：
$$
\lim_{n\to\infty}|\frac{a_{n+1}}{a_n}|=\rho
\\
\lim_{n\to\infty}|\sqrt[n]{|a_n|}=\rho
\\
\rho<1\ 收敛, \quad \rho>1\ 发散\, \quad \rho=1\ 不确定
$$
（课本 P53，表8-1）对于双边 z 变换，序列的 ROC：

1. 有限长序列：几乎为整个平面。

   （1）原点左右均有， $0<|z|<\infty$ 。

   （2）原点及其右边， $0<|z|$ 。

   （3）原点及其左边， $|z|<\infty$ 。​

   > 有限长序列，求和范围里有 $n>0$ 的项，ZT 中会含有 $\frac{1}{z}$ ，因此 ROC 不包含 $z=0$ 。
   >
   > 求和范围里有 $n<0$ 的项，ZT 中会含有 $z$ 的乘方，因此 ROC 不包含 $z=\infty$ 。

2. 右边序列：圆外（z 要足够大，抑制序列的增长）。

   （1）包含原点，有原点左边的部分（非因果）， $R_{x1}<|z|<\infty$ 。

   （2）原点及其右边（因果）， $R_{x1}<|z|$ 。

3. 左边序列：圆内（1/z 要足够大，抑制序列的增长）。

   （1）包含原点，有原点右边的部分， $0<|z|<R_{x2}$ 。

   （2）原点及其左边， $z<R_{x2}$ 。

4. 双边序列：圆环， $R_{x1}<|z|<R_{x2}$ 。​



![img](https://bkimg.cdn.bcebos.com/pic/f603918fa0ec08fa513d065caea62a6d55fbb3fb6cbd?x-bce-process=image/format,f_auto/resize,m_lfit,limit_1,h_1000)

![img](https://bkimg.cdn.bcebos.com/pic/6c224f4a20a4462309f7bb996f6a650e0cf3d6ca79bd?x-bce-process=image/format,f_auto/resize,m_lfit,limit_1,h_703)

![2024春信号与系统22第二十讲8.1-8.5_19](D:\ASUS\Pictures\Saved Pictures\2024春信号与系统22第二十讲8.1-8.5_19.png)



### 3. 逆 z 变换

定义式（可以救急用）：
$$
x(n)=\mathscr{Z}^{-1}[X(z)]
=\frac{1}{2\pi \mathrm{j}}\oint_C X(z)z^{n-1}\mathrm{d}z
$$

积分路径 C 是包含 $X(z)z^{n-1}$​ 所有极点的逆时针闭合曲线，通常选择 z 平面收敛域内以原点为中心的圆。



围线积分法（留数法）：坏

幂级数展开法（长除法）：坏

部分分式展开法：好
$$
X(z)=\frac{N(z)}{D(z)}=\frac{b_0+b_1z+\cdots+b_{r-1}z^{r-1}+b_rz^r}{a_0+a_1z+\cdots+a_{k-1}z^{k-1}+a_kz^k}
$$
因果序列的 z 变换 ROC 在圆外，为保证在无穷远处收敛，分母多项式阶次应不小于分子多项式。

大多数情况 $X(z)$ 下只有一阶极点，将 $\frac{X(z)}{z}$ 展开为 $\frac{z}{z-z_m}$ 的形式：
$$
\frac{X(z)}{z} = \sum_{m=0}^K \frac{A_m}{z-z_m}
\\
X(z) = \sum_{m=0}^K \frac{A_m z}{z-z_m}=A_0+\sum_{m=1}^K \frac{A_m z}{z-z_m}
\\
A_m = \Res[\frac{X(z)}{z}]_{z=z_m}=\bigg[(z-z_m)\frac{X(z)}{z}\bigg]_{z=z_m}
\\
A_0 = [X(z)]_{z=0}=\frac{b_0}{a_0}
$$
如果有高阶极点：
$$
X(z)=A_0+\sum_{m=1}^M \frac{A_m z}{z-z_m}+\sum_{j=1}^s \frac{B_j z}{(z-z_i)^j}
\\
X(z)=A_0+\sum_{m=1}^M \frac{A_m z}{z-z_m}+\sum_{j=1}^s \frac{C_j z^j}{(z-z_i)^j}
\\
C_s=\bigg[(\frac{z-z_i}{z})^sX(z)\bigg]_{z=z_i}
$$

> 部分分式展开要多练习（）



**查表：课本P61，表8-2 逆 z 变换表**



### 5. z 变换的性质

**查表：课本 P74，表8-5 z 变换的主要性质**

（1）线性。

（2）时移特性。

对于双边 z 变换，位移只会使 z 变换在 $z=0$ 或 $z=\infty$ 的零极点情况发生变化。若 $x(n)$ 为双边序列，ROC 为圆环，位移不会改变 ROC。

对于单边 z 变换，分左移和右移两情况。左移需要把将要移到左边的那部分减掉，右移需要把原来“藏”在左边的部分加上来。对因果序列也有特殊讨论。

（3）序列线性加权，z 域微分。

（4）序列指数加权，z 域尺度变换。

（5）初值定理。

（6）终值定理。要求 $n\to\infty$ 的时候序列 $x(n)$ 收敛，也就是 $X(z)$ 的极点在单位圆内，或者 $z=+1$​ 点处的一阶极点。

（7）时域卷积，z 域相乘。一般情况下新序列 ROC 为二者 ROC 的重叠部分，但是有可能 ROC 边缘发生零极点相消，使 ROC 扩大。

（8）序列相乘，z 域卷积。（不要求）

（9）尺度变换性质，这个在郑版课本、 ggg 课件 和 yz 的课件中均未提及。但是2020年期末考试中出过这样一道题： $x(2n+1)$ 的双边 z 变换。

我们可以直接推导（经过 ggg 验证）：
$$
\begin{align}
X(z)&=\sum_{n=-\infty}^{\infty}x(n)z^{-n}
\\
X(z^{\frac{1}{T}})&=\sum_{n=-\infty}^{\infty}x(n)z^{-\frac{n}{T}}
\\
&=\sum_{m=-\infty}^{\infty}x(mT)z^{-m}
\\
&=\sum_{n=-\infty}^{\infty}x(nT)z^{-n}
\\
\implies \mathscr{Z}[x(nT)]&=X(z^{\frac{1}{T}})
\end{align}
$$






### 6. z 变换与 Laplace 变换的关系

ZT 和 LT 表达式的对应：可以直接由 LT 表达式写出 ZT 表达式
$$
\frac{A_i}{s-p_i} \implies \frac{A_i}{1-e^{p_iT}z^{-1}}=\frac{A_iz}{z-e^{p_iT}}
$$
用这个可以推导出正弦序列的 z 变换。

（在某些点发生跳变似乎要单独讨论......但这似乎不重要）

![2024春信号与系统23第二十一讲8.6-8.8_04](D:\ASUS\Pictures\Saved Pictures\2024春信号与系统23第二十一讲8.6-8.8_04.png)



z 平面与 s 平面的映射关系：（T 为序列的时间间隔）
$$
z=e^{sT},\quad s=\frac{1}{T}\ln z
\\
\omega_s=\frac{2\pi}{T}
\\
s=\sigma+\mathrm{j}\omega
\\
z=re^{\mathrm{j}\theta}
\\
\implies r=e^{\sigma T}=e^{\frac{2\pi\sigma}{\omega_s}}
\\
\theta =\omega T=2\pi\frac{\omega}{\omega_s}
$$
s 平面的虚轴对应 z 平面单位圆，右半平面映射到单位圆外，左半平面一个射到单位圆内。

s 平面的实轴对应 z 平面正实轴，s 平面平行于实轴的直线对应 z 平面始于原点的射线，且通过 $\mathrm{j}\frac{k\omega_s}{2}$ 的平行于实轴的直线对应 z 平面的负实轴。

s 平面沿虚轴移动，z 平面上绕单位圆周期旋转。每平移 $\omega_s$ ，沿单位圆绕一圈，因此一个 z 值对应多个 s 值。

![2024春信号与系统23第二十一讲8.6-8.8_09](D:\ASUS\Pictures\Saved Pictures\2024春信号与系统23第二十一讲8.6-8.8_09.png)



**查表：课本 P80，表8-7 常用信号的 LT 与 ZT**



### 7. 利用 z 变换解差分方程

幂级数展开（长除法）、卷积定理（分解为相乘形式，在时域卷积）、留数法（不用）。



### 8. 离散系统的系统函数

#### 单位样值响应 $h(n)$ 与系统函数 $H(z)$

LTI 系统，单位样值响应 $h(n)$ 与系统函数 $H(z)$ 是一对 z 变换对：
$$
Y(z)=H(z)X(z)
\\
y(n)=h(n)*x(n)
\\
\implies H(z)=\mathscr{Z}[h(n)]=\sum_{n=0}^{\infty}h(n)z^{-n}
$$
可以根据系统函数的零极点分布确定单位样值响应。

展开为部分分式：
$$
\begin{align}
h(n) &= \mathscr{Z}[H(z)]=\mathscr{Z}^{-1}\bigg[ \sum_{k=0}^N\frac{A_k z}{z-p_k} \bigg]
\\
&= \mathscr{Z}^{-1}\bigg[ A_0+\sum_{k=1}^N\frac{A_k z}{z-p_k} \bigg]
\\
&= A_0\delta(n)+\sum_{k=1}^N A_k(p_k)^n u(n)
\end{align}
$$
极点 $p_k$ 一般以共轭复数形式出现。可见 $h(n)$ 的特性取决于 $H(z)$ 的极点，幅度由系数 $A_k$ 决定，而系数 $A_k$ 由 $H(z)$ 的零点分布有关。与 LT 类似， **$H(z)$ 的极点决定 $h(n)$ 的波形特征，零点只影响 $h(n)$​ 的幅度和相位**。



“大圆图”，课本 P86

![2024春信号与系统23第二十一讲8.6-8.8_36](D:\ASUS\Pictures\Saved Pictures\2024春信号与系统23第二十一讲8.6-8.8_36.png)

![2024春信号与系统23第二十一讲8.6-8.8_37](D:\ASUS\Pictures\Saved Pictures\2024春信号与系统23第二十一讲8.6-8.8_37.png)

![2024春信号与系统23第二十一讲8.6-8.8_38](D:\ASUS\Pictures\Saved Pictures\2024春信号与系统23第二十一讲8.6-8.8_38.png)



#### 从 z 域考察离散时间系统的因果性和稳定性

系统稳定的充要条件是单位样值响应 $h(n)$ 绝对可和：
$$
\sum_{n=-\infty}^{\infty}|h(n)|<M
\\
H(z)\bigg|_{z=1}=\sum_{n=-\infty}^{\infty}h(n)
$$
因此**稳定系统的系统函数 ROC 包含单位圆在内**。

系统因果的条件： $h(n)=h(n)u(n)$ ，z 变换的 ROC 为圆外且包含无穷远点。

综上，因果稳定的系统应该同时满足：
$$
\begin{cases}
a<|z|<\infty \\
a<1 \\
\end{cases}
$$
这也限制了**所有极点都在单位圆内**。

![2024春信号与系统23第二十一讲8.6-8.8_40](D:\ASUS\Pictures\Saved Pictures\2024春信号与系统23第二十一讲8.6-8.8_40.png)



### 9. 序列的傅里叶变换（DTFT）

单位圆上的 z 变换。注意与离散傅里叶变换 DFT 完全不同！

#### 定义、收敛条件

$$
\begin{align}
z &= e^{\mathrm{j}\omega}
\\
\implies \mathrm{DTFT}[x(n)] &= X(e^{\mathrm{j}\omega})=\sum_{n=-\infty}^{\infty} x(n)e^{-\mathrm{j}\omega n}
\\
\mathrm{IDTFT}[X(e^{\mathrm{j}\omega})] &= x(n)=\frac{1}{2\pi}\int_{-\pi}^{\pi}X(e^{\mathrm{j}\omega})e^{\mathrm{j}\omega n}\,\mathrm{d}\omega
\end{align}
$$

又有：
$$
X(e^{\mathrm{j}\omega})=|X(e^{\mathrm{j}\omega})|e^{\mathrm{j}\varphi(\omega)}
=\mathrm{Re}[X(e^{\mathrm{j}\omega})]+\mathrm{j}\mathrm{Im}[X(e^{\mathrm{j}\omega})]
$$
$X(e^{\mathrm{j}\omega})$ 称为序列 $x(n)$ 的频谱， $|X(e^{\mathrm{j}\omega})|$ 为幅度谱， $\varphi(\omega)$ 为相位谱。

由于 $\omega$ 沿单位圆旋转， $X(e^{\mathrm{j}\omega})$ 是以 $2\pi$​ 为周期的周期函数。

时域是离散的，频域是连续的。

DTFT 存在的充分条件：序列 $x(n)$ 绝对可和。必要条件至今未找到（ FT 也如此）

#### 基本性质

1. 线性。
2. 时域位移。
3. 频域位移。
4. 线性加权，频域微分。
5. 反褶。
6. 奇偶虚实性，参照 FT。
7. 时域卷积，频域卷积。
8. 帕塞瓦尔定理：能量守恒。
9. 共轭： $x^*(n)\iff X^*(e^{-\mathrm{j}\omega})$ 。因此对于实函数： $X(e^{\mathrm{j}\omega})=X^*(e^{-\mathrm{j}\omega})$

![第4章离散信号傅里叶分析_21](D:\ASUS\Pictures\Saved Pictures\第4章离散信号傅里叶分析_21.png)



### 10. 离散时间系统的频率响应

由系统函数 $H(z)$ 到频率响应 $H(e^{\mathrm{j}\omega})$ ，回忆连续时间系统的系统函数 $H(s)$ 到频率响应 $H(j\omega)$ 。

连续时间系统的特征函数是 $e^{st}=e^{\mathrm{j}\omega t}$ ，输入信号为 $e^{\mathrm{j}\omega t}$ 的情况下输出信号为：
$$
x(t)=e^{\mathrm{j}\omega t} \implies y(t)=|H(\mathrm{j}\omega)|e^{\mathrm{j}(\omega t+\varphi)}
\\
x(t)=\sin(\omega t) \implies y(t)=|H(\mathrm{j}\omega)|\sin(\omega t+\varphi)
\\
x(t)=\cos(\omega t) \implies y(t)=|H(\mathrm{j}\omega)|\cos(\omega t+\varphi)
$$


离散时间系统的频响函数为：
$$
H(e^{\mathrm{j}\omega})=\sum_{n=-\infty}^{\infty} h(n)e^{-\mathrm{j}\omega n}
$$
离散时间系统特征函数为： $z^{n}=e^{\mathrm{j}\omega n}$ ，对复指数序列 / 正弦序列激励的**稳态响应**为：
$$
x(n)=e^{\mathrm{j}\omega n} \implies y_{ss}(n)=|H(e^{\mathrm{j}\omega})|e^{\mathrm{j}(\omega n+\varphi)}
\\
x(n)=\sin(n\omega) \implies y_{ss}(n)=|H(e^{\mathrm{j}\omega})|\sin(n\omega+\varphi)
$$

> 离散信号中频率 $\omega$ 和 $\omega+2k\pi$ 不可区分： $\sin((\omega+2k\pi)n+\varphi)=\sin(\omega n+\varphi)$ 。

频率响应 $H(e^{\mathrm{j}\omega})$ 和单位样值响应 $h(n)$​ 是一对 Fourier 变换对。

频率响应 $H(e^{\mathrm{j}\omega})$ 为周期函数，周期为 $\omega_s=\frac{2\pi}{T}$ 。

判定频率响应特性，只需关注一个周期 $(0,\omega_s)$ 内的情况。

如果为实系数， $|H(e^{\mathrm{j}\omega})|$ 为偶函数， $\varphi(\omega)$ 为奇函数，也只需要关注半个周期 $(0,\omega_s/2)$ 内的情况。0和 $\omega_s$ 是最低频， $\omega_s/2$ 是最高频，以此来判断

低通/高通/带通/带阻/全通的系统特性。

![2024春信号与系统24第二十二讲8.9-8.11_14](D:\ASUS\Pictures\Saved Pictures\2024春信号与系统24第二十二讲8.9-8.11_14.png)

![2024春信号与系统24第二十二讲8.9-8.11_15](D:\ASUS\Pictures\Saved Pictures\2024春信号与系统24第二十二讲8.9-8.11_15.png)

![第4章离散信号傅里叶分析_04](D:\ASUS\Pictures\Saved Pictures\第4章离散信号傅里叶分析_04.png)



频率响应的几何确定法：

画图，长度乘除决定幅度特性，夹角加减决定相位特性。

幅度响应靠近极点处出现峰点，靠近零点处出现谷点。 $z=0$​ 处的零极点只影响相位响应，不影响幅度响应。

![2024春信号与系统24第二十二讲8.9-8.11_19](D:\ASUS\Pictures\Saved Pictures\2024春信号与系统24第二十二讲8.9-8.11_19.png)



### 11. z 变换应用实例

数字式自激振荡器：
$$
h(n)=\cos(n\omega)u(n)
\\
h(n)=\sin(n\omega)u(n)
$$
结构上改进：使用中间信号 $W(z)$ 实现结构复用，系统可同时产生 sin 和 cos 信号。

![2024春信号与系统24第二十二讲8.9-8.11_51](D:\ASUS\Pictures\Saved Pictures\2024春信号与系统24第二十二讲8.9-8.11_51.png)



数字滤波器：

原理：输入的连续信号 $x(t)$ 频带受限 $-\omega_m\sim\omega_m$ ，抽样间隔满足奈奎斯特抽样频率 $\omega_s=\frac{2\pi}{T}\geqslant2\omega_m$ 。

![2024春信号与系统24第二十二讲8.9-8.11_28](D:\ASUS\Pictures\Saved Pictures\2024春信号与系统24第二十二讲8.9-8.11_28.png)



**冲激不变法**设计数字滤波器（低通）：

根据 ZT 和 LT 的关系，由模拟域到数字域，直接改写式子：
$$
H(s)=\mathscr{L}[h(t)]=\sum \frac{A_i}{s-p_i}
\\
\Downarrow
\\
H(z)=\mathscr{Z}[h(n)]=\sum \frac{A_i}{1-e^{p_iT}z^{-1}}
$$
要求频率响应 $H(\mathrm{j}\omega)$ 在 $0\sim \omega_s/2$​ 内衰减足够快。

优点：简单，便于与模拟滤波器直接对应。

缺点：s 与 z 的多值对应关系可能引起混叠，不能用于设计高通和带阻滤波器。



## 第11章 反馈系统

闭环增益表达式，负反馈 $\frac{A}{1+AF}$ ，正反馈 $\frac{A}{1-AF}$ 。

深度负反馈 $AF\gg1\implies H\approx\frac{1}{F}$，环路增益仅由反馈系数决定。

负反馈的作用：

1. 改善系统灵敏度。增益波动变小。
2. 改善系统频响特性。可以使扩展带宽，降低增益，增益带宽积不变（~~又到了最喜欢的电电~~）。
3. 逆系统设计。直接将原系统组成反馈系统： $H_i(s)=\frac{1}{H(s)}$ 。
4. 不稳定系统变稳定。可以把极点从右半平面移到左半平面。PID算法，比例、微分、积分反馈（~~又到了最喜欢的硬设~~）。

正反馈自激振荡， $A(s)F(s)=1$ （模值为1，辐角2pi），临界稳定。

信号流图

流图转置

梅森公式

![2024春信号与系统25第二十三讲11.1,11.6_45](D:\ASUS\Pictures\Saved Pictures\2024春信号与系统25第二十三讲11.1,11.6_45.png)



## 第12章 系统的状态变量分析

状态 / 状态变量 / 状态矢量：一个动态系统的状态是表示系统的一组最少变量，只需知道 $t=t_0$ 时刻这组变量和 $t\geqslant t_0$ 时刻以后的输入， 就能确定系统在 $t\geqslant t_0$​ 时刻以后的行为。

> 引入状态变量的目的：相当于使用中间变量表示输入输出，可以把一元 N 阶方程转换为 N 元一阶方程，每一阶都用状态变量表示，相邻阶的中间变量之间是一阶关系。

算子 $p$ 是微分运算，算子 $1/p$​ 是积分运算。算子表达式就是关于积分和微分环节的组合。



### 1. 连续时间系统状态方程的建立

状态方程与输出方程分别为：
$$
\dot{\boldsymbol{\lambda}}(t)=\boldsymbol{A\lambda}(t)+\boldsymbol{Be}(t)
\\
\boldsymbol{r}(t)=\boldsymbol{C\lambda}(t)+\boldsymbol{De}(t)
$$
对于 LTI 系统 ABCD 矩阵是常数，而对于时变系统 ABCD 矩阵是时间的函数。

（看典型结构示意图）

由电路图建立（电路课重点，非本课重点）。

由系统输入输出方程或信号流图建立状态方程。对于与给定的系统，流图的形式可以不同，状态变量的选择不唯一， ABCD 矩阵也不唯一。

![2024春信号与系统26第二十四讲12.1-12.3_15](D:\ASUS\Pictures\Saved Pictures\2024春信号与系统26第二十四讲12.1-12.3_15.png)



由算子表达式分解或系统函数建立状态方程。部分分式展开 $H(p)=\sum\frac{\beta_i}{p+\alpha_i}$​ ，由基本单元串联、并联、级联组装。

基本单元：

![2024春信号与系统26第二十四讲12.1-12.3_20](D:\ASUS\Pictures\Saved Pictures\2024春信号与系统26第二十四讲12.1-12.3_20.png)



### 2. 连续时间系统状态方程的求解

#### Laplace 变换解法（较为容易）

写不动了，直接上图：

![2024春信号与系统26第二十四讲12.1-12.3_27](D:\ASUS\Pictures\Saved Pictures\2024春信号与系统26第二十四讲12.1-12.3_27.png)

![2024春信号与系统26第二十四讲12.1-12.3_28](D:\ASUS\Pictures\Saved Pictures\2024春信号与系统26第二十四讲12.1-12.3_28.png)



#### 时域解法

写不动了，直接上图：

![2024春信号与系统26第二十四讲12.1-12.3_34](D:\ASUS\Pictures\Saved Pictures\2024春信号与系统26第二十四讲12.1-12.3_34.png)



### 3. 根据状态方程求转移函数

写不动了，直接上图：

![2024春信号与系统26第二十四讲12.1-12.3_36](D:\ASUS\Pictures\Saved Pictures\2024春信号与系统26第二十四讲12.1-12.3_36.png)





### 4. 离散时间系统状态方程的建立

同连续时间系统的形式，用差分代替微分。
$$
\boldsymbol{\lambda}(n+1)=\boldsymbol{A\lambda}(n)+\boldsymbol{Bx}(n)
\\
\boldsymbol{y}(n)=\boldsymbol{C\lambda}(n)+\boldsymbol{Dx}(n)
$$
（看典型结构示意图）

由定义建立。

由框图或流图建立。



### 5. 离散时间系统状态方程的求解

#### 时域迭代法求解

写不动了，直接上图：

![2024春信号与系统27第二十五讲12.4-12.7_14](D:\ASUS\Pictures\Saved Pictures\2024春信号与系统27第二十五讲12.4-12.7_14.png)

#### z 变换求解

写不动了，直接上图：

![2024春信号与系统27第二十五讲12.4-12.7_16](D:\ASUS\Pictures\Saved Pictures\2024春信号与系统27第二十五讲12.4-12.7_16.png)

![2024春信号与系统27第二十五讲12.4-12.7_17](D:\ASUS\Pictures\Saved Pictures\2024春信号与系统27第二十五讲12.4-12.7_17.png)



### 6. 状态矢量的线性变换

选择不同的状态矢量可以得到不同的 ABCD 矩阵，各状态矢量之间存在某种约束，矩阵 ABCD 之间存在某种变换关系。

具体细节略。



判断系统的稳定性：

![2024春信号与系统27第二十五讲12.4-12.7_28](D:\ASUS\Pictures\Saved Pictures\2024春信号与系统27第二十五讲12.4-12.7_28.png)



### 7. 系统的可控性和可观性

可控性 (Controllability)：给定起始状态，可以找到容许的输入量 (控制矢量)，在有限时间内把系统的所有状态引向零状态。如果可做到这点，则称系统完全可控。

可观性 (Observability)：给定输入 (控制) 后，能在有限时间内根据系统输出唯一地确定系统的起始状态。如果可做到这点，则称系统完全可观。



判别方法：

利用可控阵和可观阵判定：

![2024春信号与系统27第二十五讲12.4-12.7_37](D:\ASUS\Pictures\Saved Pictures\2024春信号与系统27第二十五讲12.4-12.7_37.png)

A 矩阵规范化之后判别。（略）



可控可观与转移函数的关系：

![2024春信号与系统27第二十五讲12.4-12.7_44](D:\ASUS\Pictures\Saved Pictures\2024春信号与系统27第二十五讲12.4-12.7_44.png)

留意一下串联、并联、级联可能发生零极点相消，导致系统不可控不稳定就行。



## 可能的考点分析和习题选讲

抽样信号分析。

相关函数计算。

能谱和功率谱。

匹配滤波器。

离散系统的特性分析：线性，时不变，因果，稳定。

卷积和，时域经典法解差分方程。

z 变换（单边和双边）。

逆 z 变换（除了常规方法还可以级数展开），注意收敛域和右边序列 / 左边序列 / 双边序列、单边 z 变换 / 双边 z 变换。（最好的分解方式不一定是最彻底的分解方式，要根据表格选择合适的分解形式）。

z 变换性质的应用。

单边 z 变换求解差分方程，注意边界条件。几种不同组合：齐次解 + 特解，零输入 + 零状态，瞬态响应（足够长时间后为衰减为0） + 稳态响应（常数值）。

求系统函数和单位样值响应。

根据系统结构框图写差分方程，或根据差分方程画系统结构框图。注意延时单元的含义。

冲激不变法设计数字滤波器。（形式化）

分析系统的特性和频率响应，系统函数的零极点分布图、收敛域、幅频响应曲线和相频响应曲线。

信号流图的化简。根据流图列写系统转移函数（梅森公式），或按照转移函数画流图。

状态方程与输出方程列写，判断系统的可控性和可观性。



对于一个离散的 LTI 系统，如果因果稳定，显然所有极点位于单位圆之内，

如果所有零点也位于单位圆之内，称为最小相移系统；所有零点位于单位圆之外，称为最大相移系统。

类似与连续情况下全通系统的零极点关于虚轴对称分布，离散情况下全通系统的零点与极点关于单位圆“对称”分布，极点位于单位圆内，零点位于单位圆外，这里的对称指的是**零点对应矢量模长与极点对应矢量模长之积为1**，比如零点 $re^{\mathrm{j}\theta}\implies$ 极点 $\frac{1}{r}e^{\mathrm{j}\theta}$​ 。

因此我们可以进行如下转换（与极点相关的分母我们不关心）：
$$
H(z)=A\frac{z^2+az+b}{\cdots}\implies
A\frac{z^2+\frac{1}{a}z+\frac{1}{b}}{\cdots}
$$



目录：


[TOC]
