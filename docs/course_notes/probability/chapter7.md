# 7 泊松过程

??? abstract "写在前面：人生就是一个泊松过程。"
    人生就是等待的艺术，泊松过程就是一个等待的艺术。我们需要用非常理性的观点看待这件事情，因为你只有充分地等待，你才能迎来时机，你只有主动地等待，你才能抓住时机。你要想跃升一定是要抓住机会的，而你为什么能抓得住，并不是天上的馅饼会掉到你脑袋上，你需要非常主动地等待、非常积极地去等待。恰如咱们同学现在所做的事情——你现在为什么坐在这里，你不就是为了等考试吗？等考试有两种等法：一种是坐以待毙，“所有的考试都来吧”；另一种是有目的、有意识、有方法、有手段地去等待考试，我相信我一定能够在开始中发挥出我的水平，能够用考试去检验我这一个学期学习的效果，我能够通过考试来证明“我应该是一个爷们”，我能够去克服困难，我可以在电子系四年学习当中最忙的一个学期、最多课程的情况下，仍然取得让我能够满意、让我能够对得起天对得起地、对得起父母对得起乡亲的成绩。无论你未来是要继续深造，还是走向工作岗位，总之当你完成下一次跃升之后，一定又是一段或长或短时间的等待。把握住这段等待，你才能更有机会获得下一次跃升。这就是我们所说的“人生就是一个泊松过程”。以后大家区别于没有学过《随机过程》的人，你对人生的感悟一定比他高，因为你知道什么是泊松过程。[^1]

## 齐次 Poisson 过程

如果一个计数过程 $\{N(t)\}$ 满足：

1. 在 $t=0$ 时刻计数归零，即 $N(0)=0$ .
2. $\{N(t)\}$ 是 独立增量过程。
3. $\{N(t)\}$ 是 平稳增量过程。由此可推出 $P(N(t_0+t)-N(t_0)=n)=P(N(t)=n)$ .
4. 稀疏性： $P(N(t+\Delta t)-N(t)-1)=\lambda\Delta t+o(\Delta t),\;P(N(t+\Delta t)-N(t)\geqslant 2)=o(t)$ ，即一小短时间内事件几乎不可能发生多次。

称该过程为 齐次泊松过程/泊松过程， $\lambda>0$ 为 **强度参数**。

!!! note
    回顾满足泊松分布的随机变量 $X\sim\text{Poisson}(\lambda)$ ，概率分布为 $P(X=k)=\dfrac{\lambda^k}{k!}\mathrm{e}^{-\lambda}$ .

泊松过程的概率分布为 $P(N(t)=n)=\dfrac{(\lambda t)^n}{n!}\mathrm{e}^{-\lambda t},\quad t\geqslant 0,\;n=0,1,2,\cdots$ 。满足归一化条件 $\displaystyle\sum_{n=0}^{+\infty}P(N(t)=n)=1$ .

如果固定时间 $t$ ，则随机变量 $N(t)$ 代表 $0\sim t$ 时间内计数次数，且满足泊松分布 $\sim\text{Poisson}(\lambda t)$ .

<br>

泊松过程的 数字特征：

$$
\begin{align*}
\mathrm{E}(N(t))&=\lambda t, \quad \mathrm{Var}(N(t))=\lambda t \\
R_N(t_1,t_2)&=\mathrm{E}(N(t_1)N(t_2))=\lambda^2t_1t_2+\lambda\min(t_1,t_2) \\
C_N(t_1,t_2)&=\lambda\min(t_1,t_2)
\end{align*}
$$

由上式也可以看出参数 $\lambda=\dfrac{\mathrm{E}(N(t))}{t}$ 表示单位时间内事件发生的平均次数，即代表了 强度/速率。

泊松过程的 特征函数和矩母函数分别为：

$$
\begin{align*}
\phi_{N(t)}(\omega)&=\mathrm{E}\left(\mathrm{e}^{\mathrm{j}\omega N(t)}\right)=\exp[\lambda t(\mathrm{e}^{\mathrm{j}\omega}-1)] \\
G_{N(t)}(z)&=\mathrm{E}\left(z^{N(t)}\right)=\exp[\lambda t(z-1)]
\end{align*}
$$

!!! Tip
    泊松过程的 **矩母函数** 非常重要，不同类型泊松过程的矩母函数形式不同，可作为我们判断泊松过程类型的依据。

### 事件时间问题

将泊松过程计数事件的 **发生时刻** 记为 $S_n,\;n=0,1,2,\cdots$ ，且 $S_0=0$ .

相邻事件发生的间隔为 $T_n=S_n-S_{n-1},\;n=1,2,\cdots$ .

根据关系 $P(S_n\leqslant t)=P(N(t)\geqslant n)$ ，得到 $S_n$ 的累积分布函数(CDF)为 $F_{S_n}=P(S_n<t)=P(N(t)\geqslant n)=\displaystyle\sum_{k=n}^{+\infty}\frac{(\lambda t)^k}{k!}\mathrm{e}^{-\lambda t}$ ，求导得到 $f_{S_n}(t)=\dfrac{\mathrm{d}}{\mathrm{d}t}F_{S_n}(t)=\dfrac{(\lambda t)^{n-1}}{(n-1)!}\lambda\mathrm{e}^{-\lambda t}$ ，服从 $n$ 阶 $\Gamma$ 分布。当 $n=1$ 时退化为 指数分布 $S_1\sim\exp(\lambda)$ . 当 $n$ 较大时接近高斯分布。

数字特征为：

$$
\mathrm{E}(S_n)=\frac{n}{\lambda},\quad \mathrm{Var}(S_n)=\frac{n}{\lambda^2}
$$

$T_i$ 的概率分布为 $f_{T_i}(t)=\lambda\mathrm{e}^{-\lambda t},\;F_{T_i}(t)=1-\mathrm{e}^{-\lambda t},\;i=1,2,\cdots,n$ .

$$
\mathrm{E}(T_i)=\frac{n}{\lambda},\quad \mathrm{Var}(T_i)=\dfrac{n}{\lambda^2}
$$

各事件发生事件间隔 $T_n$ 是 **独立同分布** 的指数分布 $T_n\sim\exp(\lambda)$ ，意味着每件事发生相隔平均时间为 $\dfrac{1}{\lambda}$ ，也说明了 $\lambda$ 代表强度参数的含义。这也是“泊松过程是独立增量过程”的体现。另外，由于指数分布具有 **无记忆性**，从某一时刻开始计时，到下一次时间发生的时间间隔，与之前事件的发生时刻无关。例如：到公交站等待下一辆车到达的事件，与之前的车的到达时间无关。

---

如果在已知 $[0,t]$ 内泊松过程发生了 $n$ 次的**条件**下，考虑等待时间 $S_1,\cdot,S_n$ 的**条件分布** $S_1,\cdots,S_n|N(t)=n$ 的概率密度为：

$$
f_{S_1,\cdots,S_n}(u_1,\cdots,u_n|N(t)=n)=\frac{n!}{t^n},\quad 0\leqslant u_1<\cdots<u_n\leqslant t
$$

可见为**多元的均匀分布**。

如果不考虑事件发生的先后次序，记事件发生的事件为 $\{V_1,\cdots,V_n\}$ ，称 $\{S_1,\cdots,S_n\}$ 为 $\{V_1,\cdots,V_n\}$ 的**顺序统计量**，其概率密度为：

$$
f_{V_1,\cdots,V_n}(v_1,\cdots,v_n|N(t)=n)=\frac{1}{t^n},\quad 0\leqslant v_i\leqslant t,\;i=1,\cdots,n
$$

可见，由于 $\{S_1,\cdots,S_n\}$ 为 $\{V_1,\cdots,V_n\}$ 的一个特殊排列，因此 $\{S_1,\cdots,S_n\}$ 的概率密度为 $\{V_1,\cdots,V_n\}$ 的 $n!$ 倍。

---

若有两个独立的泊松过程 $A\sim\text{Poisson}(\mu t),\;B\sim\text{Poisson}(\lambda t)$ ，则 $A$ 相邻事件的发生间隔之内， $B$ 过程事件的发生次数服从 **几何分布**。在 $A$ 过程发生的相邻时间 $T$ 内， $B$ 发生 $L$ 次的概率为：

$$
\begin{gather*}
P(L=k)=\int_{0}^{+\infty}[P(L=k|T=t)f_T(t)]\mathrm{d}t=\left(\frac{\mu}{\mu+\lambda}\right)\left(\frac{\lambda}{\mu+\lambda}\right)^k,\quad k=0,1,2,\cdots \\
\mathrm{E}(L)=\frac{\mu+\lambda}{\mu}
\end{gather*}
$$

## 非齐次 Poisson 过程

独立增量，但不是平稳增量。**非平稳增量意味着一段时间内事件发生的次数与这段时间的起止时刻有关**。

强度参数不是常数，而是**关于时间** $t$ **的函数** $\lambda\to\lambda(t)$ .

$$
P(N(t_0+t)-N(t_0)=n)=\frac{\left(\int_{t_0}^{t_0+t}\lambda(u)\mathrm{d}u\right)^n}{n!}\exp\left(-\int_{t_0}^{t_0+t}\lambda(u)\mathrm{d}u\right)
$$

或者写为 $N(t_0+t)-N(t_0)\sim\text{Poisson}\left(\int_{t_0}^{t_0+t}\lambda(u)\mathrm{d}u\right)\sim\text{Poisson}(\bar{\lambda}t)$ .

可见，非齐次泊松过程使用 $[t_0,t_0+t]$ 时间段内的积分 $\int_{t_0}^{t_0+t}\lambda(u)\mathrm{d}u$ 替代了齐次泊松过程的线性表达式 $\lambda t$ . 我们仍然可以使用这段时间内的平均强度参数来刻画 $\bar{\lambda}=\dfrac{\int_{t_0}^{t_0+t}\lambda(u)\mathrm{d}u}{t}$ . 如果 $\lambda(t)\equiv\lambda$ ，则显然 非齐次泊松过程 退化为 齐次泊松过程。

均值、方差、矩母函数分别为：

$$
\begin{align*}
\mathrm{E}(N(t_0+t)-N(t_0))&=\int_{t_0}^{t_0+t}\lambda(u)\mathrm{d}u=\bar{\lambda}t \\
\mathrm{Var}(N(t_0+t)-N(t_0))&=\int_{t_0}^{t_0+t}\lambda(u)\mathrm{d}u=\bar{\lambda}t \\
G_{N(t)}(z)&=\exp\left[\left(\int_{t_0}^{t_0+t}\lambda(u)\mathrm{d}u\right)t(z-1)\right]=\exp(\bar{\lambda}t(z-1))
\end{align*}
$$

## 复合 Poisson 过程

不满足事件发生的稀疏性。

$N(t)\sim\text{Poisson}(\lambda t)$ 为齐次泊松过程， $\{Y_n\}$ 为独立同分布的一组随机变量，则 $X(t)=\displaystyle\sum_{n=1}^{N(t)}Y_n$ 为复合泊松过程。

均值、方差、矩母函数分别为：

$$
\begin{align*}
\mathrm{E}(X(t))&=\lambda t\cdot\mathrm{E}(Y_n) \\
\mathrm{Var}(X(t))&=\lambda t\cdot \mathrm{E}^2(Y_n) \\
G_{X(t)}(z)&=\exp[\lambda t(G_Y(z)-1)]
\end{align*}$$

## 随机参数 Poisson 过程

平稳增量，但不是独立增量。

泊松过程的强度参数为 非负的连续型的**随机变量** $\Lambda$ ，其概率密度函数为 $f(\Lambda)$ ，随机参数泊松过程的概率密度为：

$$
P(N(t)=n)=\int_{0}^{+\infty}\frac{(\lambda t)^n}{n!}\mathrm{e}^{-\lambda t}f(\lambda)\mathrm{d}\lambda
$$

矩母函数为：

$$
G_{N(t)}(z)=\mathrm{E}\left(\mathrm{e}^{\mathrm{\Lambda t(z-1)}}\right)=\int_{0}^{+\infty}\mathrm{e}^{\lambda t(z-1)}f(\lambda)\mathrm{d}\lambda
$$

均值为：

$$
\mathrm{E}(N(t))=\frac{\mathrm{d}}{\mathrm{d}z}G_{N(t)}(z)\Big|_{z=1}=\int_{0}^{+\infty}\lambda tf(\lambda)\mathrm{d}\lambda=\mathrm{E}(\Lambda)t
$$

## 过滤 Poisson 过程

不是平稳增量，也不是独立增量。

在复合泊松过程的基础上考虑时间的影响，每个事件发生的贡献会随时间推移而发生变化。设齐次泊松过程 $N(t)\sim\text{Poisson}(\lambda t)$ 各个事件的发生时间为 $S_k,\;k=1,2,\cdots$ ，用 $X_k(t,S_k)$ 表示在 $t$ 时刻第 $k$ 个事件的贡献值，假设所有 $X_k$ 都是 **独立同分布的**，即 **每个事件的贡献随时间的变化相同**（方便讨论和分析），可将 $X(t,S_k)$ 视为一个随机过程。令 $Y(t)=\displaystyle\sum_{k=1}^{N(t)}X(t,S_k)$ ，称为过滤泊松过程。

均值和方差为：

$$
\mathrm{E}(Y(t))=\lambda\int_{0}^t\mathrm{E}(X(t,s))\mathrm{d}s,\quad \mathrm{Var}(Y(t))=\lambda\int_{0}^t\mathrm{E}^2(X(t,s))\mathrm{d}s
$$

（1）若 $X(t,s)$ 为 **连续型随机过程**，则令 $X(t)$ 的 **特征函数** 为 $B_{X(t,s)}(\omega)=\mathrm{E}[\exp(\mathrm{j}\omega X(t,s))]$ ，则 $Y(t)$ 的 **特征函数** 为：

$$
\phi_{Y(t)}(\omega)=\exp\left[\lambda\int_0^t(B_{X(t,s)}(\omega)-1)\mathrm{d}s\right]
$$

（2）若 $X(t,s)$ 为 **离散型随机过程**，则令 $X(t)$ 的 **矩母函数** 为 $B_X(t,s)(z)=\mathrm{E}\left(z^{X(t,s)}\right)$ ，则 $Y(t)$ 的 **矩母函数** 为：

$$
G_{Y(t)}(z)=\exp\left[\lambda\int_0^t(B_{X(t,s)}(z)-1)\mathrm{d}s\right]
$$

## 补充

独立的泊松分布随机变量具有可加性： $X_i\sim\text{Poisson}(\lambda_i)\implies \sum X_i\sim\text{Poisson}\left(\sum\lambda_i\right)$ .

同样，独立的泊松过程也具有可加性：$X_i(t)\sim\text{Poisson}(\lambda_i t)\implies \sum X_i(t)\sim\text{Poisson}\left(\sum\lambda_i t\right)$ .

两个独立泊松过程的 **差值** 为 复合泊松过程， $N(t)=N_1(t)-N_2(t)$ ：

$$
\begin{align*}
G_{N(t)}(z)&=\mathrm{E}\left(z^{N_1(t)-N_2(t)}\right)=\mathrm{E}\left(z^{N_1(t)}\right)\mathrm{E}\left(z^{-N_2(t)}\right) \\
&=G_{N_1(t)}(z)G_{N_2(t)}(z^{-1}) \\
&=\exp(\lambda_1 t(z-1))\exp(\lambda_2 t(z^{-1}-1)) \\
&=\exp\left( (\lambda_1+\lambda_2)t\left( \frac{\lambda_1 z+\lambda_2 z^{-1}}{\lambda_1+\lambda_2}-1 \right) \right) \\
&=\exp[(\lambda_1+\lambda_2)t(G_Y(z)-1)]
\end{align*}
$$

可令 $N(t)=\displaystyle\sum_{n=1}^{M(t)}Y_n,\;M(t)\sim\text{Poisson}((\lambda_1+\lambda_2)t)$ ， $Y_n\sim\begin{pmatrix}1 & -1 \\ \frac{\lambda_1}{\lambda_1+\lambda_2} & \frac{\lambda_2}{\lambda_1+\lambda_2}\end{pmatrix}$ 为 **独立同分布的伯努利分布**，其矩母函数恰好为 $G_Y(z)=\mathrm{E}(z^{Y})=\dfrac{\lambda_1 z+\lambda_2 z^{-1}}{\lambda_1+\lambda_2}$ .

[^1]: 这是 2024年12月24日星期二下午 16 时 52 分左右，清华大学第六教学楼 6A016 教室内电子系本科生核心课《概率论与随机过程（2）》结课时，授课教师张颢在下课前的课程总结和对同学们的勉励。
