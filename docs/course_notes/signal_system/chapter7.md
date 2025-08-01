# 7 离散时间系统的时域分析

## 7.1 序列和离散时间系统的数学模型

离散时间信号——序列 $x(n)$ . 对应某序号 $n$​ 的函数值称为样值。

序列之间加减乘除、延时（右移）、左移、反褶、尺度倍乘（波形压缩、扩展）、差分、累加。

$x(an)$ 波形压缩， $x\left(\dfrac{x}{n}\right)$​ 波形扩展，这时可能要按规律去除某些点或补足0值。

![2024春信号与系统20第十八讲7.1-7.7_12](https://cdn.jsdelivr.net/gh/DerrickMarcus/picgo_image/images/ch7_img1.png)

![2024春信号与系统20第十八讲7.1-7.7_13](https://cdn.jsdelivr.net/gh/DerrickMarcus/picgo_image/images/ch7_img2.png)

典型序列：

1. 单位样值信号： $\delta(n)$ .
2. 单位阶跃序列： $u(n)$ .
3. 矩形序列： $R_N(n)=u(n)-u(n-N)$ ，有 $0 \sim N-1$ 的 $N$ 个值，或 $R_N(n-m)=u(n-m)-u(n-m-N)$ ，有 $m \sim m+N-1$ 的 $N$ 个值。
4. 斜变序列： $x(n)=nu(n)$ ，显然有 $u(n)=x(n)-x(n-1)$ .
5. 指数序列： $x(n)=a_n u(n)$ . ​
6. 正弦序列：$x(n)=\sin(n\omega_0)$ ，不一定有周期 $T$ ，但是有频率 $\omega_0$ .

与连续信号类似， $\delta(n),\ u(n),\ nu(n)$ 仍然有差分关系： $\delta(n)=u(n)-u(n-1),\ u(n)=nu(n)-(n-1)u(n-1)$ .

系统方框图中 $\dfrac{1}{E}$ 代表单位延时。

![2024春信号与系统20第十八讲7.1-7.7_20](https://cdn.jsdelivr.net/gh/DerrickMarcus/picgo_image/images/ch7_img3.png)

## 7.2 常系数线性差分方程

差分方程：

阶数=未知序列变量序号的极差。

前向差分方程，表现为 $x(n-\cdots),\ y(n-\cdots)$ .

后向差分方程，表现为 $x(n+\cdots),\ y(n+\cdots)$ .

差分方程的解法：

1. 迭代法。
2. 时域分解法，齐次解 + 特解，对应自由响应 + 强迫响应。
3. 求零输入响应 + 零状态响应，用求齐次解的方法（激励置为0）得到零输入响应，用卷积和（或边界条件全置0）求零状态响应。
4. $z$ 变换法（下一章）。

![2024春信号与系统20第十八讲7.1-7.7_32](https://cdn.jsdelivr.net/gh/DerrickMarcus/picgo_image/images/ch7_img4.png)

![2024春信号与系统20第十八讲7.1-7.7_33](https://cdn.jsdelivr.net/gh/DerrickMarcus/picgo_image/images/ch7_img5.png)

![2024春信号与系统20第十八讲7.1-7.7_34](https://cdn.jsdelivr.net/gh/DerrickMarcus/picgo_image/images/ch7_img6.png)

## 7.3 单位样值响应、卷积和

单位样值响应 $h(n)$ ：因为只在0处去非0值可以通过迭代求出。

对于离散的 LTI 系统：（1）因果 $\iff h(n)=h(n)u(n)$ 单边；（2）稳定 $\iff \displaystyle\sum_{m=-\infty}^{\infty}|h(n)|\leqslant M$​ 绝对可和。

利用单位样值响应+卷积和求系统响应：

$$
y(n)=x(n)*h(n)=\sum_{m=-\infty}^{\infty} x(m)h(n-m)
$$

有限长序列可以快速求卷积：对位相乘求和，再不断移动。

性质：交换，分配，结合，筛选（与冲激序列卷积）。

**查表：课本 P34，表 7-1 常见序列的卷积和。**

解卷积/反卷积：矩阵运算，另外第8章的课后习题 8-20 介绍了一种简单方法。
