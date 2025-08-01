# 12 系统的状态变量分析

状态 / 状态变量 / 状态矢量：一个动态系统的状态是表示系统的一组最少变量，只需知道 $t=t_0$ 时刻这组变量和 $t\geqslant t_0$ 时刻以后的输入， 就能确定系统在 $t\geqslant t_0$​ 时刻以后的行为。

引入状态变量的目的：相当于使用中间变量表示输入输出，可以把一元 $N$ 阶方程转换为 $N$ 元一阶方程，每一阶都用状态变量表示，相邻阶的中间变量之间是一阶关系。

算子 $p$ 是微分运算，算子 $\dfrac{1}{p}$​ 是积分运算。算子表达式就是关于积分和微分环节的组合。

## 12.1 连续时间系统状态方程的建立

状态方程与输出方程分别为：

$$
\begin{align*}
\dot{\boldsymbol{\lambda}}(t)&=\boldsymbol{A\lambda}(t)+\boldsymbol{Be}(t) \\
\boldsymbol{r}(t)&=\boldsymbol{C\lambda}(t)+\boldsymbol{De}(t)
\end{align*}
$$

对于 LTI 系统， $\boldsymbol{A},\boldsymbol{B},\boldsymbol{C},\boldsymbol{D}$ 矩阵是常数；对于时变系统 $\boldsymbol{A},\boldsymbol{B},\boldsymbol{C},\boldsymbol{D}$ 矩阵是时间的函数。

<br>

状态方程的建立方法：

（1）由电路图建立（电路课重点，非本课重点）。

（2）由系统输入输出方程或信号流图建立状态方程。对于与给定的系统，流图的形式可以不同，状态变量的选择不唯一， $\boldsymbol{A},\boldsymbol{B},\boldsymbol{C},\boldsymbol{D}$ 矩阵也不唯一。

![2024春信号与系统26第二十四讲12.1-12.3_15](https://cdn.jsdelivr.net/gh/DerrickMarcus/picgo_image/images/ch12_img1.png)

由算子表达式分解或系统函数建立状态方程。部分分式展开 $H(p)=\displaystyle\sum\dfrac{\beta_i}{p+\alpha_i}$​ ，由基本单元串联、并联、级联组装。

基本单元：

![2024春信号与系统26第二十四讲12.1-12.3_20](https://cdn.jsdelivr.net/gh/DerrickMarcus/picgo_image/images/ch12_img2.png)

## 12.2 连续时间系统状态方程的求解

### Laplace 变换解法（较为容易）

![2024春信号与系统26第二十四讲12.1-12.3_27](https://cdn.jsdelivr.net/gh/DerrickMarcus/picgo_image/images/ch12_img3.png)

![2024春信号与系统26第二十四讲12.1-12.3_28](https://cdn.jsdelivr.net/gh/DerrickMarcus/picgo_image/images/ch12_img4.png)

### 时域解法

![2024春信号与系统26第二十四讲12.1-12.3_34](https://cdn.jsdelivr.net/gh/DerrickMarcus/picgo_image/images/ch12_img5.png)

## 12.3 根据状态方程求转移函数

![2024春信号与系统26第二十四讲12.1-12.3_36](https://cdn.jsdelivr.net/gh/DerrickMarcus/picgo_image/images/ch12_img6.png)

## 12.4 离散时间系统状态方程的建立

同连续时间系统的形式，用差分代替微分。

$$
\begin{align*}
\boldsymbol{\lambda}(n+1)&=\boldsymbol{A\lambda}(n)+\boldsymbol{Bx}(n) \\
\boldsymbol{y}(n)&=\boldsymbol{C\lambda}(n)+\boldsymbol{Dx}(n)
\end{align*}
$$

（看典型结构示意图）

由定义建立。由框图或流图建立。

## 12.5 离散时间系统状态方程的求解

### 时域迭代法求解

![2024春信号与系统27第二十五讲12.4-12.7_14](https://cdn.jsdelivr.net/gh/DerrickMarcus/picgo_image/images/ch12_img7.png)

### z 变换求解

![2024春信号与系统27第二十五讲12.4-12.7_16](https://cdn.jsdelivr.net/gh/DerrickMarcus/picgo_image/images/ch12_img8.png)

![2024春信号与系统27第二十五讲12.4-12.7_17](https://cdn.jsdelivr.net/gh/DerrickMarcus/picgo_image/images/ch12_img9.png)

## 12.6 状态矢量的线性变换

选择不同的状态矢量可以得到不同的 $\boldsymbol{A},\boldsymbol{B},\boldsymbol{C},\boldsymbol{D}$ 矩阵，各状态矢量之间存在某种约束，矩阵 $\boldsymbol{A},\boldsymbol{B},\boldsymbol{C},\boldsymbol{D}$ 之间存在某种变换关系。

具体细节略。

判断系统的稳定性：

![2024春信号与系统27第二十五讲12.4-12.7_28](https://cdn.jsdelivr.net/gh/DerrickMarcus/picgo_image/images/ch12_img10.png)

## 12.7 系统的可控性和可观性

可控性 (Controllability)：给定起始状态，可以找到容许的输入量 (控制矢量)，在有限时间内把系统的所有状态引向零状态。如果可做到这点，则称系统完全可控。

可观性 (Observability)：给定输入 (控制) 后，能在有限时间内根据系统输出唯一地确定系统的起始状态。如果可做到这点，则称系统完全可观。

判别方法：

利用可控阵和可观阵判定：

![2024春信号与系统27第二十五讲12.4-12.7_37](https://cdn.jsdelivr.net/gh/DerrickMarcus/picgo_image/images/ch12_img11.png)

A 矩阵规范化之后判别。（略）

可控可观与转移函数的关系：

![2024春信号与系统27第二十五讲12.4-12.7_44](https://cdn.jsdelivr.net/gh/DerrickMarcus/picgo_image/images/ch12_img12.png)

留意一下串联、并联、级联可能发生零极点相消，导致系统不可控不稳定就行。
