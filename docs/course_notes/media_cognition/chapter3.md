# 3 Machine Learning 机器学习

## 3.1 Introduction

机器学习的基本目的：根据给定的训练样本求对某系统输入输出之间依赖关系的估计，使它能够对未知输出作出尽可能准确的预测。

数学建模表示：

根据 $n$ 个独立同分布观测样本确定预测函数 $f(\boldsymbol{x},\boldsymbol{w})$ ，在一组预测函数 $\{f(\boldsymbol{x},\boldsymbol{w})\}$ 中选择一个最优的函数 $f(\boldsymbol{x},\boldsymbol{w}_0)$ 对依赖关系进行估计，使预测的期望风险最小。

机器学习的3个基本问题：

1. 分类，输出为离散的类别标号。应用于图像分类，物体检测和识别，方法有支持向量机 SVM、卷积神经网络 CNN 等。属于监督学习。
2. 回归，输出为连续变量。应用于预测、姿态估计，方法有 Fisher 线性回归、逻辑回归等。属于监督学习。
3. 聚类（概率密度估计问题）。根据训练样本确定其概率分布。应用于聚类、异常检测。方法中，参数法有最大似然估计，非参数法有 K 近邻。属于监督学习。

机器学习的一般模型：

1. 监督学习：训练集标签样本已知。
2. 无监督学习：训练集样本标签未知。
3. 弱监督学习：图像级标注。
4. 半监督学习：小数据标注建模+无标注数据。
5. 自监督学习：利用 pretask 先验知识。
6. 跨监督学习：利用所有可用的数据，全方位学习。

机器学习的一般方法：

1. 有监督/无监督学习（针对样本标注）：监督学习（用于分类、回归），无监督学习（用于概率密度估计、聚类），半监督学习。
2. 强化学习 reinforcement learning。智能体不断与环境进行交互，通过试错的方式来获得最佳策略。
3. 元学习 meta learning。
4. 多任务学习 multi-task learning。

机器学习的条件：数据，样本，机器，评价准则。

---

机器学习的准则：偏差，方差，过拟合，欠拟合。

对于监督学习，误差的期望值为：

$$
\mathbb{E}\{ (y-h(x))^2 \}=\sigma^2 + \text{Var}[h(x)] + \text{Bias}^2[h(x)]
$$

等式右边三项分别为：采样自身的噪声方差，模型预测值本身的方差，预测值相对于真值的偏差的平方。

- 模型越复杂，则方差增加、偏差减小，可能过拟合（训练样本学的太好，缺乏对本质的理解，泛化能力下降）。
- 模型越简单，则偏差增加、方差减小，可能欠拟合（训练样本的性质没学好）。

因此应当选择合适的模型复杂度。

---

机器学习的2个经典模型：

1. **生成式** 模型(Generative Model)：对概率分布 $p(\boldsymbol{x},\omega)$ 建模，利用贝叶斯公式 $p(\omega|\boldsymbol{x})=p(\boldsymbol{x}|\omega)p(\omega)/p(\boldsymbol{x})$ . 典型方法：贝叶斯估计，高斯混合模型 HMM，隐马尔可夫模型 HMM，生成对抗网络 GAN 等。
2. **鉴别式** 模型(Discriminative Model)：直接用函数（而非概率）对 $p(\omega|\boldsymbol{x})$ 建模，**一般性能更好**。典型方法：线性判别分析 LDA，支持向量机 SVM，神经网络。

## 3.2 Perceptron

感知机算法是对生物神经细胞的简单抽象。感知机(perception)是一种线性分类模型，即它针对线性可分数据。

神经网络又称多层感知机。神经网络之所以能够发挥强大的作用就是在感知机的结构上作了叠加设计，而支持向量机算法的基础即为感知机。

感知机模型，是机器学习二分类问题中一个简单的模型。输入为样本的特征向量，输出为样本的类别代码，记为+1和-1。感知机对应于样本空间中的 **分类超平面**，属于 **鉴别式模型**。基于误分类样本点到分类界面距离的损失函数，利用梯度下降法，可对损失函数进行极小化。

激活函数：加入非线性因素，解决线性模型所不能解决的问题。如果没有非线性的激活函数，无论叠加多少层线性层，最终都等效于一个线性层。

### 3.2.1 Activation Function {#course_notes/media_cognition/section-3.2.1}

常见 激活函数 汇总

（1）阈值函数：

$$
f(x) =
\begin{cases}
1 & x \geqslant 0 \\
0 & x < 0
\end{cases}
$$

（2）分段线性函数：

$$
f(x) =
\begin{cases}
1 & x \geqslant 1 \\
\dfrac{1}{2}(1 + x) & -1 < x < 1 \\
0 & x \leqslant -1
\end{cases}
$$

（3）Sigmoid 函数：

$$
f(x)=\sigma(x) = \frac{1}{1 + e^{-x}}
$$

其导数为：

$$
f'(x)=\sigma(x)(1-\sigma(x)) = \frac{e^{-x}}{(e^{-x} + 1)^2}=\frac{e^x}{(e^x + 1)^2}
$$

特点：中心不为0，输出范围为 $(0, 1)$ ，适用于二分类问题。饱和区域可能梯度消失，通常应用于 DNN 的最后一层。

Sigmoid 激活函数多使用在二分类问题。对于大于二分类问题，如果类别之间存在相互关系使用 Sigmoid，反之使用 Softmax。

（4）双曲正切函数：

$$
\tanh(x) = 2\sigma(2x)-1 = \frac{e^x - e^{-x}}{e^x + e^{-x}}
$$

其导数为：

$$
f'(x)=1 - \tanh^2(x) = \frac{4}{(e^x + e^{-x})^2}
$$

特点：中心为0，输出范围为 $(-1, 1)$ . 饱和区域可能梯度消失，通常应用于 RNN。

（5）ReLU 函数：

$$
f(x)=\text{ReLU}(x) = \max(0, x)=
\begin{cases}
x & x \geqslant 0 \\
0 & x < 0
\end{cases}
$$

其导数为：

$$
f'(x)=
\begin{cases}
1 & x \geqslant 0 \\
0 & x < 0
\end{cases}
$$

特点：中心不为0，正值部分梯度有效传递，有效解决梯度消失问题，但是可能会有 dead neuron 问题（神经元死亡），负值部分输出为0导致梯度为0，不会更新权重。

（6）LeakyReLU 函数：

$$
f(x) =
\begin{cases}
x & x \geqslant 0 \\
\alpha x & x < 0
\end{cases}
$$

其导数为：

$$
f'(x) =
\begin{cases}
1 & x \geqslant 0 \\
\alpha & x < 0
\end{cases}
$$

主要解决 ReLU 的输出为0问题， $\alpha$ 为小常数。

（7）PReLU 函数：将 LeakyReLU 的 $\alpha$ 设为可学习的参数。

（8）Softmax 函数，可以看作为 Sigmoid 函数的扩展，适用于多分类任务。

$$
\begin{align*}
\boldsymbol{x} &= [x_1, \cdots, x_n]^T \in \mathbb{R}^n \\
\text{softmax}(\boldsymbol{x}) &= [\text{softmax}(x_1), \cdots, \text{softmax}(x_n)]^T \\
\text{softmax}(x_i) &= \frac{\exp(x_i)}{\displaystyle\sum_{j=1}^n \exp(x_j)}
\end{align*}
$$

它将原始分数（也称为 logits）转换为表示概率分布的数值，且所有类别的概率之和等于1，因此适用于多类别分类问题，且类别之间互斥的场合，每个样本只属于一个类别。

Softmax 将多个神经元的输出，映射到 $(0,1)$ 区间内，并做归一化，可以看成是当前输出属于各个分类的概率，从而进行多分类。

Softmax 常作为网络输出层，可以自然地表示具有 $n$ 个可能值的离散型随机变量的概率分布。

!!! note "为什么使用 Softmax 作为回归分类函数"
    首先，Softmax 使用了指数，这样可以让大的值更大，让小的更小，增加区分对比度，学习效率更高。其次，Softmax 是连续可导的，消除了拐点，这个特性在机器学习的梯度下降法等地方非常必要。

### 3.2.2 Binary Classification

讨论二分类问题的感知机模型。

假设 $\boldsymbol{x}\in X \subseteq \mathbb{R}^n$ 为样本的特征向量，样本类别标号 $y \in Y=\{1,-1\}$ ，将特征向量映射为类别标号的函数：

$$
f(\boldsymbol{x})=\text{sign}(\boldsymbol{w}^T\boldsymbol{x}+b)
$$

通过符号函数，将大于 0 的分为 +1 类，小于 0 的分为 -1 类。

分类界面即为 $\boldsymbol{w}^T\boldsymbol{x}+b=0$ ，误分类的点满足 $-y(\boldsymbol{w}^T\boldsymbol{x}+b)>0$ . 选择损失函数 $L(\boldsymbol{w},b)$ 为<span style="color:red">误分类点到超平面的总距离，越小越好</span>。得到优化问题（ $M$ 为误分类点集合）：

$$
\min_{\boldsymbol{w},b} L(\boldsymbol{w},b) = \sum_{\boldsymbol{x}_i\in M} -y_i(\boldsymbol{w}^T\boldsymbol{x}_i+b)
$$

求损失函数的梯度：

$$
\frac{\partial L(\boldsymbol{w},b)}{\partial \boldsymbol{w}} = -\sum_{\boldsymbol{x}_i\in M} y_i\boldsymbol{x}_i ,\quad \frac{\partial L(\boldsymbol{w},b)}{\partial b} = -\sum_{\boldsymbol{x}_i\in M} y_i
$$

若采用 随机梯度下降 SGD ，每次选取一个误分类点，更新权重与偏置：

$$
\boldsymbol{w} \leftarrow \boldsymbol{w}-\eta\frac{\partial L}{\partial \boldsymbol{w}} =\boldsymbol{w} + \eta y_i\boldsymbol{x}_i ,\quad b \leftarrow b-\eta\frac{\partial L}{\partial b}=b +\eta y_i
$$

如果采用 批量梯度下降 BGD，则使用全部误分点，更新权重与偏置：

$$
\boldsymbol{w} \leftarrow \boldsymbol{w} + \eta \sum_{\boldsymbol{x}_i\in M} y_i\boldsymbol{x}_i ,\quad b \leftarrow b + \eta \sum_{\boldsymbol{x}_i\in M} y_i
$$

感知机算法步骤：

1. 选择初值 $\boldsymbol{w}=\boldsymbol{0},b=0$ ，学习率 $\eta,\;0<\eta\leqslant 1$ .
2. 在训练集中选取数据 $\boldsymbol{x}_i$ ，前向传播计算。
3. 若 $y_i(\boldsymbol{w}^T\boldsymbol{x}_i+b)\leqslant0$ ，误分类，则更新权重： $\boldsymbol{w} \leftarrow \boldsymbol{w} + \eta y_i\boldsymbol{x}_i$ ，偏置：$b \leftarrow b + \eta y_i$ .

说明：采用随机梯度下降法时，感知机每一轮学习，逐点计算 $\boldsymbol{w}^T\boldsymbol{x}_i+b$ ，对正确分类点不更新参数，对误分类点按更新公式更新参数，然后计算下一个点，直至没有误分类点或损失函数取得极小值，训练结束。

上面是采用了 **随机梯度下降法 SGD** 求解（一般均采用，但容易收敛到局部最优）。如果采用 **批量梯度下降法 BGD**，则使用全部误分样本进行参数更新，即公式中的学习率后有求和符号。

<br>

当训练数据集线性可分时，感知机学习算法是收敛的，例如二维平面的 and 函数、or 函数。对于线性不可分数据集，迭代过程振荡，例如<span style="color:red">单个感知机不能解决二维平面的异或函数</span>。

感知机算法存在许多解，既依赖于初值，也依赖迭代过程中误分类点的**选择顺序**。

为得到唯一分离超平面，需要增加约束，如后续课程中介绍的支持向量机。

多层感知机比单层感知机具有更好的模型描述能力。

## 3.3 Regression

### 3.3.1 Linear Regression

从最简单的线性回归说起。线性回归问题的3个特点：

1. 线性
    1. 关于特征 $\boldsymbol{x}$ 是线性的。由此提出多项式回归。
    2. 关于参数 $\boldsymbol{w}$ 是线性的。由此提出神经网络。
    3. 从全局看线性组合之后直接输出结果，无任何处理。由此提出线性分类，比如加激活函数等。
2. 全局性：对所有的特征空间拟合，不是将特征分段拟合。由此提出决策树方法。
3. 数据未加工：没有降维映射等。由此提出 PCA，流形学习等方法。

类似于分类问题，线性回归问题的模型为：

$$
y=\boldsymbol{w}^T\boldsymbol{x}+b,\quad \boldsymbol{x},\boldsymbol{w}\in\mathbb{R}^n
$$

假设有 $N$ 个数据点 $\boldsymbol{X}=[\boldsymbol{x}_1,\cdots,\boldsymbol{x}_N]^T\in \mathbb{R}^{N\times n}, \boldsymbol{y}=[y_1,\cdots,y_N]^T\in\mathbb{R}^n$ . 注意此时矩阵 $\boldsymbol{X}$ 中每一行是一个数据点，即列数为数据点的维度 $d$ ，行数为数据点的个数 $N$ .

最小二乘法：找到一条直线，使所有样本到直线上预测点的均方误差最小，由此定义该均方误差为损失函数：

$$
L(\boldsymbol{X},\boldsymbol{y},\boldsymbol{w},b) = \frac{1}{N}\sum_{i=1}^N \left[y_i - (\boldsymbol{w}^T\boldsymbol{x}_i + b)\right]^2
$$

将偏置 $b$ 添加到权重中，得到增广向量：

$$
\begin{gather*}
\boldsymbol{x}=\begin{pmatrix}
1 \\ \boldsymbol{x}
\end{pmatrix}, \quad
\boldsymbol{w}=\begin{pmatrix}
b \\ \boldsymbol{w}
\end{pmatrix} \\
L(\boldsymbol{X},\boldsymbol{y},\boldsymbol{w}) = \frac{1}{n}\|\boldsymbol{y} - \boldsymbol{X}\boldsymbol{w}\|^2 \\
\frac{\partial L}{\partial \boldsymbol{w}} = -\frac{2}{N}(\boldsymbol{y} - \boldsymbol{X}\boldsymbol{w})^T\boldsymbol{X}
\end{gather*}
$$

（1）损失函数是凸函数，由梯度为0，**直接得到闭式解**：

$$
\boldsymbol{w}^* = (\boldsymbol{X}^T\boldsymbol{X})^{-1}\boldsymbol{X}^T\boldsymbol{y}
$$

（2）优化方法求解，梯度下降：

$$
\boldsymbol{w}_t = \boldsymbol{w}_{t-1} - \eta\frac{\partial L}{\partial \boldsymbol{w}_{t-1}}
$$

---

### 3.3.2 Polynomial Regression

与线性回归几乎相同，只需把特征向量上各个维度的值，改为同一个 $x$ 的幂次即可：

$$
\begin{align*}
\boldsymbol{x}=\begin{pmatrix}
    1 \\ x_1 \\ x_2 \\ \cdots \\ x_n
\end{pmatrix} &\to
\boldsymbol{x} =\begin{pmatrix}
    1 \\ x \\ x^2 \\ \cdots \\ x^n
\end{pmatrix} \\
\boldsymbol{X} =\begin{pmatrix}
    1 & x^{(1)}_1 & x^{(1)}_2 & \cdots & x^{(1)}_n \\
    1 & x^{(2)}_1 & x^{(2)}_2 & \cdots & x^{(2)}_n \\
    \vdots & \vdots & \vdots & \ddots & \vdots \\
    1 & x^{(N)}_1 & x^{(N)}_2 & \cdots & x^{(N)}_n
\end{pmatrix} &\to
\boldsymbol{X}=\begin{pmatrix}
    1 & x_1 & x_1^2 & \cdots & x_1^n \\
    1 & x_2 & x_2^2 & \cdots & x_2^n \\
    \vdots & \vdots & \vdots & \ddots & \vdots \\
    1 & x_N & x_N^2 & \cdots & x_N^n
\end{pmatrix}
\end{align*}
$$

### 3.3.3 Logistic Regression

逻辑回归(Logistic Regression)也称逻辑回归分析，是一种广义线性回归分析模型，常用于目标检测、数据挖掘，疾病诊断等领域。实质上多用于分类任务。

例如：以病情分析为例，因变量 $y$ 是否有病，值为“是”或“否”，而自变量可包括很多因素，如年龄、性别、饮食习惯、感染等。自变量既可以是连续，也可以是离散。

逻辑回归模型也具有 $\boldsymbol{w}^T\boldsymbol{x}+b$ 的形式，其区别在于因变量不同，线性回归直接将 $w^Tx+b$ 作为因变量，而 logistic 回归则通过将 $\boldsymbol{w}^T\boldsymbol{x}+b$ 对应一个隐状态 $p=L(\boldsymbol{w}^T\boldsymbol{x}+b)$ ，根据 $p$ 值决定因变量值。虽然被称为回归，但其实际上是分类模型，常用于二分类，估计可能性大小。线性模型只返回实数，需要逻辑回归转换为概率。

类标签为 $y\in\{0,1\}$ ，逻辑回归模型为：

$$
y=h(\boldsymbol{x})=\sigma(\boldsymbol{w}^T\boldsymbol{x}+b)=\frac{1}{1+\exp(-(\boldsymbol{w}^T\boldsymbol{x}+b))}
$$

逻辑函数也称 Sigmoid 函数，将逻辑回归也可看作感知机 Perceptron, 不同的是使用了 Sigmoid 激活函数。

定义 LR 损失函数为交叉熵损失函数 Cross-Entropy Loss：

$$
L=-\sum_{i=1}^N \left[y_i\log(h(\boldsymbol{x}_i)) + (1-y_i)\log(1-h(\boldsymbol{x}_i))\right]
$$

交叉熵函数表征真实样本标签和预测概率之间的差值（交叉熵函数由最大似然估计推导得出）。

考虑单个损失函数项：

$$
\begin{align*}
L_i &= -y_i\log(h(\boldsymbol{x}_i)) - (1-y_i)\log(1-h(\boldsymbol{x}_i)) \\
&=\begin{cases}
-\log(h(\boldsymbol{x}_i)), & y_i=1 \\
-\log(1-h(\boldsymbol{x}_i)), & y_i=0
\end{cases}
\end{align*}
$$

上式表明 $\boldsymbol{w}^T\boldsymbol{x}+b$ 越接近真实标签（实际是 $h(\boldsymbol{x})$ 接近真实标签），损失函数越小、越接近0。

优化方法：梯度下降

$$
\begin{align*}
\frac{\partial L}{\partial \boldsymbol{w}} &= \sum_{i=1}^N \left(h(\boldsymbol{x}_i)-y_i\right)\boldsymbol{x}_i ,&\quad
\frac{\partial L}{\partial b} &= \sum_{i=1}^N \left(h(\boldsymbol{x}_i)-y_i\right) \\
\boldsymbol{w} &\leftarrow \boldsymbol{w} - \eta\frac{\partial L}{\partial \boldsymbol{w}} ,&\quad
b &\leftarrow b - \eta\frac{\partial L}{\partial b}
\end{align*}
$$

由于模型的输出范围为 $h(\boldsymbol{x})\in(0,1)$ ，且 $\boldsymbol{w}^T\boldsymbol{x}+b>0 \Leftrightarrow h(\boldsymbol{x})>0.5$ ，因此可以 $0.5$ 为分类基准，这也符合我们的的直观认知。

### 3.3.4 Fisher LDA

Fisher 线性分类器(LDA, Linear Discriminant Analysis)的基本思想：通过寻找一个投影方向（线性变换，线性组合），将高维问题降低到一维问题来解决，并且要求变换后的一维数据具有如下性质：同类样本尽可能聚集在一起，不同类的样本尽可能地远。Fisher 判别准则为：最小化类别重叠，得不同类均值投影分开大，而每个类的内部方差小。即：<span style="color:red">类间方差大，类内方差小</span>。

Fisher 线性判别，即通过给定的训练数据，确定投影方向 $\boldsymbol{w}$ 和阈值 $w_0$ ，即确定线性判别函数，然后根据这个线性判别函数，对测试数据进行测试得到它的类别。

<br>

算法步骤：

（1）假设有 $N$ 个样本 $\boldsymbol{x}_1,\cdots,\boldsymbol{x}_N\in\mathbb{R}^n$ ，对应的标签为 $y_1,\cdots,y_N$ .

（2）其中 $N_1$ 个属于类别 $\omega_1$ ， $N_2$ 个属于类别 $\omega_2$ ，满足 $N_1+N_2=N$ .

（3）获取投影向量 $\boldsymbol{w}$ . 计算 $z=\boldsymbol{w}^T\boldsymbol{x}$ 将样本投影到一维空间，并设置阈值 $w_0$ . 当 $z\geqslant w_0$ 时判定为类别 $\omega_1$ ，当 $z< w_0$ 时判定为类别 $\omega_2$ .

（4）寻找使得类别之间区分度最大的投影向量 $\boldsymbol{w}$ . 具体方法为：

计算 **均值**： $\boldsymbol{\mu}_i=\dfrac{1}{N_i}\displaystyle\sum_{y_j\in\omega_i}\boldsymbol{x}_j$ .

计算 **类内散度矩阵**： $\boldsymbol{S}_i=\displaystyle\sum_{y_j\in\omega_i}(\boldsymbol{x}_j-\boldsymbol{\mu}_i)(\boldsymbol{x}_j-\boldsymbol{\mu}_i)^T$ ，相加得到**总类内散度矩阵** $\boldsymbol{S}_\omega=\displaystyle\sum_i\boldsymbol{S}_i$ .

计算 **总类间散度矩阵**： $\boldsymbol{S}_b=(\boldsymbol{\mu}_1-\boldsymbol{\mu}_2)(\boldsymbol{\mu}_1-\boldsymbol{\mu}_2)^T$ ，为对称矩阵。

注：如果是多分类问题，则 $\boldsymbol{S}_b=\displaystyle\sum_i N_i(\boldsymbol{\mu}_i-\boldsymbol{\mu})(\boldsymbol{\mu}_i-\boldsymbol{\mu})^T$ .

定义**准则函数** $\displaystyle\max_{\boldsymbol{w}}J(\boldsymbol{w})=\dfrac{\boldsymbol{w}^T\boldsymbol{S}_b\boldsymbol{w}}{{\boldsymbol{w}^T\boldsymbol{S}_\omega}\boldsymbol{w}}$ ，我们期望求它的 **最大值**。拉格朗日函数 $L(\boldsymbol{w},\lambda)=\boldsymbol{w}^T\boldsymbol{S}_b\boldsymbol{w}-\lambda(\boldsymbol{w}^T\boldsymbol{S}_\omega\boldsymbol{w}-1)$ .

通过推导得到 **投影向量** 为 $\boldsymbol{w}=\boldsymbol{S}_\omega^{-1}(\boldsymbol{\mu}_1-\boldsymbol{\mu}_2)$ .

**判定阈值** 为： $w_0=\dfrac{n_1\tilde{\mu}_1+n_2\tilde{\mu}_2}{n_1+n_2}$ ，即为投影后均值向量 $\tilde{\mu}_i=\boldsymbol{w}^T\boldsymbol{\mu}_i$ 的加权平均。将每一个样本 $\boldsymbol{x}_i$ 的投影值 $z_i=\boldsymbol{w}^T\boldsymbol{x}_i \gtrless w_0$ ，与阈值比较，得到类别。

## 3.4 SVM

支持向量机 SVM(support vector machine)是一种基于统计学习理论的机器学习方法。

核心理念是分类间隔最大化，寻找一个置信度最大的分类面。对训练数据集找到几何间隔最大的超平面，以最大的置信度对训练数据进行分类。即不仅将正负实例分开，而且对难分的样本点（离超平面最近的点）也有足够大的置信度将其分开。这样超平面对未知的新样本似有较好的分类预测能力。

SVM 的特点：是小样本条件下的统计学习方法；具备严格的数学理论基础和直观集合解释；在处理不均匀、离散、稀疏的数据时有明显优势；适用于适用于样本有限情况下的优化求解。

目标：找到一个超平面，使得其能够尽可能得将两类数据正确的分开，且 **分开的两类数据点距离分类界面最远** 。构造约束条件下的优化问题（约束二次优化问题），求解最优分类器。

通常由3三种典型问题：线性可分问题，近似线性可分问题，非线性分类问题。

### 3.4.1 线性可分问题

类似于感知机问题，我们有训练样本 $\{(\boldsymbol{x}_i,y_i)\},i=1,\cdots N$ ，假设有一个超平面 $\boldsymbol{w}^T\boldsymbol{x}+b=0$ ，能够将样本分开，则：对于类别 $y_i=1$ 有 $\boldsymbol{w}^T\boldsymbol{x}_i+b>0$ ，对于类别 $y_i=-1$ 有 $\boldsymbol{w}^T\boldsymbol{x}_i+b<0$ . 因此正确分类的点满足 $y_i(\boldsymbol{w}^T\boldsymbol{x}_i+b)>0$ ，误分类的点满足 $y_i(\boldsymbol{w}^T\boldsymbol{x}_i+b)\leqslant 0$ .

对于最大间隔超平面点（支持向量），有 $\boldsymbol{w}^T\boldsymbol{x}_i+b=\pm1,\;y_i(\boldsymbol{w}^T\boldsymbol{x}_i+b)=1$ （此处右边原本为任意实数 $r$ 但是可以通过放缩取1）。因此，正确分类的点均满足 $y_i(\boldsymbol{w}^T\boldsymbol{x}_i+b)\geqslant 1$ .

两类支持向量分别满足 $\boldsymbol{w}^T\boldsymbol{x}^+b=1,\;\boldsymbol{w}^T\boldsymbol{x}^-=-1$ ，超平面的法向量为 $\dfrac{\boldsymbol{w}}{\|\boldsymbol{w}\|}$ ，因此间隔宽度为 $M=(\boldsymbol{x}^+-\boldsymbol{x}^-)\dfrac{\boldsymbol{w}}{\|\boldsymbol{w}\|}=\dfrac{2}{\|\boldsymbol{w}\|}$ .

优化函数：

$$
\max \dfrac{2}{\|\boldsymbol{w}\|} \iff \min \dfrac{1}{2}\|\boldsymbol{w}\|^2 \quad \text{s.t.} \; y_i(\boldsymbol{w}^T\boldsymbol{x}_i+b)\geqslant 1,\; i=1,\cdots,N
$$

为有线性约束的优化问题，Lagrange 函数为（满足 KKT 条件下）：

$$
L(\boldsymbol{w}, b, \boldsymbol{\alpha}) = \frac{1}{2} \|\boldsymbol{w}\|^2 - \sum_{i=1}^{n} \alpha_i \left( y_i (\boldsymbol{w}^T \boldsymbol{x}_i + b) - 1 \right) ,\quad \alpha_i \geqslant 0
$$

求解其最小值：

$$
\begin{gather*}
\min_{\boldsymbol{w},b}L(\boldsymbol{w}, b, \boldsymbol{\alpha}) \\
\frac{\partial L}{\partial \boldsymbol{w}}=0 \implies \boldsymbol{w} = \sum_{i=1}^{N} \alpha_i y_i \boldsymbol{x}_i \\
\frac{\partial L}{\partial b}=0 \implies \sum_{i=1}^{N} \alpha_i y_i = 0
\end{gather*}
$$

由于为凸问题，且满足 KKT 条件，可交换求解次序，原优化问题等价于：

$$
\begin{gather*}
\max_{\boldsymbol{\alpha}}(\min_{\boldsymbol{w},b} L(\boldsymbol{w}, b, \boldsymbol{\alpha}))
=\max_{\boldsymbol{\alpha}} \left( \sum_{i=1}^{n} \alpha_i -\frac{1}{2} \sum_{i=1}^{n} \sum_{j=1}^{n} \alpha_i \alpha_j y_i y_j (\boldsymbol{x}_i^T \boldsymbol{x}_j) \right) \\
\text{s.t.}\; \alpha_i \geqslant 0,\; \sum_{i=1}^{n} \alpha_i y_i = 0
\end{gather*}
$$

对于上式采用序列最小优化算法 SMO (sequential minimal optimization)可得到最优解 $\boldsymbol{\alpha}$ .

不等式约束的优化问题在最优值处，须满足 KKT 条件。本问题中的 KKT 条件：

$$
\begin{cases}
\alpha_i \geqslant 0, \\
y_i(\boldsymbol{w}^T\boldsymbol{x}_i+b)-1\geqslant 0,\\
\alpha_i[y_i(\boldsymbol{w}^T\boldsymbol{x}_i+b)-1]=0
\end{cases}
$$

由第一条和第三条可知， $\alpha_i>0 \Rightarrow y_i(\boldsymbol{w}^T\boldsymbol{x}_i+b)=1$ ，即 $\boldsymbol{x}_i$ 正好是支持向量。

最优解 $\boldsymbol{\alpha}$ 满足 $\boldsymbol{w}^*=\displaystyle\sum_{i=1}^N \alpha_i^*y_i\boldsymbol{x}_i$ ，且 $\alpha_i$ 只在 $\boldsymbol{x}_1$ **是支持向量时非零**，由此得到 **权重的最优解**。这表明，<span style="color:red">最优分类面的权重系数向量由支持向量决定，是支持向量的线性组合</span>。

任选一个支持向量 $\boldsymbol{x}_j$ 带入 $y_j(\boldsymbol{w}^{*T}\boldsymbol{x}_j+b^*)=1$ 得到 **偏置的最优解**： $b^*= y_j -\boldsymbol{w}^{*T}\boldsymbol{x}_j$ .

最终有：

$$
\boldsymbol{w}^*=\sum_{\boldsymbol{x}_i\in \text{SV}} \alpha_i^*y_i\boldsymbol{x}_i,\quad b^*=y_j-\boldsymbol{w}^{*T}\boldsymbol{x}_j,\;\boldsymbol{x}_j\in \text{SV}
$$

> 注意，最后得到的判决面系数 $\boldsymbol{w}^*,b^*$ 系数是固定的，不能任意缩放，由 支持向量满足 $y_j(\boldsymbol{w}^{*T}\boldsymbol{x}_j+b^*)=1$ 保证。

### 3.4.2 近似线性可分问题

由于噪声和样本异常值的出现等因素，数据有时是近似线性可分的有特异点(outlier)，个别样本出现在异类区域。

为每个样本点加入松弛因子 $\xi_i\geqslant 0$ ，使函数间隔加上松弛变量后大于等于1，代表容错范围。约束条件变为：

$$
y_i(\boldsymbol{w}^T\boldsymbol{x}_i+b)\geqslant 1-\xi_i,\quad \xi_i\geqslant 0
$$

求解：

$$
\min_{\boldsymbol{w},b,\xi} \left(\frac{1}{2}\|\boldsymbol{w}\|^2 + C\sum_{i=1}^{N}\xi_i \right) ,\quad
\text{s.t.} \; y_i(\boldsymbol{w}^T\boldsymbol{x}_i+b)\geqslant 1-\xi_i,\; \xi_i\geqslant 0
$$

引入参数 $C$ 防止过拟合，是一个大于 0 的常数，可以理解为错误样本的惩罚程度， $C$ 越大松弛因子就只能越小，容忍度就越小。

当 $C\to +\infty$ 时，不允许出现分类误差的样本存在，这就是一个 hard-margin SVM 问题（过拟合）。当 $C\to 0$ 时，即不再关注分类是否正确，只要求间隔越大越好，那么将无法得到有意义的解且算法不会收敛（欠拟合）。

Lagrange 函数和优化问题为：

$$
\begin{gather*}
L(\boldsymbol{w}, b, \boldsymbol{\alpha}, \boldsymbol{\xi}) = \frac{1}{2}\|\boldsymbol{w}\|^2 + C\sum_{i=1}^{N}\xi_i - \sum_{i=1}^{N} \alpha_i (y_i(\boldsymbol{w}^T\boldsymbol{x}_i+b) - 1 + \xi_i) - \sum_{i=1}^{N} \beta_i \xi_i \\
\boxed{\min L(\boldsymbol{w}, b, \boldsymbol{\alpha}, \boldsymbol{\xi}) \quad \text{s.t.}\; \alpha\geqslant 0,\beta_i\geqslant 0}
\end{gather*}
$$

$$
\begin{align*}
\frac{\partial L}{\partial \boldsymbol{w}}=0 &\implies \boldsymbol{w} = \sum_{i=1}^{N} \alpha_i y_i \boldsymbol{x}_i \\
\frac{\partial L}{\partial b}=0 &\implies \sum_{i=1}^{N} \alpha_i y_i = 0 \\
\frac{\partial L}{\partial \xi_i}=0 &\implies C-\alpha_i-\beta_i = 0
\end{align*}
$$

类似于线性可分问题，得到等价的对偶问题：

$$
\max_{\boldsymbol{\alpha}} \left( \sum_{i=1}^{N} \alpha_i - \frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_i \alpha_j y_i y_j (\boldsymbol{x}_i^T \boldsymbol{x}_j) \right) ,\quad
\text{s.t.} \; 0\leqslant \alpha_i \leqslant C,\; \sum_{i=1}^{N} \alpha_i y_i = 0
$$

可见，与线性分类问题相比，只多了一个 $0\leqslant \alpha_i \leqslant C$ 的约束条件。

### 3.4.3 非线性分类问题

处理思路：使用 **核函数** (kernel function) $K$ ，将样本映射到高维空间使之线性可分。

$$
\begin{align*}
\boldsymbol{x}_i\in\mathbb{R}^n &\rightarrow \varphi(\boldsymbol{x}_i)\in\mathbb{R}^d,\quad d \gg n \\
\boldsymbol{x}_i^T\boldsymbol{x}_j &\rightarrow \varphi(\boldsymbol{x}_i)^T\varphi(\boldsymbol{x}_j)=K(\boldsymbol{x}_i,\boldsymbol{x}_j)
\end{align*}
$$

!!! question "为什么使用核函数 $K$ ，而不是直接使用映射函数 $\varphi$ ？"
    样本从低维空间映射到高维空间后，维度可能会很大，如果将全部样本的点乘全部进行计算，这样的计算量过大。如果找到一个核函数 $K$ ，使其在高维空间的内积，等于在原始样本空间通过核函数 $K$ 的直接计算结果，就不需要找映射函数 $\varphi$ ，也避免计算高维甚至无穷维空间的内积了。

使用核函数构造非线性 SVM 时，既可以先构造 $\varphi$ 的变换，然后基于上式得到核函数 $K$ ；也可以越过变换直接构造核函数 $K$ 解决问题，因为最终 SVM 的优化和推断都不再依赖 $\varphi$ .

<br>

**常用核函数**，假设 $\boldsymbol{x}_i \in \mathbb{R}^p\rightarrow\varphi(\boldsymbol{x}_i)\in\mathbb{R}^d$：

（1）线性核函数： $K(\boldsymbol{x}_i,\boldsymbol{x}_j)=\boldsymbol{x}_i^T\boldsymbol{x}_j$ .

（2）多项式核函数：

有序单项式空间核函数： $K_d(\boldsymbol{x}_i,\boldsymbol{x}_j)=(1+\boldsymbol{x}_i^T\boldsymbol{x}_j)^d$ .

齐次多项式核函数： $K_d(\boldsymbol{x}_i,\boldsymbol{x}_j)=(\boldsymbol{x}_i^T\boldsymbol{x}_j)^d$ ，其中 $d$ 为多项式的次数。

上面二者的区别在于 **是否扩充为所有不超过** $d$ **阶的有序单项式**，例如当 $p=d=2$ 时：

$$
\begin{align*}
\text{case 1: }\varphi(\boldsymbol{x})&=[x_1^2,x_2^2,x_1x_2,x_2x_1,\sqrt{2}x_1,\sqrt{2}x_2,1]^T \\
\text{case 2: }\varphi(\boldsymbol{x})&=[x_1^2,x_2^2,x_1x_2,x_2x_1]^T
\end{align*}
$$

（3）高斯核函数： $K(\boldsymbol{x}_i,\boldsymbol{x}_j)=\exp\left(-\dfrac{\|\boldsymbol{x}_i-\boldsymbol{x}_j\|^2}{2\sigma^2}\right)$ ， $\sigma$ 为高斯核的宽度。

（4）Sigmoid 核函数： $K(\boldsymbol{x}_i,\boldsymbol{x}_j)=\tanh(\beta_0 \boldsymbol{x}_i^T\boldsymbol{x}_j+\beta_1)$ ，其中 $\beta_0,\beta_1$ 为可调参数。

优化函数：

$$
\max_{\boldsymbol{\alpha}} \left( \sum_{i=1}^{N} \alpha_i - \frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_i \alpha_j y_i y_j K(\boldsymbol{x}_i,\boldsymbol{x}_j) \right) , \quad
\text{s.t.} \; 0\leqslant \alpha_i \leqslant C,\; \sum_{i=1}^{N} \alpha_i y_i = 0
$$

非线性分类的决策函数：

$$
\begin{gather*}
f(\boldsymbol{x};\boldsymbol{\alpha},b)=\boldsymbol{w}^T\varphi(\boldsymbol{x})+b
=\boxed{\sum_{\boldsymbol{x}_i\in\text{SV}} \alpha_i^* y_i K(\boldsymbol{x}_i,\boldsymbol{x}) + b} \\
\boldsymbol{w}^* = \sum_{i=1}^{N} \alpha_i^* y_i \varphi(\boldsymbol{x}_i), \quad
b^* = y_j - \sum_{i=1}^{N} \alpha_i^* y_i K(\boldsymbol{x}_i,\boldsymbol{x}_j),\; \boldsymbol{x}_j\in\text{SV}
\end{gather*}
$$

决策判别的超平面即为 $\boldsymbol{w}^T\varphi(\boldsymbol{x})+b=0$ .

---

补充：有效核函数的充要条件

Mercer 定理：有效核函数的充要条件为 核矩阵（核函数矩阵形式）$K$ 是对称半正定的。

$$
K =
\begin{pmatrix}
K(\boldsymbol{x}_1, \boldsymbol{x}_1) & \cdots & K(\boldsymbol{x}_1, \boldsymbol{x}_n) \\
\vdots & \ddots & \vdots \\
K(\boldsymbol{x}_n, \boldsymbol{x}_1) & \cdots & K(\boldsymbol{x}_n, \boldsymbol{x}_n)
\end{pmatrix}
$$

对称性：对于任意 $\boldsymbol{x}_i$ 和 $\boldsymbol{x}_j$ ，有 $K(\boldsymbol{x}_i, \boldsymbol{x}_j) = K(\boldsymbol{x}_j, \boldsymbol{x}_i)$ .

半正定性：对于任意非零向量 $\boldsymbol{x}$ ，有 $\boldsymbol{x}^T K \boldsymbol{x} \geqslant 0$ .
