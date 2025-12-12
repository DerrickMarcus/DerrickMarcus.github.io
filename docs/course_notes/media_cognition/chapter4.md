# 4 Deep Learning 深度学习

## 4.1 Introduction

浅层学习（Shallow Learning）：传统机器学习和信号处理仅含单层非线性变换，称为浅层学习结构。例如感知器模型，线性判别分析，支持向量机，隐马尔可夫模型，条件随机场。它们对复杂函数表示能力有限，对复杂分类问题性能受限。

深度学习（Deep Learning）：受大脑的分层结构启发，利用多个隐层的人工神经网络赋予特征学习能力；通过学习深层非线性网络结构，实现复杂函数的逼近，从数据中学习本质特征。

深度学习强调模型结构的深度，通常有 5 层以上的隐层节点。深度学习通过**逐层特征变换**，将样本从原特征空间变换到新的特征空间，从而使分类预测更加容易；利用**大数据**来学习特征，实现对模式本质特征的自动学习。

深度学习的发展历史：神经网络概念及人工神经元的数学模型（1943），神经集合体假设（1949），感知器模型 Perceptron（1957），BP 神经网络算法，Neocognitron 模型（1980），卷积神经网络 CNN，长短时记忆网络 LSTM（1997），深度置信网络 DBN（2006），自编码器 Auto encoder（2006），生成式对抗网络 GAN（2014），Self-attention & Transformer（2017）。

## 4.2 人工神经元模型

回顾机器学习中的感知机 Perception，感知机一种最简单形式的前馈神经网络，可应用于二元线性分类。逻辑回归 Logistic Regression 是解决二分类问题的机器学习方法，用于估计某种事物的可能性。

人工神经元模型（neuron）：神经网络的基本单元

1. 输入：接受外界或者前层输入。
2. 连接：根据权重对输入加权。
3. 激活函数：连接层输出的非线性映射。
4. 输出：输出至下个隐含层。

关于**激活函数**的总结，可见 [第3讲 机器学习-感知机-激活函数](./chapter3.md#course_notes/media_cognition/section-3.2.1)。

损失函数（Loss Function），衡量模型预测值和真实值之间的差异，以评价模型优劣。损失函数一般是非负的，有下界。

多分类问题中的损失函数：**Softmax 激活 + 负对数似然损失**

$$
L=-\sum_{i} \log\left(\frac{\exp(z_{i,j})}{\displaystyle\sum_{k} \exp(z_{i,k})}\right)
$$

上式中 $j$ 为第 $i$ 个样本 $x_i$ 的真实类别标签， $z_{i,k}$ 为输出的预测向量。

假设有 $K$ 个类别，某样本输出 $K$ 维向量 $\boldsymbol{z}_i = [z_{i,1}, z_{i,2}, \cdots, z_{i,K}]^T$ ，则其属于第 $j$ 类的概率为 $\hat{y}_{i,j} = \dfrac{\exp(z_{i,j})}{\sum_{k=1}^{K} \exp(z_{i,k})}$ ；若样本的真实类别标签为第 $j$ 类，则 $\hat{y}_{i,j}$ 越大越接近于 1 则损失越小越接近于 0；反之损失越大。

<br>

回归问题中的损失函数：

（1）均方损失 $L=\dfrac{1}{2} \displaystyle\sum_{i}\|y_i - \hat{y}_i\|^2,\;\hat{y}_i=h(\boldsymbol{x}_i)$ ，其中 $y_i$ 为真实值， $\hat{y}_i$ 为预测值。

（2）L1 损失 $L=\displaystyle\sum_{i}\|y_i - \hat{y}_i\|$ ，其中 $y_i$ 为真实值， $\hat{y}_i$ 为预测值。

（3）Smooth-L1 损失：

$$
L_i=\begin{cases}
\dfrac{1}{2}(y_i - \hat{y}_i)^2 & \text{if } |y_i - \hat{y}_i| < 1 \\
|y_i - \hat{y}_i| - \dfrac{1}{2} & \text{otherwise}
\end{cases},\quad L=\sum_{i} L_i
$$

## 4.3 FFN

全连接前馈神经网络（Fully-connected FeedForward Networks, FFN）。

在网络结构中，每个神经元有对应权重和偏置参数，所有神经元的权重和偏置定义为网络参数。

若相邻两层的神经元的数量分别为 $n_1,n_2$ ，则两层神经元之间的网络，权重的参数量为 $n_1n_2$ ，偏置的参数量为 $n_2$ ，因此两层神经元之间的参数量为 $n_1n_2+n_2$ . 整个神经网络的参数量为所有层之间的参数量之和，有 $N$ 层则求和 $N-1$ 次。

> 这一点很容易理解，因为单层网络的输出形如 $\mathbb{R}^{n_1}\to\mathbb{R}^{n_2}:\;\boldsymbol{y}=\boldsymbol{w}^T\boldsymbol{x}+\boldsymbol{b},\;\boldsymbol{w}\in\mathbb{R}^{n_1\times n_2},\;\boldsymbol{b}\in\mathbb{R}^{n_2}$ .

### 4.3.1 BP

Back Propagation（BP）学习算法

最常用的神经网络的监督学习算法，其数学基础是**链式求导法则**。BP 学习算法由前向传播和误差反向传播组成：

**前向传播**是输入信号从输入层经隐含层，传向输出层。若输出层得到了期望的输出，则学习算法结束；否则，转至反向传播。

**反向传播**是将误差（样本输出与网络输出之差）按原联接通路反向计算，由**梯度下降法**调整各层节点的权值和阈值，使误差减小。

!!! warning "说明"
    前传和反传、梯度的计算过程需要结合具体的神经网络图像才能较好讲解，因此此处略过。

    对于简单的网络，例如考试题，可以对参数逐个手动计算，也可以利用矩阵形式计算，但是要注意矩阵的求导的方法。

前向传播：

$$
\begin{align*}
\boldsymbol{z}^{(l)} &= \boldsymbol{W}^{(l)}\boldsymbol{a}^{(l-1)} + \boldsymbol{b}^{(l)},&\quad \boldsymbol{a}^{(l)} &= f(\boldsymbol{z}^{(l)}) \\
\frac{\partial \boldsymbol{z}^{(l)}}{\partial \boldsymbol{W}^{(l)}} &= \boldsymbol{a}^{(l-1)},&\quad \frac{\partial \boldsymbol{z}^{(l)}}{\partial \boldsymbol{b}^{(l)}} &= 1
\end{align*}
$$

> 与前面讨论的向量 $\boldsymbol{w}$ 不同，全连接层的权重 $\boldsymbol{W}$ 是一个二维矩阵，因此每一层的输出形如 $\boldsymbol{y}=\boldsymbol{W}\boldsymbol{x}+\boldsymbol{b}$ ，权重不需要做转置。

<br>

反向传播（递推表达式）：

$$
\begin{gather*}
\frac{\partial L}{\partial \boldsymbol{z}^{(l)}}=\delta^{(l)} = (\boldsymbol{W}^{(l+1)})^T \odot f'(\boldsymbol{z}^{(l)}) \delta^{(l+1)} \\
\frac{\partial L}{\partial \boldsymbol{W}^{(l)}} = \delta^{(l)}(\boldsymbol{a}^{(l-1)})^T,\quad \frac{\partial L}{\partial \boldsymbol{b}^{(l)}} = \delta^{(l)}
\end{gather*}
$$

则损失函数对第 $l$ 层权重矩阵 $\boldsymbol{W}^{(l)}$ 中第 $(i,j)$ 个权重 $\boldsymbol{W}_{ij}^{(l)}$ 的梯度为：

$$
\frac{\partial L}{\partial \boldsymbol{W}_{ij}^{(l)}} = \frac{\partial L}{\partial z_i^{(l)}} \cdot \frac{\partial z_i^{(l)}}{\partial \boldsymbol{W}_{ij}^{(l)}} = \delta_i^{(l)} \cdot a_j^{(l-1)}
$$

---

简单总结**使用矩阵求导方法做 BP**，可能会比逐个权重元素计算更简单快捷。这类题目不会出的很难，因为矩阵、向量之间千变万化、过于复杂，而逐个元素计算梯度也很容易计算量爆炸，因此掌握基本的求导公式就能应付大部分题目。

经典的链式求导公式：

$$
\frac{\partial g(f(\boldsymbol{x}))}{\partial \boldsymbol{x}} = \frac{\partial g}{\partial f} \cdot \frac{\partial f}{\partial \boldsymbol{x}}
$$

有 3 种常见的求导：

1. 激活函数求导： $\mathbb{R}^N \to \mathbb{R}^N$ .
2. 矩阵求导。
3. Loss 求导： $\mathbb{R}^N \to \mathbb{R}$ .

对于**激活函数求导**，由于是对向量各个元素作用，因此需要使用 Hadamard 乘积（逐元素相乘），以 Sigmoid 函数为例：

$$
\begin{align*}
\sigma'(x) &= \frac{e^{-x}}{(1 + e^{-x})^2} = \sigma(x)(1 - \sigma(x)),\quad x\in\mathbb{R} \\
\frac{\partial \sigma(\boldsymbol{x})}{\partial \boldsymbol{x}} &= \sigma(\boldsymbol{x}) \odot (1 - \sigma(\boldsymbol{x})) ,\quad \boldsymbol{x}\in\mathbb{R}^N
\end{align*}
$$

对于**矩阵/向量求导**，有以下常用公式：

$$
\begin{gather*}
\dfrac{\partial \boldsymbol{W} \boldsymbol{x}}{\partial \boldsymbol{W}} = \boldsymbol{x}^T ,\quad \dfrac{\partial \boldsymbol{W} \boldsymbol{x}}{\partial \boldsymbol{x}} = \boldsymbol{W}^T \\
\dfrac{\partial L}{\partial \boldsymbol{W}} = \dfrac{\partial L}{\partial (\boldsymbol{W}\boldsymbol{x})} \boldsymbol{x}^T ,\quad \dfrac{\partial L}{\partial \boldsymbol{x}}=\boldsymbol{W}^T \dfrac{\partial L}{\partial (\boldsymbol{W}\boldsymbol{x})} \\
\|\boldsymbol{A}\|_2^2 = \text{trace}(\boldsymbol{A}^T \boldsymbol{A}) \\
\dfrac{\partial \,\text{trace}(\boldsymbol{A}^T \boldsymbol{B})}{\partial \boldsymbol{A}} = \boldsymbol{B} ,\quad \dfrac{\partial \,\text{trace}(\boldsymbol{A} \boldsymbol{B})}{\partial \boldsymbol{B}} = \boldsymbol{A}^T \\
\text{trace}(\boldsymbol{A} \boldsymbol{B} \boldsymbol{C}) = \text{trace}(\boldsymbol{C} \boldsymbol{A} \boldsymbol{B}) = \text{trace}(\boldsymbol{B} \boldsymbol{C} \boldsymbol{A}) \\
\text{trace}(\boldsymbol{A}) = \text{trace}(\boldsymbol{A}^T) \\
\text{trace}(\boldsymbol{A} + \boldsymbol{B}) = \text{trace}(\boldsymbol{A}) + \text{trace}(\boldsymbol{B})
\end{gather*}
$$

标量对矩阵/向量 求导： $\dfrac{\partial \mathbb{R}^{1\times 1}}{\partial \mathbb{R}^{m \times n}}\to \mathbb{R}^{m \times n}$ ，标量对任何矩阵/向量求导，结果的维度都与被求导的矩阵/向量的维度相同。

向量对向量求导： $\dfrac{\partial \mathbb{R}^{m\times 1}}{\partial \mathbb{R}^{n \times 1}}\to \mathbb{R}^{m \times n}$ .

矩阵对矩阵求导： $\dfrac{\partial \mathbb{R}^{m\times n}}{\partial \mathbb{R}^{p \times q}}\to \mathbb{R}^{mn \times pq}$ . 严格来讲“矩阵对矩阵”导数本质是 4 阶张量，但是经常被 reshape 成二维矩阵。这种情况不常见。

!!! tip
    导数和被导数可以同时转置。

<br>

对于 **Loss 求导**，本质是标量对矩阵/向量求导，我们只需要掌握 MSE 和交叉熵两种：

$$
\begin{align*}
\text{MSE:} \quad L(\boldsymbol{y},\boldsymbol{t}) &= \frac{1}{2} \|\boldsymbol{y}-\boldsymbol{t}\|^2 ,\quad
\frac{\partial \|\boldsymbol{y}-\boldsymbol{t}\|^2}{\partial \boldsymbol{y}} = 2(\boldsymbol{y}-\boldsymbol{t}) \\
\text{Cross Entropy:} \quad L(\boldsymbol{y},\boldsymbol{t}) &= -\sum_{i=1}^{N} t_i \log(y_i) ,\quad \frac{\partial L}{\partial \boldsymbol{y}} = -\frac{\boldsymbol{t}}{\boldsymbol{y}}
\end{align*}
$$

上式中 $\boldsymbol{t}$ 为真实值， $\boldsymbol{y}$ 为预测值，它们做**逐元素除法**。

!!! danger "注意"
    如果完全按照上面的方法，那么链式求导各项相乘的时候，很可能会出现维度不匹配的现象，这是正常现象，这时候就需要**随机应变**了。另外，最好不要一步写到位，从后往前一步一步来，每一步使用添加合适的转置等方法，保证中间结果每一步都是正确的。

---

训练方法：梯度下降

神经网络全部参数 $\theta=\{\boldsymbol{W}_1,\cdots,\boldsymbol{b}_1,\cdots\}$ ，训练目标是学习获取使损失函数最小化的网络参数 $\theta^*$ ，参数更新规则为：

$$
w \leftarrow w - \eta \frac{\partial L}{\partial w},\quad b \leftarrow b - \eta \frac{\partial L}{\partial b} ,\quad
\theta \leftarrow \theta - \eta \nabla_\theta L(\theta)
$$

批次梯度下降 BGD（Batch Gradient Descent），所有样本都参与计算梯度： $\theta \leftarrow \theta - \eta \nabla_\theta L(\theta)$ ，缺点是速度慢，数据量大时容易导致内存不足。

随机梯度下降 SGD（Stochastic Gradient Descent），每次只用一个样本参与计算梯度： $\theta \leftarrow \theta - \eta \nabla_\theta L(\theta,x_i,y_i)$ ，优点是速度快，缺点是方差大，损失函数震荡严重。

小批次梯度下降 Mini-batch Gradient Descent（M-SGD），介于 BGD 和 SGD 之间，每次随机选取 $M$ 个样本参与计算梯度： $\theta \leftarrow \theta - \eta\left[\dfrac{1}{M}\displaystyle\sum_{i=1}^M\nabla_\theta L(\theta,x_i,y_i)\right]$ .

3 种方法的比较：

|      特性      |    BGD     |   SGD    |        M-SGD         |
| :------------: | :--------: | :------: | :------------------: |
| 单次迭代样本数 | 整个数据集 | 单个样本 | 整个数据集的一个子集 |
|   算法复杂度   |     高     |    低    |         一般         |
|     时效性     |     低     |   一般   |         一般         |
|     收敛性     |    稳定    |  不稳定  |        较稳定        |

训练流程：

1. 初始化神经网络，初始化设置网络参数。
2. 前向传播，计算梯度，反向传播。
3. 重复，直到梯度的更新非常小。

### 4.3.2 Optimization

梯度下降可能存在的问题：能找到局部最优，但是无法保证找到全局最优。因为可能遇到鞍点（Saddle Point）。改进方法有：

（1）**动量** Momentum

避免随机梯度下降陷入局部最优。维护一个“状态变量” $\beta$ 记录之前的梯度，每次更新参数时不仅考虑当前的梯度，也考虑之前存下来的动量，从而帮助模型越过上面的局部最小值。

$$
\beta_{t+1} = \mu \beta_t - \eta \nabla_{\theta} L(\theta_t), \quad \theta_{t+1} = \theta_t + \beta_{t+1}
$$

其中 $\beta_t$ 即为动量， $\mu\in[0,1]$ 为对应的常系数。参数中梯度方向不大的维度加速更新，同时减少在梯度方向变化大的维度上的更新幅度。

（2）**权重衰减** Weight Decay

在损失函数中加入正则化项（增加对较大系数的惩罚项），防止过拟合。常用的正则化方法有 L1 正则化和 L2 正则化。例如 L2 正则化：

$$
\begin{align*}
\tilde{L}(w)&=L(w) + \frac{1}{2}\lambda \|w\|^2 \\
w &\leftarrow w - \eta \left(\frac{\partial L}{\partial w} + \lambda w\right)
\end{align*}
$$

（3）**AdaGrad**: Adaptive Gradient

利用梯度平方累加和的平方根。不同的参数使用不同的学习率：

$$
\begin{align*}
c_t &= \sum_{j=1}^{t} \left( \nabla_\theta L(\theta_j) \right)^2 \\
\theta_{t+1} &= \theta_t - \eta \frac{\nabla_\theta L(\theta_t)}{\sqrt{c_t} + \varepsilon}
\end{align*}
$$

不同参数的学习率依赖于 $c_t$ . 对低频参数做较大的更新，对高频参数做较小的更新，提升了 SGD 的鲁棒性，对于稀疏数据表现好。

（4）**RMSprop**: Root-Mean-Square Prop

与 AdaGrad 思想类似，利用梯度平方加权取倒数对梯度进行加权。定义 RMS 梯度：

$$
\begin{align*}
s_t &= \gamma s_{t-1} + (1 - \gamma) \left( \nabla_\theta L(\theta_t) \right)^2 \\
\theta_{t+1} &= \theta_t - \eta \frac{\nabla_\theta L(\theta_t)}{\sqrt{s_t} + \varepsilon}
\end{align*}
$$

保证各维度导数在一个量级，减少摆动。[Hinton](https://en.wikipedia.org/wiki/Geoffrey_Hinton) 建议 $\gamma=0.9,\mu=0.001$ .

（5）**Adam**: Adaptive Moment Optimization

Adam 是 RMSprop 和 Momentum 的结合。计算梯度和梯度平方的平滑平均，利用梯度的一阶矩和二阶矩的指数加权平均。这也是目前主流的优化器。

$$
\begin{align*}
v_t &= \beta_1 v_{t-1} + (1 - \beta_1) \nabla_\theta L(\theta_t) \\
s_t &= \beta_2 s_{t-1} + (1 - \beta_2) \left( \nabla_\theta L(\theta_t) \right)^2 \\
\theta_{t+1} &= \theta_t - \eta \frac{v_t}{\sqrt{s_t} + \varepsilon}
\end{align*}
$$

参数设置： $\beta_1 = 0.9$ ， $\beta_2$ 接近于 $1$ ，例如 $0.9999$ .

---

参数初始化方法

一般将偏置初始化为 0，而权重的初始化方法有：

随机初始化：权重初始化为 0，标准差为 $\sigma$ ，则 $w_i^l \sim \mathcal{N}(0, \sigma^2)$

Xavier 初始化：

$$
w_i^l \sim \mathcal{N} \left( 0, \frac{2}{n_l + n_{l-1}} \right) ,\quad w_i^l \sim \mathcal{U} \left[ -\sqrt{\frac{6}{n_l + n_{l-1}}}, \sqrt{\frac{6}{n_l + n_{l-1}}} \right]
$$

Kaiming 初始化：

$$
w_i^l \sim \mathcal{N} \left( 0, \frac{2}{n_l} \right) ,\quad w_i^l \sim \mathcal{U} \left[ -\sqrt{\frac{6}{n_l}}, \sqrt{\frac{6}{n_l}} \right]
$$

---

动态学习率：在训练过程中根据迭代次数逐步调整学习率。在训练初期，离目标损失远，设置较大学习率。训练一段时间后，已到达目标损失附近，距离最优点近，选择较小学习率。

---

梯度消失问题

以 Sigmoid 函数为例：

$$
\begin{gather*}
\sigma'(z) = \frac{e^{-z}}{(1 + e^{-z})^2} = \sigma(z)(1 - \sigma(z)) \in \left(0, \frac{1}{4}\right] \\
\frac{\partial L}{\partial b_1}\leqslant \left(\frac{1}{4}\right)^n w_2 w_3 \cdots w_n \frac{\partial L}{\partial b_n}
\end{gather*}
$$

随着网络深度的加深，幂指数项很快趋于 0，梯度衰减非常严重，梯度消失导致无法继续训练。

如何缓解梯度消失：

1. 分层预训练：Hinton 于2006年提出，利用无监督数据进行分层预训练，再利用有监督数据调整网络参数。
2. ReLU 激活函数。缓解 Sigmoid 和 tanh 激活函数存在较严重的梯度消失问题。
3. Batch Normalization：逐层对数据进行尺度归一化。
4. 辅助损失函数：对浅层神经元输出建立辅助损失函数，直接传递梯度。在 GoogleNet 中使用到。

---

梯度爆炸问题

$$
\frac{\partial L}{\partial b_1} = \sigma'(b_1) w_2 \sigma'(b_2) w_3 \cdots \sigma'(b_{N-1}) w_N \frac{\partial L}{\partial b_N},\quad
\text{if }  \left| w_j \sigma'(b_j)\right| > 1 \text{ then }\frac{\partial L}{\partial b_1} \gg 1
$$

如何缓解梯度爆炸：

1. 重新设计网络模型/更换激活函数。
2. RNN 中使用 ReLU 激活函数可减少梯度爆炸。
3. 使用梯度截断（gradient clipping）。
4. 权重正则化，增加 L1/L2 正则惩罚项。
