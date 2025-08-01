# 5 CNN 卷积神经网络

[Convolutional neural network - Wikipedia](https://en.wikipedia.org/wiki/Convolutional_neural_network)

## 5.1 Basic Unit

为什么需要卷积神经网络？

1. 局部感知：对于某些特定的问题，例如人脸识别，输入图像被归一化至特定尺寸，在归一化大小受限的条件下，网络参数可接受。
2. 权值共享：如果局部滤波器的权重是相同的，即不同空间位置的神经元共享权值，参数数目可以进一步减少。卷积操作：学习卷积核参数。

CNN 的基本单元：卷积层、池化层、激活函数、Batch Normalization。

在卷积层，每个神经元的输入为前层输出的局部感受区域，通过卷积计算提取该局部区域的特征。通过卷积层的运算，提取到的局部特征之间的相互位置关系同时被保留。

在池化层，利用图像局部相关性原理，对卷积层输出的局部特征图进行下抽样，在保留有用信息的基础上减少数据处理量。注意，<span style="color:red">池化层没有可学习参数</span>！

### 5.1.1 Convolution Layer

输入数据： $C_{in}\times H_{in} \times W_{in}$ ，其中 $C_{in}$ 为输入通道数，对于输入层一般为 RGB 三通道， $H_{in}$ 为输入高度， $W_{in}$ 为输入宽度。

卷积滤波器： $C_{in} \times K_h \times K_w$ ，其中 $K_h,K_w$ 分别为卷积核的高度和宽度。

卷积层： $C_{out} \times C_{in} \times K_h \times K_w$ ，等价于多个卷积核叠加，卷积核的数目即为输出通道数。**偏置** 为 $C_{out}$ 维的向量。

> 默认情况下，我们使用的都是全通道卷积，即 卷积核通道数与输入数据通道数一致。

当同时输入 $N$ 个样本，即 $\text{batch\_size}=N$ ，输入数据维度为 $N \times C_{in} \times H_{in} \times W_{in}$ ，卷积核数目 $C_{out}$ ，卷积步长 $\text{stride}=S$ ，填充 $\text{padding}=P$ 时，输出数据维度为： $N \times C_{out} \times H_{out} \times W_{out}$ ，其中：

$$
H_{out}=\left \lfloor \frac{H_{in} - K_h + 2P}{S} + 1 \right \rfloor,\;
W_{out}=\left \lfloor \frac{W_{in} - K_w + 2P}{S} + 1 \right \rfloor
$$

注意计算结果<span style="color:red">向下取整</span>！！！

整个卷积层的参数量（**考虑偏置**）为 $C_{out} (C_{in} K_h  K_w + 1)$ . 若不考虑偏置项，则参数量简化为 $C_{out} C_{in} K_h K_w$ .

乘法次数 FLOPs（考虑偏置）为 $(K_w K_h C_{in} + 1) C_{out} H_{out} W_{out} N$ . 若不考虑偏置项，则 FLOPs 简化为 $K_w K_h C_{in} C_{out} H_{out} W_{out} N$ .

加法次数 FLOPs（考虑偏置）为 $(K_w K_h C_{in} - 1) C_{out} H_{out} W_{out} N$ . 若不考虑偏置项，则 FLOPs 简化为 $K_w K_h C_{in} C_{out} H_{out} W_{out} N$ .

（边界延拓）若卷积核尺寸为 $K\times K$ ，步长为 $S=1$ ，若要输出特征图与输入特征图长度宽度尺寸一致，则填充大小为 $P=\dfrac{K-1}{2}$ . 同时应注意到，只有设置步长 $S=1$ 的时候，填充大小 $P$ 才只与卷积核大小 $K$ 有关，否则还将和图像原本尺寸 $H_{in}, W_{in}$ 有关。

!!! question "卷积核的尺寸为什么是奇数？"
    一是保护位置信息，保证锚点刚好在中间，方便以卷积核中心为标准进行滑动卷积，避免了位置信息发生偏移。二是保证 padding 时，图像的两边依然对称，因为 $(K-1)/2$ 必须为整数 。

---

感受野：神经网络每一层输出特征图上的像素点对应于原始输入图的映射区域大小。

单层卷积层的感受野与卷积核大小、卷积步长有关。多层卷积层的感受野采用回溯方法估计。

---

卷积层的变体

**空洞卷积** Dilated convolution

卷积核中注入空洞，目标是增加感受野，同时不丢失分辨率（池化也能扩大感受野但是丢失分辨率），也能够减小计算量。相比正常的卷积，空洞卷积多出一个 dilation rate 参数，如 PyTorch 中 `torch.nn.Conv2d()` 的 `dilation` 参数。

空洞卷积的 **感受野** 为 $(K-1)D+1$ ，其中 $K$ 为卷积核尺寸， $D$ 为空洞率。也就是说，一个尺寸 $K$ 的、空洞率为 $D$ 的卷积核，实际上等效于一个尺寸为 $(K-1)D+1$ 的卷积核。例如：

卷积核尺寸为 $5\times 5$ ，空洞率为 $4$ 的空洞卷积，其感受野为 $(5-1) \cdot 4 + 1 = 17$ ，等效于一个 $17\times 17$ 的卷积核。然后输出特征图的尺寸，仍然按照上面常规卷积的计算公式即可。

**分组卷积** Group convolution

最早在 AlexNet 中出现。在深度上进行划分，即某几个通道编为一组，相应的，卷积核深度等比例缩小而大小不变。利用每组的卷积核同它们对应组内的输入数据卷积，得到了输出数据以后，再连接组合，分组后并行计算。

**深度可分离卷积** Depthwise separable convolution

每一个通道用一个 filter 卷积之后得到对应一个通道的输出，然后再进行信息的融合。

深度可分离卷积比普通卷积减少了所需要的参数。重要的是深度可分离卷积将以往普通卷积操作同时考虑通道和区域改变成，卷积先只考虑区域，然后再考虑通道，实现了通道和区域的分离。

其中使用到了 $1\times 1$ 卷积核滤波器，其作用为：对特征图降维或升维，实现信息的跨通道整合和交互，大幅增加非线性特性，提升网络的表达能力。

**可变形卷积** Deformable convolution

在感受野中引入了偏移量，而且偏移量是可学习参数，这样卷积核不再是传统的方形，而可以与物体的实际形状更贴近。

$$
\begin{align}
y(\mathbf{p}_0) &= \sum_{\mathbf{p}_n \in \mathcal{R}} \mathbf{w}(\mathbf{p}_n) \cdot \mathbf{x}(\mathbf{p}_0 + \mathbf{p}_n) \\
y(\mathbf{p}_0) &= \sum_{\mathbf{p}_n \in \mathcal{R}} \mathbf{w}(\mathbf{p}_n) \cdot \mathbf{x}(\mathbf{p}_0 + \mathbf{p}_n + \Delta\mathbf{p}_n)
\end{align}
$$

其中 (1) 为传统卷积，(2) 为可变形卷积， $\Delta\mathbf{p}_n$ 为偏移量。

### 5.1.2 Pooling Layer

池化层/下采样层 Pooling layer。作用是：

1. 保持主要信息，剔除次要信息。
2. 增加感受野，精炼特征图尺寸。
3. 仅对特征图单通道进行下采样，不考虑通道间关系。

最常用的池化方式是 **最大池化 Max pooling** 和 **平均池化 Average pooling**。

Pytorch 中的池化层为 `torch.nn.MaxPool2d()` 和 `torch.nn.AvgPool2d()` 。

无论输入特征图尺寸、池化核的大小、池化的步长为多少，**池化层的参数量都为0**。

池化层的反向传播：考虑池化层时，对于最大值池化，误差会 **回传到当初最大值**的 位置上，而其它位置对应误差都是 0；对于平均值池化，误差会 **平均回传到原始的几个位置** 上。

### 5.1.3 Dropout

在训练过程中每次迭代时，将各层的输出节点以一定概率随机置为0。

训练阶段：根据 Bernoulli 分布随机生成随机数 $u$ ，并基于如下准则判断该节点参数是否更新：

$$
y_{\text{train}} =
\begin{cases}
\dfrac{x}{1-p} & \text{if } u > p \\
0 & \text{otherwise}
\end{cases}
$$

通过隐式的模型集成来提升性能；通过限制模型容量提高模型泛化能力。

### 5.1.4 Batch Normalization

逐步尺度归一化，避免梯度消失和梯度溢出。可以 5x~20x 加速收敛，提升泛化能力。也能减少对初始化依赖。

方法：对特征图按每一批次样本计算均值和方差，对各维度归一化处理得到标准正态分布的数据：

$$
y = \frac{x - \mathbb{E}[x]}{\sqrt{\text{Var}[x]} + \varepsilon} * \gamma + \beta
$$

训练时，每个批次计算所用的均值和方差采用当前批次的统计量。

测试时，每个批次计算所用的均值和方差采用训练完成时得到的统计量。

批量归一化操作可以看作一个特殊的网络层，加在每一层非线性激活函数之前。可以加速训练过程的收敛速度，**防止出现梯度消失**问题，也可看作一种**正则化**方法。

Pytorch 中的 Batch Normalization 为 `torch.nn.BatchNorm2d()` 。

## 5.2 Typical CNN

LeNet: 卷积神经网络的开山之作

- C1 卷积层: 核大小 5x5，输出特征图大小为 6x28x28;
- S2 下采样层: 输出特征图大小为 6x14x14;
- C3 卷积层: 局部组合，输出特征图大小为 16x10x10;
- S4 下采样层: 提取得到16个 5x5 的特征图;
- C5 卷积层: 卷积核大小为 1x1，提取120个特征图;
- F6 是全连接层;
- 输出层由径向基函数(Radial Basis Function, RBF)单元组成，每个 RBF 单元计算输入向量和参数向量之间的欧式距离。输入离参数向量越远，RBF 输出的越大。

<br>

AlexNet

- 前五层的卷积层，剩余三层的全连接层；
- ReLU 非线性激活函数；
- 数据增强 (Data augmentation)；
- Dropout；
- 训练：双 GPU 并行，基于 ImageNet 百万量级数据；
- 输入 RGB 三通道图像归一化大小为 227x227x3;

<br>

VGGNet

- 卷积核大小固定为 3x3，利用多个叠加的最小 3x3 的卷积核的卷积层，来模仿 5x5, 7x7 的卷积层；
- 通过降低卷积核的大小来减少网络参数，提升网络的深度以提升深度网络的性能。
- 典型版本：VGG16 和 VGG19。两个网络的相同点是最后都有三层的全连接，且层与层之间均有 Max Pooling。

!!! note
    2个 3x3 的卷积核叠加，感受野与1个 5x5 的卷积核相同，但参数量更少。例如前者是 $18C^2$ ，后者是 $25C^2$ . 同理，3个 3x3 的卷积核叠加，感受野与1个 7x7 的卷积核相同，但参数量更少。例如前者是 $27C^2$ ，后者是 $49C^2$ .

<br>

GoogleNet

- 使用 Inception 结构：利用 1x1, 3x3, 5x5 卷积以及 3x3 池化后的特征图进行深度上的合并，拓宽网络的广度，增加网络的尺度适应性，称为多尺度感受野(multiple receptive fields)。
- 带特征降维的 Inception 结构。1x1 卷积：特征降维与升维。在 3x3, 5x5 卷积层之前引入 1x1 卷积，在 3x3 max pooling 层之后引入 1x1 卷积

<br>

Inception-Net

- Inception-V2 增加了 BN 层，减少每层数据的内部方差漂移，利用两个 3x3 卷积滤波器代替 5x5 的滤波器以减少参数；
- Inception-V3 通过分解增强了网络的非线性；
- Inception-V4 结合了残差网络的设计理念。

<br>

ResNet(深度残差网络)

- 针对问题：信息传递受阻；
- 解决方案：普通浅层网络中引入“快捷连接”(shortcut connection), 转变成残差网络实现；
- 残差网络将输入作为参照，学习输入与输出之间的残差函数，输出表示为残差函数与输入之和；
- 残差函数更容易优化，提升网络的深度；
- 典型结构包括 ResNet34, ResNet50, ResNet101, ResNet152 等。

!!! note "为什么 ResNet 可以缓解梯度消失/网络可以更深？"
    残差块通过跳跃连接，使得梯度能够更好的回传，解决梯度消失和梯度爆炸问题，允许网络直接学习残差映射 $H(x)-x$ （学习输出与输入之间的差异），结合跳跃连接就能完成 $x\to (H(x)-x)+x$ 的映射，而非学习一个恒等映射，提升深层网络的训练稳定性。这样，网络可以更深，因为每一层都可以直接访问到输入的原始信息，从而更容易学习到有效的特征。而且 Loss 更平滑，从而有利于收敛到最优解。

<br>

DenseNet

- 稠密连接：每层之前层的输出为输入，对于有 $L$ 层的传统网络，一共有 $L(L+1)/2$ 个连接。
- 尽量缩短前层和层之间的连接，最大化信息的流动。
- 有效解决梯度消失问题；强化特征传播；支持特征重用。大幅度减少参数数量。

SENet：通道注意力机制

SKNet(selective kernels)：卷积核选择。

MobileNet

- 基本单元：深度可分离卷积；
- 首先是采用 depthwise convolution 对不同输入通道分别进行卷积，然后采用 pointwise convolution 将上面的输出再进行结合；
- 整体效果和标准卷积差不多，可大大减少计算量和模型参数量。

<br>

ShuffleNet

- 利用分组卷积，但分组卷积会造成输出通道只和某些输入通道有关；
- 利用 channel shuffle 解决全局信息流通不畅，网络表达能力不足的问题。

## 5.3 Application

图像分类 image classification

1. 准备数据集，包括训练集和测试集。
2. 设计卷积神经网络 CNN。
3. 确定损失函数，选择优化策略。
4. 在训练集上训练学习神经网络参数。
5. 在测试集上测试神经网络性能。

目标检测 object detection （属于 **多任务学习** Multi-task learning）

语义分割 semantic segmentation

人脸认证/识别 face verification/recognition （属于 **度量学习** Metric learning）
