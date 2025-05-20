# 第六章 图像复原

## 6.1 图像退化和复原模型

图像退化。退化原因：传感器内部噪声，摄像机未聚焦，物体与镜头之间相对移动，胶片的非线性和几何畸变。退化实例：运动模糊，几何畸变，大气湍流影响。

图像复原：利用退化现象的某种先验知识，复原被退化的模糊图像。

图像复原和图像增强的联系，见图片：

![第十周课件上课_27](https://cdn.jsdelivr.net/gh/DerrickMarcus/picgo_image/images/第十周课件上课_27.png)

退化过程原理：（退化可以理解为使图像变形、失真等等操作，不同于加噪声。）

![第十周课件上课_29](https://cdn.jsdelivr.net/gh/DerrickMarcus/picgo_image/images/第十周课件上课_29.png)

图像复原中，往往用线性系统近似，模拟非线性系统模型。

图像超分辨率(Image Super Resolution)：由一幅低分辨率图像或图像序列恢复出高分辨率图像。方法：基于插值，基于重建，基于学习。

补充常见插值方法：

1. 最近邻插值：选取离待插值点最近的点的像素值作为待插值点的像素值；会出现锯齿效应。
2. 双线性插值：在x和y两个方向上分别进行线性插值；不会出现锯齿效应，但比较模糊。
3. 双三次插值：通过最近的16个采样点的加权平均得到，需要使用两个多项式插值三次函数，每个方向使用一个：计算量比较大，但效果相对较好。

退化系统模型 H 具有性质：相加性、一致性、线性、位置/空间不变性（线性系统在图像任意空间位置的响应只与在该空间位置的输入值有关，而与空间位置本身无关）。

一维退化模型的离散计算（不考虑噪声）：

![第十周课件上课_43](https://cdn.jsdelivr.net/gh/DerrickMarcus/picgo_image/images/第十周课件上课_43.png)

注意 $f(x),g(x)$ 都要补0扩展为 $f_e(x),g_e(x)$ 。

推广到二维L：

![第十周课件上课_45](https://cdn.jsdelivr.net/gh/DerrickMarcus/picgo_image/images/第十周课件上课_45.png)

![第十周课件上课_46](https://cdn.jsdelivr.net/gh/DerrickMarcus/picgo_image/images/第十周课件上课_46.png)

## 6.2 图像无约束复原

![第十周课件上课_48](https://cdn.jsdelivr.net/gh/DerrickMarcus/picgo_image/images/第十周课件上课_48.png)

![第十周课件上课_49](https://cdn.jsdelivr.net/gh/DerrickMarcus/picgo_image/images/第十周课件上课_49.png)

![第十周课件上课_53](https://cdn.jsdelivr.net/gh/DerrickMarcus/picgo_image/images/第十周课件上课_53.png)

逆滤波（频域复原方法）：不考虑噪声时，已知退化图像的傅里叶变换 $G(u,v)$ 和“滤波”传递函数 $H(u,v)$ ，可得原图像傅里叶变换 $\hat{F}(u,v)$ ，再求反变换，即可得原图像的估计 $\hat{f}(x,y)$ 。

![第十周课件上课_58](https://cdn.jsdelivr.net/gh/DerrickMarcus/picgo_image/images/第十周课件上课_58.png)

![第十周课件上课_59](https://cdn.jsdelivr.net/gh/DerrickMarcus/picgo_image/images/第十周课件上课_59.png)

![第十周课件上课_60](https://cdn.jsdelivr.net/gh/DerrickMarcus/picgo_image/images/第十周课件上课_60.png)

![第十周课件上课_61](https://cdn.jsdelivr.net/gh/DerrickMarcus/picgo_image/images/第十周课件上课_61.png)

!!! note "补充"
    图像超分辨率重建：给定的低分辨率图像通过特定的算法恢复成相应的高分辨率图像，是一种欠定问题，旨在克服或补偿由于图像采集系统或采集环境本身的限制，导致的成像图像模糊、质量低下、感兴趣区域不显著等问题。

    传统图像超分辨率通过插值来实现，例如最近邻插值、双线性插值、双三次插值等，通过邻域的像素信息拟合未知的像素，这种方式假设像素变化是连续、平滑的，因此在处理边缘、纹理处时效果差。

    通用超分辨率：根据低分辨率图像重建得到对应的高分辨率图像；利用局部纹理或者内容信息恢复图像细节；经典深度学习方法有 SRCNN，EDSR，SRGAN 等。
    低分辨率人脸图像重建与识别系统。

## 6.3 图像有约束复原

运动模糊的建模估计，退化函数为：

$$
H(u,v)=\frac{T}{\pi(uc+vb)}\sin[\pi(uc+vb)]e^{-j\pi(uc+vb)}
$$

其中 $T$ 为曝光时间（运动模糊的时间），$c$ 和 $b$ 分别为 x 方向和 y 方向的位移。

在无约束图像复原中，只要已知退化函数 $H(u,v)$ 即可进行图像复原。但是实际图像存在噪声！

![第十二周课件上课_24](https://cdn.jsdelivr.net/gh/DerrickMarcus/picgo-image/images/第十二周课件上课_24.png)

![第十二周课件上课_25](https://cdn.jsdelivr.net/gh/DerrickMarcus/picgo-image/images/第十二周课件上课_25.png)

### 维纳滤波

复原图像的傅里叶变换的估计为：

$$
\hat{F}(u,v) = \left[ \frac{1}{H(u,v)} \frac{|H(u,v)|^2}{|H(u,v)|^2 + s[S_n(u,v)/S_f(u,v)]} \right] G(u,v)
$$

实际中 $S_n(u,v),S_f(u,v)$ 很难已知，常做近似：$s=1$，将噪声用白噪声近似，$S_n(u,v)/S_f(u,v)$ 代表“噪声与信号的功率密度比”为常数 $K$。

$$
\hat{F}(u,v) = \left[ \frac{1}{H(u,v)} \frac{|H(u,v)|^2}{|H(u,v)|^2 + K} \right] G(u,v)
=\left[ \frac{H^*(u,v)}{|H(u,v)|^2 + K} \right] G(u,v)
$$

!!! quote
    有噪声时维纳滤波效果比逆滤波强。

    维纳滤波，假定原始图像f和噪声n是随机变量，其图像集形成平稳随机过程，并需要已知未退化图像和噪声的功率谱。虽然可以近似，但常数 K 不易估计。即：维纳滤波器在图像统计最小二乘误差意义下最优，但对某一具体图像而言不一定最优。

### 最小平方法

有约束最小平方复原，只要求噪声方差和均值信息就可对给定图像复原出最优结果。

$$
\hat{F}(u,v) = \left[ \frac{H(u,v)^*}{|H(u,v)|^2 + s|P(u,v)|^2} \right] G(u,v)
$$

其中 $P(u,v)$ 为模板卷积核 $p(x,y)$ 的傅里叶变换。

!!! note "两种有约束复原方法总结"
    评价标准：维纳滤波在图像复原误差统计平均意义下最优；最小平方在最大平滑准则下最优。

    应用场景：维纳滤波中，图像和噪声都属于随机过程，且已知噪声和未退化图像功率谱或用常数替代（该值不易找到）。最小平方中以像素间平滑准则为基础，只需要知道噪声的均值和方差或者迭代选择标量参数s。
