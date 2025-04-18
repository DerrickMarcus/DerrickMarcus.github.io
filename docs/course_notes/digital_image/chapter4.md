---
comments: true
---

# 第四章 频域图像增强

## 4.1 傅里叶变换

2-D 变换核

对于 $M\times N$ 二维图像：

$$
F(u,v)=\frac{1}{MN}\sum_{x=0}^{M-1}\sum_{y=0}^{N-1}f(x,y)\exp\left[-j2\pi (\frac{ux}{M}+\frac{vy}{N})\right]
$$

对于 $N\times N$ 二维图像：

$$
F(u,v)=\frac{1}{N}\sum_{x=0}^{N-1}\sum_{y=0}^{N-1}f(x,y)\exp\left(-j2\pi \frac{ux+vy}{N}\right)
$$

1个2-D变换核可分解成2个1-D变换核，对应 x 和 y 两个方向。

平移定理： $f(x-a,y-b)\iff F(u,v)\exp(-j2\pi \dfrac{au+bv}{N})$ 。

旋转定理：$f(x,y)$ 旋转角度 $\theta_0$ ，$F(u,v)$ 也转过相同角度。$F(u,v)$ 旋转角度 $\theta_0$ ，$f(x,y)$ 也转过相同角度。

尺度定理：$f(ax,by)\iff \dfrac{1}{|ab|}F(\dfrac{u}{a},\dfrac{v}{b})$ 。

卷积定理：$f(x,y)*g(x,y)\iff F(u,v)G(u,v),f(x,y)g(x,y)\iff F(u,v)*G(u,v)$ 。

## 4.2 低通和高通滤波

低通滤波：模糊图像中的边缘，滤除高频噪声。

理想低通滤波器有振铃效应。可以使用巴特沃斯滤波器、切比雪夫滤波器等。

高通滤波：可锐化图像，主要提取边缘信息。高通滤波之后图像背景的平均强度减小到接近黑色。

理想高通滤波器有振铃效应。可以使用巴特沃斯滤波器、切比雪夫滤波器等。

高频加强：在高通滤波器函数前乘以一个常数，再加一个偏移，使零频率不被掉。 $H_{hfe}(u,v)=a+bH_{hp}(u,v)$ 。

## 4.3 带通和带阻滤波

带阻滤波器：可消除周期性噪声。可用 N 阶巴特沃斯带阻滤波器实现。

带通滤波器：可提取周期性噪声。直接用“1-带阻滤波器”。

## 4.4 同态滤波

见图片：

![digital_image_lesson7_img1](https://cdn.jsdelivr.net/gh/DerrickMarcus/picgo_image/images/digital_image_lesson7_img1.png)
