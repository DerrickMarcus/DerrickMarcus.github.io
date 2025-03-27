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

平移定理： $f(x-a,y-b)\iff F(u,v)\exp(-j2\pi (au+bv)/N)$ 。

旋转定理：$f(x,y)$ 旋转角度 $\theta_0$ ，$F(u,v)$ 也转过相同角度。$F(u,v)$ 旋转角度 $\theta_0$ ，$f(x,y)$ 也转过相同角度。

尺度定理：$f(ax,by)\iff \dfrac{1}{|ab|}F(\dfrac{u}{a},\dfrac{v}{b})$ 。

卷积定理：$f(x,y)*g(x,y)\iff F(u,v)G(u,v),f(x,y)g(x,y)\iff F(u,v)*G(u,v)$ 。

## 4.2 低通和高通滤波

低通滤波：模糊图像中的边缘，滤除高频噪声。

## 4.3 带通和带阻滤波

## 4.4 同态滤波
