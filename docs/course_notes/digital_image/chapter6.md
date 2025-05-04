---
comments: true
---

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

## 6.3 图像有约束复原
