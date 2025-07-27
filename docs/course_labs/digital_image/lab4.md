# Week 4 Assignment

![202506121150091](https://cdn.jsdelivr.net/gh/DerrickMarcus/picgo-image/images/202506121150091.png)

题目要求：对给出的频谱值数据进行对数变换，并比较对数变换前后频谱图的视觉效果变化。

---

编写 MATLAB 代码：首先载入 `./work.mat` 得到变量 `image` ，其类型是 `complex double` ，取幅度值得到 `spectrum` 。然后进行对数变换：

```matlab
c = 1;
log_spectrum = c * log(1 + spectrum);
```

最后将处理前后的两幅图片进行对比，同时使用灰度映射和添加颜色条增强可视化效果。

源代码见 `./main.m` ，运行结果图片见 `./result.jpg` 。

![result](https://cdn.jsdelivr.net/gh/DerrickMarcus/picgo_image/images/result.jpg)

由于对数函数始终在 $t = s$ 上方，所以低灰度值扩展，高灰度值压缩，处理后图片整体变亮，部分原本的灰度值低的区域明显变亮。

附 MATLAB 源代码：

```matlab
load('./work.mat');

spectrum = abs(image);
c = 1;
log_spectrum = c * log(1 + spectrum);

figure;
subplot(1, 2, 1);
imagesc(spectrum);
title('Original specturm');
xlabel('X');
ylabel('Y');
colormap(gray); % 使用灰度映射
colorbar; % 添加颜色条

subplot(1, 2, 2);
imagesc(log_spectrum);
title('Spectrum after log transformation');
xlabel('X');
ylabel('Y');
colormap(gray);
colorbar;

saveas(gcf, 'result.jpg');
```
