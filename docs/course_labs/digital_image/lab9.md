# 数字图像处理 第9周作业

题目要求：

一、对所给图像（img1.jpg）用迭代确定阈值、局部阈值法和大津法进行分割并分析。尝试以上三种分割方法，运用膨胀腐蚀等操作对分割结果进行优化或其它探索。

![202506121159248](https://cdn.jsdelivr.net/gh/DerrickMarcus/picgo-image/images/202506121159248.png)

二、对下面的两张灰度图像（img2-1.jpg, img2-2.jpg ）采用不同的变换函数进行伪彩色增强，并分析图像效果和伪彩色增强的作用。

![202506121200432](https://cdn.jsdelivr.net/gh/DerrickMarcus/picgo-image/images/202506121200432.png)

## Task 1

该部分源代码为 `./task1.py` 。

待分割的原图像如下：

![digital_image_hw9_img1](https://cdn.jsdelivr.net/gh/DerrickMarcus/picgo_image/images/digital_image_hw9_img1.jpg)

分别采用迭代阈值法、局部阈值法、大津算法进行图像分割。效果如下：

![digital_image_hw9_img1_result_1](https://cdn.jsdelivr.net/gh/DerrickMarcus/picgo_image/images/digital_image_hw9_img1_result_1.jpg)

整体上局部阈值法效果略好于迭代阈值法、大津算法，文字辨识度更好、更加均匀，但是由于分块区域最终选定阈值不同，图像中出现了明显的孤立黑点，类似于椒盐噪声。而迭代阈值法、大津算法的结果中，字母有粗有细，比较不均匀。

若对风格后的图像进行闭运算 `cv2.MORPH_CLSOE` ，先腐蚀后膨胀，可以减小图像中不连续的点和突刺，但是最终效果并不理想，没有比腐蚀膨胀之前的效果更好，因为经过图像分割一些字母之后可能会被分成两个小部分，这些小部分在第一步腐蚀的时候就会被消除，导致最后的效果图中文字变瘦、部分出现残缺、像素点减少。不适合使用开运算 `cv2.MORPH_OPEN` ，因为图像中原本字体较小，一些字母比如 a,e,o 中间有小孔，先膨胀后腐蚀会填满这些小孔和缺口，让字母变成一团黑，难以辨认。

![digital_image_hw9_img1_result_2](https://cdn.jsdelivr.net/gh/DerrickMarcus/picgo_image/images/digital_image_hw9_img1_result_2.jpg)

## Task 2

该部分源代码为 `./task2.py` 。

第一张——卫星云图：

![digital_image_hw9_img21](https://cdn.jsdelivr.net/gh/DerrickMarcus/picgo_image/images/digital_image_hw9_img21.jpg)

第二张——焊接 X 光：

![digital_image_hw9_img22](https://cdn.jsdelivr.net/gh/DerrickMarcus/picgo_image/images/digital_image_hw9_img22.jpg)

使用折线型的变换函数，伪彩色增强效果如下：

![digital_image_hw9_img21_line](https://cdn.jsdelivr.net/gh/DerrickMarcus/picgo_image/images/digital_image_hw9_img21_line.jpg)

![digital_image_hw9_img22_line](https://cdn.jsdelivr.net/gh/DerrickMarcus/picgo_image/images/digital_image_hw9_img22_line.jpg)

对于卫星云图，云部分亮度较高，映射之后红色通道最高，几乎为255，蓝色次之，绿色通道几乎为0，因此彩色图中云部分呈现红色和紫色。大陆和海洋部分亮度低，映射之后主要为绿色，因此彩色图中大陆和海洋呈现绿色，海洋亮度更低因此绿色更“纯正”。

对于焊接 X 光，裂缝部分亮度高，映射之后是明显的红色，背景部分亮度稍低，蓝色和红色合成紫色。右侧圆圈部分最暗，映射之后为青色和蓝色，说明蓝色和绿色通道像素值高，对应原图灰度级约在64以下。

使用正弦函数型的变换函数，固定周期 $T$ ，改变偏置 $\delta$ ，伪彩色增强效果如下：

![digital_image_hw9_img21_sine_1](https://cdn.jsdelivr.net/gh/DerrickMarcus/picgo_image/images/digital_image_hw9_img21_sine_1.jpg)

![digital_image_hw9_img22_sine_1](https://cdn.jsdelivr.net/gh/DerrickMarcus/picgo_image/images/digital_image_hw9_img22_sine_1.jpg)

对于卫星云图，第一种结果中以红绿蓝三种颜色为主，第二种结果以橙色和深蓝色为主，对比更加明显，能更清晰地看出云位于深蓝色部分，橙色是陆地和海洋。

对于焊接 X 光，右图相比于左图，中间绿色部分的蓝色分量增加，出现了一些黄色。

绘制出各个通道的变换函数如下：

![digital_image_hw9_sine_plot_1](https://cdn.jsdelivr.net/gh/DerrickMarcus/picgo_image/images/digital_image_hw9_sine_plot_1.png)

在一定范围内，偏置越大，各个通道变换函数达到峰值的点间距也越大，也就是峰值恰好是错开的，当其中一个通道达到峰值的时候，其他通道像素值较低，因此变换图中以红绿蓝为主。而偏置较小时，通常情况下都有两个像素值相近的通道，加和成另外一种颜色，因此变换图中出现了橙色、深蓝色等，看起来颜色更鲜艳丰富。

使用正弦函数型的变换函数，固定偏置 $\delta=T/3$ ，改变周期 $T$ ，伪彩色增强效果如下：

![digital_image_hw9_img21_sine_2](https://cdn.jsdelivr.net/gh/DerrickMarcus/picgo_image/images/digital_image_hw9_img21_sine_2.jpg)

![digital_image_hw9_img22_sine_2](https://cdn.jsdelivr.net/gh/DerrickMarcus/picgo_image/images/digital_image_hw9_img22_sine_2.jpg)

对于卫星云图，从左到右随着周期减小，图像看起来更破碎，色块之间连续性减弱，也难以直观看出云层所在位置。

对于焊接 X 光，从左到右随着周期减小，同样是图像更加破碎，在上下带状区域的过渡部分，出现了红黄蓝交替的现象，这是因为周期减小后，较为微小的灰度级变化就对应一个周期，经过红黄蓝颜色峰值，因此出现类似于彩虹形状的边界。

绘制出各个通道的变换函数如下：

![digital_image_hw9_sine_plot_2](https://cdn.jsdelivr.net/gh/DerrickMarcus/picgo_image/images/digital_image_hw9_sine_plot_2.png)

周期改变时减小，变换函数振荡次数增大，也就是灰度图中不同的亮度对应到彩色图中可能是相同的颜色，对应到同一亮度的灰度级个数随着周期减小而增多，同时相邻灰度级对应的色彩差异更大，对比更明显。反映在变换效果上，直观感受就是图像更加破碎，图象被分成更多个更细小的色块，对比度更加明显。

## 源代码

`task1.py` :

```py
import cv2
import matplotlib.pyplot as plt
import numpy as np


def iterative_threshold(img, eps=1):
    prev_T = img.mean()
    while True:
        G1 = img[img > prev_T]
        G2 = img[img <= prev_T]
        if len(G1) == 0 or len(G2) == 0:
            break
        m1 = G1.mean()
        m2 = G2.mean()
        T = (m1 + m2) / 2
        if abs(prev_T - T) < eps:
            break
        prev_T = T
    _, binary = cv2.threshold(img, prev_T, 255, cv2.THRESH_BINARY)
    print(f"Iterative threshold: {prev_T}")
    return binary


def main():
    image = cv2.imread("./digital_image_hw9_img1.jpg", cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Image read failed. Please check the file path.")
        return

    # 迭代阈值法
    iterative_result = iterative_threshold(image)

    # 局部阈值法（自适应阈值）
    local_result = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=11, C=2
    )

    # 大津法
    thres, otsu_result = cv2.threshold(
        image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    print(f"Otsu's threshold: {thres}")

    titles = [
        "Original Image",
        "Iterative threshold",
        "Local threshold",
        "Otsu Algorithm",
    ]
    images = [image, iterative_result, local_result, otsu_result]

    plt.figure(figsize=(12, 8))
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.imshow(images[i], cmap="gray")
        plt.title(titles[i], color="red")
        plt.axis("off")
        # if i != 0:
        #     cv2.imwrite(f"./digital_image_hw9_img1_segment_{i}.jpg", images[i])

    plt.tight_layout()
    plt.savefig("./digital_image_hw9_img1_result_1.jpg")
    plt.show()

    # 膨胀与腐蚀
    kernel = np.ones((3, 3), dtype=np.uint8)
    iterative_result = cv2.morphologyEx(iterative_result, cv2.MORPH_CLOSE, kernel)
    local_result = cv2.morphologyEx(local_result, cv2.MORPH_CLOSE, kernel)
    otsu_result = cv2.morphologyEx(otsu_result, cv2.MORPH_CLOSE, kernel)

    images = [image, iterative_result, local_result, otsu_result]

    plt.figure(figsize=(12, 8))
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.imshow(images[i], cmap="gray")
        plt.title(titles[i], color="red")
        plt.axis("off")
        # if i != 0:
        #     cv2.imwrite(f"./digital_image_hw9_img1_segment_{i + 3}.jpg", images[i])

    plt.tight_layout()
    plt.savefig("./digital_image_hw9_img1_result_2.jpg")
    plt.show()


if __name__ == "__main__":
    main()

```

`task2.py` :

```py
import cv2
import matplotlib.pyplot as plt
import numpy as np


def pseudo_color_line(gray_img):
    def f_r(p):
        if p < 64:
            return 0
        elif p > 128:
            return 255
        else:
            return 255 / 64 * (p - 64)

    def f_g(p):
        if p < 64:
            return 255
        elif p > 128:
            return 0
        else:
            return 255 / 64 * (128 - p)

    def f_b(p):
        if p < 64:
            return 255 / 64 * p
        elif p > 192:
            return 255 / 63 * (255 - p)
        else:
            return 255

    # 构建 LUT 映射
    lut_r = np.array([np.clip(f_r(i), 0, 255) for i in range(256)], dtype=np.uint8)
    lut_g = np.array([np.clip(f_g(i), 0, 255) for i in range(256)], dtype=np.uint8)
    lut_b = np.array([np.clip(f_b(i), 0, 255) for i in range(256)], dtype=np.uint8)

    r = cv2.LUT(gray_img, lut_r)
    g = cv2.LUT(gray_img, lut_g)
    b = cv2.LUT(gray_img, lut_b)
    color_img = cv2.merge([b, g, r])
    return color_img


def pseudo_color_sine(gray_img, T, delta):
    def f_r(p):
        return (1 + np.sin(2 * np.pi / T * p)) * 255 / 2

    def f_g(p):
        return (1 + np.sin(2 * np.pi / T * (p - delta))) * 255 / 2

    def f_b(p):
        return (1 + np.sin(2 * np.pi / T * (p - 2 * delta))) * 255 / 2

    lut_r = np.array([np.clip(f_r(i), 0, 255) for i in range(256)], dtype=np.uint8)
    lut_g = np.array([np.clip(f_g(i), 0, 255) for i in range(256)], dtype=np.uint8)
    lut_b = np.array([np.clip(f_b(i), 0, 255) for i in range(256)], dtype=np.uint8)

    r = cv2.LUT(gray_img, lut_r)
    g = cv2.LUT(gray_img, lut_g)
    b = cv2.LUT(gray_img, lut_b)
    color_img = cv2.merge([b, g, r])
    return color_img


def main(img_path):
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Image read failed. Please check the file path.")
        return

    # 折线变换
    line_result = pseudo_color_line(image)
    plt.figure(figsize=(5, 5))
    plt.imshow(cv2.cvtColor(line_result, cv2.COLOR_BGR2RGB))
    plt.title("Pseudo color transform: line")
    plt.axis("off")
    plt.savefig(img_path.rsplit(".", 1)[0] + "_line" + "." + img_path.rsplit(".", 1)[1])
    plt.show()

    # 正弦函数变换
    # delta 大小对伪彩色增强的影响
    sine_result_1 = pseudo_color_sine(image, T=255, delta=255 / 3)
    sine_result_2 = pseudo_color_sine(image, T=255, delta=255 / 6)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(sine_result_1, cv2.COLOR_BGR2RGB))
    plt.title("Pseudo color transform: sine\nT=255, delta=255/3")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(sine_result_2, cv2.COLOR_BGR2RGB))
    plt.title("Pseudo color transform: sine\nT=255, delta=255/6")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(
        img_path.rsplit(".", 1)[0] + "_sine_1" + "." + img_path.rsplit(".", 1)[1]
    )
    plt.show()

    # T 大小对伪彩色增强的影响
    sine_result_3 = pseudo_color_sine(image, T=255, delta=255 / 3)
    sine_result_4 = pseudo_color_sine(image, T=255 / 2, delta=255 / 2 / 3)
    sine_result_5 = pseudo_color_sine(image, T=255 / 4, delta=255 / 4 / 3)
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(sine_result_3, cv2.COLOR_BGR2RGB))
    plt.title("Pseudo color transform: sine\nT=255, delta=T/3")
    plt.axis("off")
    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(sine_result_4, cv2.COLOR_BGR2RGB))
    plt.title("Pseudo color transform: sine\nT=255/2, delta=T/3")
    plt.axis("off")
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(sine_result_5, cv2.COLOR_BGR2RGB))
    plt.title("Pseudo color transform: sine\nT=255/4, delta=T/3")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(
        img_path.rsplit(".", 1)[0] + "_sine_2" + "." + img_path.rsplit(".", 1)[1]
    )
    plt.show()


if __name__ == "__main__":
    main("./digital_image_hw9_img21.jpg")
    main("./digital_image_hw9_img22.jpg")

```
