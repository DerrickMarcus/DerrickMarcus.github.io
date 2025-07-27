# Week 8 Assignment

题目要求：

编程（可调用内置函数）对比 Robert 算子，Sobel 算子，Laplace 算子和 Canny 算子在所给图片的边缘检测效果并加以分析。在边缘检测结果的基础上利用 Hough 变换进行直线检测。

---

原图像为 `./digital_image_hw8_img1.png` ，首先运行 `./add_noise.py` 文件，对原图像添加轻度和重度的高斯噪声，分别得到 `./digital_image_hw8_img2.png` 和 `./digital_image_hw8_img3.png` ，三张图像的对比图为：

![digital_image_hw8_img_noise](https://cdn.jsdelivr.net/gh/DerrickMarcus/picgo_image/images/digital_image_hw8_img_noise.png)

进行边缘检测和直线检测的源代码为 `./main.py` ，对原图像运行，可以得到四种边缘检测结果：

![digital_image_hw8_edge1](https://cdn.jsdelivr.net/gh/DerrickMarcus/picgo_image/images/digital_image_hw8_edge1.png)

对 Canny 检测结果进行 Hough 变换，得到直线检测结果：

![digital_image_hw8_line1](https://cdn.jsdelivr.net/gh/DerrickMarcus/picgo_image/images/digital_image_hw8_line1.png)

可见，4种边缘检测算法均能较好的提取边缘，其中 Roberts 和 Sobel 提取的结果接近，但是 Sobel 提取的边缘似乎亮度更大一点。Laplace 提取的边缘宽度在视觉上更大，可能是因为作为二级梯度算子，在边缘处垂直明暗变化时经过两次求导，会出现两个尖峰。Canny 算法包含高斯平滑滤波、非极大值抑制、双阈值检测等优化方式，因此提取的边缘亮度均匀，且更加符合实际边缘。

在进行直线检测时，由于图像边缘错综复杂，因此检测结果受 Hough 变换的参数影响较大，但是对于 Canny 算子提取的边缘仍能够提取出原图像中较为明显的直线，比如图像上方的较长的恒直线、右下角正方形的四条边。

对添加轻度噪声的图像做同样操作，得到结果：

![digital_image_hw8_edge2](https://cdn.jsdelivr.net/gh/DerrickMarcus/picgo_image/images/digital_image_hw8_edge2.png)

![digital_image_hw8_line2](https://cdn.jsdelivr.net/gh/DerrickMarcus/picgo_image/images/digital_image_hw8_line2.png)

在添加轻度噪声时，对4种边缘检测均产生了干扰，Roberts 算子和 Sobel 算子的检测结果中，背景亮度稍有提高，整体但边缘结果仍较为明显，受影响程度较低。而 Laplace 算子受影响最大，因为它是二阶梯度算子，对噪声十分敏感，其检测结果已经几乎被噪声填满，难以看出边缘。Canny 算子检测图像中边缘原有边缘处伸展出一些噪点，一些背景处出现了一些孤立的噪声点群，可能会对直线检测造成干扰，背景依然保持黑色，与边缘对比度明显。

对添加重度噪声的图像做同样操作，得到结果：

![digital_image_hw8_edge3](https://cdn.jsdelivr.net/gh/DerrickMarcus/picgo_image/images/digital_image_hw8_edge3.png)

![digital_image_hw8_line3](https://cdn.jsdelivr.net/gh/DerrickMarcus/picgo_image/images/digital_image_hw8_line3.png)

此时仍是 Roberts 算子和 Sobel 算子检测效果较好，且 Sobel 算子最好，在较大噪声情况下，原图像主要边缘（上方针脚、左下方几个圆边矩形、右下角芯片针脚）仍然较为清晰，可能原因是 Sobel 核相比 Roberts 更大、更平滑，抗噪能力更强。而 Laplace 算子和 Canny 算子已经完全被噪声淹没，无法识别出边缘信息。

对于直线检测结果（不同图像都使用相同的 Hough 直线检测参数），在添加轻度噪声时，相比于原图像，在右下方正方形芯片处增加了几条直线，主要原因是该处黑色针脚密集，遍布在正方形框的四周，增加噪声之后多检测出了几条竖向直线和斜向直线。对于重度噪声的图像，由于边缘检测结果已经“面目全非”，其直线检测结果几乎全部是两种斜向 $45^\circ$ 的直线，掩盖了原有边缘直线，失去了参考意义。

如果此时修改 Hough 直线检测的参数，将 `HoughLinesP` 的参数 `maxLineGap` 减小为5，如下图，此时斜向错误直线的识别减少，但是原图像的一些边缘没有识别出来，尤其是图像上方一条最长的横向直线在修改参数之后变得很短，原因是噪声使原本的直线出现中断，而参数 `maxLineGap` 的意义是允许将两条直线是为同一条直线时它们的最小间隔，因此受噪声影响原本的长直线被分割为若干段直线，大部分因为长度没有超过阈值被忽略，而最后只剩下中间的一条中等长度的直线。

![digital_image_hw8_line3](https://cdn.jsdelivr.net/gh/DerrickMarcus/picgo_image/images/digital_image_hw8_line31.png)

源代码文件 `main.py` ：

```py
import cv2
import matplotlib.pyplot as plt
import numpy as np


def roberts_edge(image):
    kernel_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
    kernel_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)

    grad_x = cv2.filter2D(image, cv2.CV_64F, kernel_x)
    grad_y = cv2.filter2D(image, cv2.CV_64F, kernel_y)

    output = cv2.magnitude(grad_x, grad_y)
    output = cv2.normalize(output, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    return output


def sobel_edge(image):
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    output = cv2.magnitude(grad_x, grad_y)
    output = cv2.normalize(output, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    return output


def laplace_edge(image):
    output = cv2.Laplacian(image, cv2.CV_64F, ksize=3)
    output = cv2.convertScaleAbs(output)
    return output


def hough_transform(image, edge):
    lines = cv2.HoughLinesP(
        edge,
        rho=1,  # 距离精度
        theta=np.pi / 180,  # 角度精度
        threshold=80,  # 阈值
        minLineLength=60,  # 最小可接受线段长度
        maxLineGap=5,  # 最大线段间隔
    )

    image_lines = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # 画红色线，线条宽度为1像素
            cv2.line(image_lines, (x1, y1), (x2, y2), (0, 0, 255), 1)
    return image_lines


def main():
    image_path = "./digital_image_hw8_img3.png"
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    result_roberts = roberts_edge(image)
    result_sobel = sobel_edge(image)
    result_laplace = laplace_edge(image)
    result_canny = cv2.Canny(image, 100, 200)

    plt.figure()

    plt.subplot(2, 3, 1)
    plt.imshow(image, cmap="gray")
    plt.title("Original image")
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.imshow(result_roberts, cmap="gray")
    plt.title("Roberts operator")
    plt.axis("off")

    plt.subplot(2, 3, 3)
    plt.imshow(result_sobel, cmap="gray")
    plt.title("Sobel operator")
    plt.axis("off")

    plt.subplot(2, 3, 4)
    plt.imshow(result_laplace, cmap="gray")
    plt.title("Laplace operator")
    plt.axis("off")

    plt.subplot(2, 3, 5)
    plt.imshow(result_canny, cmap="gray")
    plt.title("Canny operator")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig("./digital_image_hw8_edge3.png")
    plt.show()

    # 对 Canny 算子边缘检测结果进行 Hough 变换
    image_lines = hough_transform(image, result_canny)

    plt.figure()

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(image_lines, cv2.COLOR_BGR2RGB))
    plt.title("Hough line detection (Canny edges)")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig("./digital_image_hw8_line31.png")
    plt.show()


if __name__ == "__main__":
    main()
```

添加高斯噪声 `add_noise.py` ：

```py
import cv2
import matplotlib.pyplot as plt
import numpy as np


def add_gaussian_noise(image, mean=0, sigma=10):
    noise = np.random.normal(mean, sigma, image.shape).astype(np.float32)
    noisy_image = image.astype(np.float32) + noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image


def main():
    img = cv2.imread("./digital_image_hw8_img1.png", cv2.IMREAD_GRAYSCALE)

    # 添加轻度高斯噪声
    light_noise_img = add_gaussian_noise(img, mean=0, sigma=10)

    # 添加重度高斯噪声
    heavy_noise_img = add_gaussian_noise(img, mean=0, sigma=20)

    cv2.imwrite("digital_image_hw8_img2.png", light_noise_img)
    cv2.imwrite("digital_image_hw8_img3.png", heavy_noise_img)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.title("Original image")
    plt.imshow(img, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Image with light noise")
    plt.imshow(light_noise_img, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Image with heavy noise")
    plt.imshow(heavy_noise_img, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig("digital_image_hw8_img_noise.png")
    plt.show()


if __name__ == "__main__":
    main()
```
