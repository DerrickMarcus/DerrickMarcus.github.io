---
comments: true
---

# 数字图像处理 第6周作业

首先对图像添加高斯噪声和椒盐噪声，分别封装为函数 `add_gaussian_noise()` 和 `add_salt_pepper_noise()` 。添加高斯噪声时，原图像加上噪声后的结果应该做 0~255 区间截断处理，防止溢出。添加椒盐噪声时，设定出现胡椒噪声（黑点）和盐噪声（白点）的概率相同，简化分析。

添加噪声后的图像与原图像的对比图为 `img_noise.jpg` 。

![img_noise](https://cdn.jsdelivr.net/gh/DerrickMarcus/picgo_image/images/digital_image_hw6_img_noise.jpg)

添加了高斯噪声的直观感觉是图像变得更加模糊，在一小片区域中黑白变化更明显，因为高斯噪声是在正负值范围内分布，可能增大或减小相邻像素之间灰度值差。添加了椒盐噪声之后，就好像在图片中均匀撒了椒盐，出现明显的小白点和小黑点。

然后使用两种滤波方法：中值滤波和高斯滤波。

对于添加高斯噪声的图像，滤波前后效果对比图为：

![img_gaussian_filtered](https://cdn.jsdelivr.net/gh/DerrickMarcus/picgo_image/images/digital_image_hw6_img_gaussian_filtered.jpg)

中值滤波和高斯滤波的峰值信噪比 PSNR 分别为：

```text
Image with gaussian noise, median filtered, PSNR: 78.60 dB
Image with gaussian noise, gaussian filtered, PSNR: 78.08 dB
```

添加高斯噪声的图像，经过中值滤波和高斯滤波之后，都有一定程度的模糊，小片区域内像素灰度值对比差异减小，左侧黑色圆孔也更加圆润，但差异并不明显，PSNR 也很接近。

对于添加椒盐噪声的图像，滤波前后效果对比图为：

![img_salt_filtered](https://cdn.jsdelivr.net/gh/DerrickMarcus/picgo_image/images/digital_image_hw6_img_salt_filtered.jpg)

中值滤波和高斯滤波的峰值信噪比 PSNR 分别为：

```text
Image with salt&pepper noise, median filtered, PSNR: 87.97 dB
Image with salt&pepper noise, gaussian filtered, PSNR: 80.61 dB
```

添加椒盐噪声的图像，经过中值滤波之后噪声明显减小，效果比高斯滤波更好。因为中值滤波选取的卷积核较小 $3\times3$ ，而椒盐噪声分布较为稀疏（设定的概率值只有0.05），因此在一个卷积核覆盖范围内椒盐噪声数量较少，其极端灰度值0或255不会影响中间值，因此中值滤波对于椒盐噪声非常有效。而经过高斯滤波之后，图像更加模糊，而且椒盐噪声也没有很好的去除。

下面使用了两种高频提升方法：拉普拉斯算子、Sobel 算子、高斯-拉普拉斯算子。处理过后的图像为：

![img_sharpened](https://cdn.jsdelivr.net/gh/DerrickMarcus/picgo_image/images/digital_image_hw6_img_sharpened.jpg)

经过拉普拉斯算子锐化之后，图像中边缘更加明显，尤其是左上角的几个同心圆线条更加黑，与桌子对比更加明显。Sobel 算子和高斯-拉普拉斯算子能够提取出图形中的边缘，但后者亮度更高、边缘更清晰，效果更好。

源代码 `./main.py` 如下：

```py
import cv2
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(2025)


def add_gaussian_noise(image, mu=0, sigma=20):
    image = image.astype(np.float32)
    noise = np.random.normal(mu, sigma, image.shape).astype(np.float32)
    output = image + noise
    output = np.clip(output, 0, 255).astype(np.uint8)
    return output


def add_salt_pepper_noise(image, prob=0.05):
    output = np.copy(image)
    rnd = np.random.rand(image.shape[0], image.shape[1])
    output[rnd < prob / 2] = 0
    output[rnd > 1 - prob / 2] = 255
    output = output.astype(np.uint8)
    return output


def compute_psnr(original, filtered, n=8):
    mse = np.mean((original - filtered) ** 2)
    if mse == 0:
        psnr = float("inf")
    else:
        psnr = 20 * np.log10((2**n - 1) ** 2 / np.sqrt(mse))
    return psnr


def laplacian_sharpen(image):
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32)
    # kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    output = cv2.filter2D(image, -1, kernel)
    return output


def sobel_sharpen(image):
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
    gradient_x = cv2.filter2D(image, -1, kernel_x)
    gradient_y = cv2.filter2D(image, -1, kernel_y)
    output = cv2.addWeighted(gradient_x, 0.5, gradient_y, 0.5, 0)
    return output


def log_sharpen(image):
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    kernel = np.array(
        [
            [0, 0, -1, 0, 0],
            [0, -1, -2, -1, 0],
            [-1, -2, 16, -2, -1],
            [0, -1, -2, -1, 0],
            [0, 0, -1, 0, 0],
        ],
        dtype=np.float32,
    )
    output = cv2.filter2D(blurred, -1, kernel)
    return output


def main():
    image_path = "./img.jpg"
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    image_gaussian = add_gaussian_noise(image, mu=0, sigma=25)
    image_salt = add_salt_pepper_noise(image, prob=0.05)

    plt.figure(figsize=(12, 8))

    plt.subplot(1, 3, 1)
    plt.title("Original image")
    plt.imshow(image, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Image with gaussian noise")
    plt.imshow(image_gaussian, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Image with salt&pepper noise")
    plt.imshow(image_salt, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig("./img_noise.jpg")
    plt.show()

    image_gaussian_median = cv2.medianBlur(image_gaussian, ksize=3)
    image_gaussian_gaussian = cv2.GaussianBlur(image_gaussian, ksize=(3, 3), sigmaX=0)
    psnr_gaussian_median = compute_psnr(image, image_gaussian_median)
    psnr_gaussian_gaussian = compute_psnr(image, image_gaussian_gaussian)
    print(
        f"Image with gaussian noise, median filtered, PSNR: {psnr_gaussian_median:.2f} dB"
    )
    print(
        f"Image with gaussian noise, gaussian filtered, PSNR: {psnr_gaussian_gaussian:.2f} dB"
    )

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.title("Original image")
    plt.imshow(image, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.title("Image with gaussian noise")
    plt.imshow(image_gaussian, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.title("Image with gaussian noise, after median filter")
    plt.imshow(image_gaussian_median, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.title("Image with gaussian noise, after gaussian filter")
    plt.imshow(image_gaussian_gaussian, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig("./img_gaussian_filtered.jpg")
    plt.show()

    image_salt_median = cv2.medianBlur(image_salt, ksize=3)
    image_salt_gaussian = cv2.GaussianBlur(image_salt, ksize=(3, 3), sigmaX=0)
    psnr_salt_median = compute_psnr(image, image_salt_median)
    psnr_salt_gaussian = compute_psnr(image, image_salt_gaussian)
    print(
        f"Image with salt&pepper noise, median filtered, PSNR: {psnr_salt_median:.2f} dB"
    )
    print(
        f"Image with salt&pepper noise, gaussian filtered, PSNR: {psnr_salt_gaussian:.2f} dB"
    )

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.title("Original image")
    plt.imshow(image, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.title("Image with salt&pepper noise")
    plt.imshow(image_salt, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.title("Image with salt&pepper noise, after median filter")
    plt.imshow(image_salt_median, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.title("Image with salt&pepper noise, after gaussian filter")
    plt.imshow(image_salt_gaussian, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig("./img_salt_filtered.jpg")
    plt.show()

    image_laplacian = laplacian_sharpen(image)
    image_sobel = sobel_sharpen(image)
    image_log = log_sharpen(image)

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.title("Original image")
    plt.imshow(image, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.title("Laplacian sharpened image")
    plt.imshow(image_laplacian, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.title("Sobel sharpened image")
    plt.imshow(image_sobel, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.title("LoG sharpened image")
    plt.imshow(image_log, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig("./img_sharpened.jpg")
    plt.show()


if __name__ == "__main__":
    main()

```
