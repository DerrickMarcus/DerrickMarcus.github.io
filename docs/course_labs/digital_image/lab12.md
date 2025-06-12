# 数字图像处理 第12周作业

题目要求：

1. 使用平滑函数 $h(x)=\exp(\sqrt{x^2+y^2}/120)$ (尺寸7×7)与 DIP 图卷积产生模糊，然后用逆滤波实现对有模糊图像的恢复。
2. 在模糊图像上叠加高斯噪声(均值0，方差8、16、24)，对比逆滤波与维纳滤波对图像的恢复效果。
3. 在原始图像上再叠加其他退化因素（如运动模糊、湍流或其他噪声等）后再次进行实验，给出退化图像及恢复后的图像。
4. 使用其他方法对图像进行恢复，对比效果。
5. 最后显示结果注意归一化。

## Prepare modules

文件 `noise.py` ，包含生成指数模糊卷积核、运动模糊卷积核、模糊图像、为图像添加高斯噪声等。

```py
import cv2
import numpy as np


def exp_kernel(size=7, D0=120):
    assert size % 2 == 1, "Kernel size must be odd."
    center = size // 2
    kernel = np.zeros((size, size), dtype=np.float32)

    for i in range(size):
        for j in range(size):
            x = i - center
            y = j - center
            kernel[i, j] = np.exp(np.sqrt(x**2 + y**2) / D0)

    kernel /= np.sum(kernel)
    return kernel


def motion_kernel(size=7, angle=0):
    assert size % 2 == 1, "Kernel size must be odd."
    psf = np.zeros((size, size), dtype=np.float32)
    psf[(size - 1) // 2, :] = 1.0
    M = cv2.getRotationMatrix2D((size / 2 - 0.5, size / 2 - 0.5), angle, 1)
    psf = cv2.warpAffine(psf, M, (size, size))
    psf /= psf.sum()
    return psf


def blur_image(image, kernel):
    output = cv2.filter2D(image, ddepth=-1, kernel=kernel)
    return output


def add_gaussian_noise(image, mean=0, var=25):
    sigma = np.sqrt(var)
    noise = np.random.normal(mean, sigma, image.shape).astype(np.float32)

    output = image.astype(np.float32) + noise
    output = np.clip(output, 0, 255).astype(np.uint8)
    return output
```

文件 `restore.py` ，包含逆滤波、维纳滤波、最小平方法等图像复原的函数。

```py
import numpy as np


def inverse_filter(image, H_degrade, k=0.5, d=0.1, eps=1e-3):
    G = np.fft.fft2(image.astype(np.float32))
    M = np.where(np.abs(H_degrade) <= d, k, 1.0 / (H_degrade + eps))

    F_hat = G * M
    f_hat = np.fft.ifft2(F_hat)
    f_hat = np.real(f_hat).clip(0, 255).astype(np.uint8)
    return f_hat


def wiener_filter(image, H_degrade, K=0.01):
    G = np.fft.fft2(image.astype(np.float32))
    M = np.conj(H_degrade) / (np.abs(H_degrade) ** 2 + K)

    F_hat = G * M
    f_hat = np.fft.ifft2(F_hat)
    f_hat = np.real(f_hat).clip(0, 255).astype(np.uint8)
    return f_hat


def cls_filter(image, H_degrade, s=0.01):
    G = np.fft.fft2(image.astype(np.float32))

    # fmt: off
    lap = np.array([
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0]
    ], dtype=np.float32)
    # fmt: on
    P = np.fft.fft2(np.fft.ifftshift(lap), s=image.shape)

    M = np.conj(H_degrade) / (np.abs(H_degrade) ** 2 + s * np.abs(P) ** 2)

    F_hat = G * M
    f_hat = np.fft.ifft2(F_hat)
    f_hat = np.real(f_hat).clip(0, 255).astype(np.uint8)
    return f_hat
```

## Task 1

先使用指数平滑函数 $h(x,y)=\exp(\sqrt{x^2+y^2}/120)$ 卷积产生模糊，然后使用逆滤波进行图像恢复，逆滤波函数采用：

$$
M(u,v)=\begin{cases}\
k, & H(u,v)<d\\
\dfrac{1}{H(u,v)}, & \text{else}
\end{cases}
$$

运行文件 `task1.py` ，得到结果：

![202505111526624](https://cdn.jsdelivr.net/gh/DerrickMarcus/picgo-image/images/202505111526624.png)

使用逆滤波之后，确实在一定程度上消除模糊、图像变得更加清晰，但是可以发现图像中出现很多横向和纵向的白色条纹，主要是因为退化函数 $H(u,v)$ 在某些点频率值很低，几乎为零，因此频域做除法时，这些频率点会被过度放大，在时域呈现出周期性的、密密麻麻的细条纹和网格。这是已经对逆滤波加了限制之后的结果，如果不加限制，直接进行 $\hat F(u,v)=\dfrac{G(u,v)}{H(u,v)}$ 处理，恢复后的图像将根本无法辨认。经过改进后的逆滤波需要选择一个较为平衡的 $d$ 值，因为如果 $d$ 值较小，高频分量越强，图像失真严重；如果 $d$ 值较大，处理后图像仍较为模糊，复原效果不佳。

源代码 `task1.py` （绘图部分省略）：

```py
import cv2
import matplotlib.pyplot as plt
import numpy as np

from noise import blur_image, exp_kernel
from restore import inverse_filter

def main():
    image = cv2.imread("data/DIP.bmp", cv2.IMREAD_GRAYSCALE)

    kernel = exp_kernel(size=7, D0=120)
    blurred_image = blur_image(image, kernel)

    H = np.fft.fft2(np.fft.ifftshift(kernel), s=image.shape)
    restored_image = inverse_filter(blurred_image, H, k=0.8, d=0.1, eps=0)

if __name__ == "__main__":
    main()
```

## Task 2

在 Task 1 中的得到的模糊图像中添加3种高斯噪声，均值为0，方差分别为 8, 16, 24 。3个图像分别进行逆滤波和维纳滤波。运行文件 `task2.py` ，效果对比如下：

![202505111526566](https://cdn.jsdelivr.net/gh/DerrickMarcus/picgo-image/images/202505111526566.png)

维纳滤波同样出现了像逆滤波那样的白色细长条纹，但是效果稍好于逆滤波，主要也是因为维纳滤波中 $K$ 值得选取，如果 $K$ 值较小，则对高频率分量抑制效果不佳，条纹和网格越明显；如果 $K$ 值较大，图像仍然模糊，恢复效果不佳。

源代码 `task2.py` （绘图部分省略）：

```py
import cv2
import matplotlib.pyplot as plt
import numpy as np

from noise import add_gaussian_noise, blur_image, exp_kernel
from restore import inverse_filter, wiener_filter

def main():
    image = cv2.imread("data/DIP.bmp", cv2.IMREAD_GRAYSCALE)
    kernel = exp_kernel(size=7, D0=120)
    H = np.fft.fft2(np.fft.ifftshift(kernel), s=image.shape)
    blurred = blur_image(image, kernel)

    noisy_1 = add_gaussian_noise(blurred, mean=0, var=8)
    noisy_2 = add_gaussian_noise(blurred, mean=0, var=16)
    noisy_3 = add_gaussian_noise(blurred, mean=0, var=24)

    inv_1 = inverse_filter(noisy_1, H, k=0.8, d=0.1, eps=0)
    inv_2 = inverse_filter(noisy_2, H, k=0.8, d=0.1, eps=0)
    inv_3 = inverse_filter(noisy_3, H, k=0.8, d=0.1, eps=0)

    wiener_1 = wiener_filter(noisy_1, H, K=0.02)
    wiener_2 = wiener_filter(noisy_2, H, K=0.02)
    wiener_3 = wiener_filter(noisy_3, H, K=0.02)

if __name__ == "__main__":
    main()
```

## Task 3

在原始图像上加入运动模糊，

运行文件 `task3.py` ，效果如下：

![202505111526790](https://cdn.jsdelivr.net/gh/DerrickMarcus/picgo-image/images/202505111526790.png)

可见，逆滤波和维纳滤波都较好地恢复了运动模糊，但是可以发现图像中竖直方向（也就是之前运动模糊的方向）似乎存在白色的、一些部位运动到别处的残影。原因可能是：在频域进行除法的时候，逆滤波和维纳滤波都会在一定程度上会把一些高频噪声放大（难以找到最合适的参数进行抑制），频域的乘除法对应时域的循环卷积，这些模糊部分会被周期性的搬移到这个方向上，产生残影。

源代码 `task3.py` （绘图部分省略）：

```py
import cv2
import matplotlib.pyplot as plt
import numpy as np

from noise import add_gaussian_noise, blur_image, motion_kernel
from restore import inverse_filter, wiener_filter

def main():
    image = cv2.imread("data/DIP.bmp", cv2.IMREAD_GRAYSCALE)

    size = 21
    angle = 90
    psf = motion_kernel(size, angle)

    blurred = blur_image(image, psf)

    blurred = add_gaussian_noise(blurred, mean=0, var=8)

    H = np.fft.fft2(np.fft.ifftshift(psf), s=image.shape)

    inv_restored = inverse_filter(blurred, H, k=0.8, d=0.1, eps=1e-3)

    wiener_restored = wiener_filter(blurred, H, K=0.01)

if __name__ == "__main__":
    main()
```

## Task 4

使用最小平方法恢复图像，$\hat F(u,v)=\dfrac{H^*(u，v)}{|H(u,v)|^2+s|P(u,v)|^2}G(u,v)$ 。

运行文件 `task4.py` ，效果如下：

![202505111527874](https://cdn.jsdelivr.net/gh/DerrickMarcus/picgo-image/images/202505111527874.png)

最小平方法与逆滤波和维纳滤波相比，没有出现在竖直方向上的残影，但是出现了类似之前处理卷积核模糊得到的结果的细小线条和网格。原因可能是拉普拉斯算子频域某些地方值也为0，此时 $s|P(u,v)|^2$ 在分母上相当于不起作用，本应被抑制的高频成分局部放大，加上时域的循环卷积特性，导致出现细小的网格和条纹。

源代码 `task4.py` （绘图部分省略）：

```py
import cv2
import matplotlib.pyplot as plt
import numpy as np

from noise import add_gaussian_noise, blur_image, exp_kernel
from restore import cls_filter

def main():
    image = cv2.imread("data/DIP.bmp", cv2.IMREAD_GRAYSCALE)
    kernel = exp_kernel(size=7, D0=120)
    H = np.fft.fft2(np.fft.ifftshift(kernel), s=image.shape)
    blurred = blur_image(image, kernel)
    blurred = add_gaussian_noise(blurred, mean=0, var=8)

    cls_restored = cls_filter(blurred, H, s=0.001)

if __name__ == "__main__":
    main()

```
