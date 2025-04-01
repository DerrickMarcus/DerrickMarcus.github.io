# 数字图像处理 第2周作业

修改待处理图片的路径然后运行文件 `task.py` ，分解图像的 R,G,B 分量，分解 C,M,Y 分量，分解 H,S,I 分量。结果分别保存在 `img1_decomposed.jpg` 和 `img2_decomposed.jpg` 文件中。

第一张图：

![img1_decomposed](https://cdn.jsdelivr.net/gh/DerrickMarcus/picgo_image/images/img1_decomposed.jpg)

这张图片的 R,G,B 三个通道的分量强度较为接近，整体上颜色逐渐变深，约有 R>G>B ，三种分量在白人皮肤、衬衫、部分背景处强度较强，在黑人皮肤、衣服等其他地方强度较弱。

又因为 C,M,Y 通道是 R,G,B 的互补，因此约有 C < M < Y ，但是整体强度相近。三种分量均在白人衬衫、面部以及背景处较弱，在其他地方强度较强。

H 分量图中可以看出皮肤处红色分量最强，衣服处蓝色分量最强，背景以黄、绿为主。S 通道分量图片整体较暗，说明整体饱和度较低。I 通道反映图像明暗信息，对比明显，也是在白人皮肤、衬衫、部分背景处较亮，其他地方较暗。

第二张图：

![img2_decomposed](https://cdn.jsdelivr.net/gh/DerrickMarcus/picgo_image/images/img2_decomposed.jpg)

这张图片的左侧区域部分花的红色分量和绿色分量强度较大，明显高于右侧区域，且右侧区域绿色分量最弱，而蓝色分量分布较为均匀。

与 R,G,B 相反，左侧区域部分花的青色分量和品红色分量强度较弱，明显弱于右侧区域，且右侧区域品红分量最强，而黄色分量分布较为均匀。

色调以黄色、绿色、品红色为主。整体饱和度较高，说明花朵颜色较为鲜艳。左侧区域亮度较高，右侧区域亮度较低。

附源代码 `task.py`：

```py
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def decompose_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image_array = np.array(image)

    r = image_array[:, :, 0]
    g = image_array[:, :, 1]
    b = image_array[:, :, 2]

    c = 255 - r
    m = 255 - g
    y = 255 - b

    r_norm = r / 255.0
    g_norm = g / 255.0
    b_norm = b / 255.0

    i = (r_norm + g_norm + b_norm) / 3.0

    min_rgb = np.minimum(np.minimum(r_norm, g_norm), b_norm)
    s = 1 - (3.0 / (r_norm + g_norm + b_norm + 1e-6)) * min_rgb

    numerator = 0.5 * ((r_norm - g_norm) + (r_norm - b_norm))
    denominator = (
        np.sqrt((r_norm - g_norm) ** 2 + (r_norm - b_norm) * (g_norm - b_norm)) + 1e-6
    )
    h = np.arccos(numerator / denominator)
    h[b_norm > g_norm] = 2 * np.pi - h[b_norm > g_norm]
    h = h / (2 * np.pi)

    plt.figure(figsize=(12, 8))

    plt.subplot(3, 3, 1)
    plt.imshow(r, cmap="gray")
    plt.title("Red Channel")
    plt.axis("off")

    plt.subplot(3, 3, 2)
    plt.imshow(g, cmap="gray")
    plt.title("Green Channel")
    plt.axis("off")

    plt.subplot(3, 3, 3)
    plt.imshow(b, cmap="gray")
    plt.title("Blue Channel")
    plt.axis("off")

    plt.subplot(3, 3, 4)
    plt.imshow(c, cmap="gray")
    plt.title("Cyan Channel")
    plt.axis("off")

    plt.subplot(3, 3, 5)
    plt.imshow(m, cmap="gray")
    plt.title("Magenta Channel")
    plt.axis("off")

    plt.subplot(3, 3, 6)
    plt.imshow(y, cmap="gray")
    plt.title("Yellow Channel")
    plt.axis("off")

    plt.subplot(3, 3, 7)
    plt.imshow(h, cmap="hsv")
    plt.title("Hue Channel")
    plt.axis("off")

    plt.subplot(3, 3, 8)
    plt.imshow(s, cmap="gray")
    plt.title("Saturation Channel")
    plt.axis("off")

    plt.subplot(3, 3, 9)
    plt.imshow(i, cmap="gray")
    plt.title("Intensity Channel")
    plt.axis("off")

    plt.suptitle("Image Decomposition")
    plt.tight_layout()

    output_filename = image_path.rsplit(".", 1)[0] + "_decomposed.jpg"
    plt.savefig(output_filename, bbox_inches="tight", dpi=300)
    print(f"Decomposed image saved as: {output_filename}")

    plt.show()


def main():
    image_path = "./img1.jpg"
    decompose_image(image_path)


if __name__ == "__main__":
    main()

```
