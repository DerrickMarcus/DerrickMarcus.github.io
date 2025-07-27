# Week 13 Assignment

题目要求：

对 lena.bmp 图像进行仿射变换，包括平移、放缩和旋转（至少每种变换各进行一次），参数自定，但应能较明显看出变换效果，且需要在文档中进行说明。对于放缩和旋转变换后的图像分别采用最近邻插值和双线性插值，并对这两种插值方法进行对比和说明。

## 1. 平移变换

运行文件 `src/translate.py` ，向右平移50像素，向左平移30像素，将平移之后空出来的像素设为0（黑色），平移变换的结果为：

![202505171122810](https://cdn.jsdelivr.net/gh/DerrickMarcus/picgo-image/images/202505171122810.png)

源代码 `translate.py` ：

```py
import cv2
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 指定中文字体为黑体
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号


def main():
    img = cv2.imread("images/Lena.bmp", cv2.IMREAD_GRAYSCALE)

    h, w = img.shape

    # 向右平移50像素，向下平移30像素
    tx, ty = 50, 30
    M = np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)

    translated = cv2.warpAffine(
        img, M, (w, h), None, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, (0,)
    )

    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap="gray")
    plt.title("原始图像")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(translated, cmap="gray")
    plt.title("平移后的图像")
    plt.axis("off")

    plt.savefig("images/translate.png", dpi=300)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
```

## 2. 放缩变换

运行文件 `src/scale.py` ，宽度放大至原来的 1.4 倍，高度缩小为原来的 0.7 倍，分别进行最近邻插值和双线性插值，放缩变换的结果为：

![202505171122355](https://cdn.jsdelivr.net/gh/DerrickMarcus/picgo-image/images/202505171122355.png)

可见，最近邻插值的图像，边缘处出现阶梯形状的锯齿，毛刺明显，部分区域（如背景处）层次感明显，像素块不连续，跳跃感较强，是由于最近邻插值仅用一个最靠近的原图像素赋值，会丢弃周围像素的信息。而双线性插值边缘更加平滑，无明显的锯齿，如果是放大变换的话，仅有轻微的模糊，但整体视觉上更加符合图片直接放缩变形的结果。双线性插值通过对周围四个像素做加权平均，弱化了像素间的突变，新像素值在空间上过渡更加平滑。

源代码 `scale.py` ：

```py
import cv2
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 黑体
plt.rcParams["axes.unicode_minus"] = False  # 负号正常显示


def main():
    img = cv2.imread("images/Lena.bmp", cv2.IMREAD_GRAYSCALE)
    h, w = img.shape

    # 放缩比例
    sx, sy = 1.4, 0.7
    M = np.array([[sx, 0, 0], [0, sy, 0]], dtype=np.float32)

    new_w, new_h = int(w * sx), int(h * sy)
    # 最近邻插值
    scaled_1 = cv2.warpAffine(
        img,
        M,
        (new_w, new_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0,),
    )
    # 双线性插值
    scaled_2 = cv2.warpAffine(
        img,
        M,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0,),
    )

    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(scaled_1, cmap="gray")
    plt.title(f"放缩 sx:{sx}, sy:{sy} + 最近邻插值")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(scaled_2, cmap="gray")
    plt.title(f"放缩 sx:{sx}, sy:{sy} + 双线性插值")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig("images/scaled.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
```

## 3. 旋转变换

运行文件 `src.rotate.py` ，以中间宽度处、三分之一高度处为中心，将图片旋转逆时针旋转30度，分别进行最近邻插值和双线性插值，旋转变换的结果为：

![202505171123783](https://cdn.jsdelivr.net/gh/DerrickMarcus/picgo-image/images/202505171123783.png)

同样，最近邻插值的图像，边缘处锯齿和毛刺明显，而双线性插值边缘更加平滑。

源代码 `rotate.py` ：

```py
import cv2
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 黑体
plt.rcParams["axes.unicode_minus"] = False  # 负号正常显示


def main():
    img = cv2.imread("images/Lena.bmp", cv2.IMREAD_GRAYSCALE)
    h, w = img.shape

    # 设置旋转参数：角度、中心、比例
    angle = 30
    center = (w / 2, h / 3)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # 最近邻插值
    rotated_1 = cv2.warpAffine(
        img,
        M,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0,),
    )
    # 双线性插值
    rotated_2 = cv2.warpAffine(
        img,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0,),
    )

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(rotated_1, cmap="gray")
    plt.title(f"旋转 {angle} + 最近邻插值")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(rotated_2, cmap="gray")
    plt.title(f"旋转 {angle} + 双线性插值")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig("images/rotated.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
```

## 4. 透视变换

运行文件 `perpective.py` ，原图像中选定4个点，指定变换后4个点的位置，然后计算透视矩阵并应用透视变换，结果为：

![202505171146656](https://cdn.jsdelivr.net/gh/DerrickMarcus/picgo-image/images/202505171146656.png)

与上面的放缩变换与旋转变换类似，最近邻插值的边缘锯齿和不连续的视觉效果更加突出，而双线性插值的图像更加平滑连续。

源代码 `perpective.py` ：

```py
import cv2
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def main():
    img = cv2.imread("images/Lena.bmp", cv2.IMREAD_GRAYSCALE)
    h, w = img.shape

    # 定义透视变换的四对点
    # 原图中的矩形区域四个顶点
    src_pts = np.float32([[50, 50], [w - 50, 50], [w - 50, h - 50], [50, h - 50]])
    # 透视后映射到的四边形顶点
    dst_pts = np.float32([[10, 100], [w - 100, 50], [w - 50, h - 100], [100, h - 50]])

    # 计算透视矩阵
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # 最近邻插值
    warped_1 = cv2.warpPerspective(
        img,
        M,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0,),
    )
    wraped_2 = cv2.warpPerspective(
        img,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0,),
    )

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap="gray")
    plt.scatter(src_pts[:, 0], src_pts[:, 1], c="r")
    for i, pt in enumerate(src_pts):
        plt.text(pt[0] + 5, pt[1] + 5, f"{i + 1}", color="red")
    plt.title("原始图像")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(warped_1, cmap="gray")
    plt.scatter(dst_pts[:, 0], dst_pts[:, 1], c="r")
    for i, pt in enumerate(dst_pts):
        plt.text(pt[0] + 5, pt[1] + 5, f"{i + 1}", color="red")
    plt.title("透视变换图像+最近邻插值")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(wraped_2, cmap="gray")
    plt.scatter(dst_pts[:, 0], dst_pts[:, 1], c="r")
    for i, pt in enumerate(dst_pts):
        plt.text(pt[0] + 5, pt[1] + 5, f"{i + 1}", color="red")
    plt.title("透视变换图像+双线性插值")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig("images/perspective.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
```
