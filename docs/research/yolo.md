# YOLO

Official site: <https://docs.ultralytics.com>

YOLO（You Only Look Once）是目标检测领域最流行的模型系列。
·
YOLO 系列是基于 CNN 的，但它在 CNN 的末端添加了特定的检测头（Detection Head）来输出边界框、类别和置信度，而普通的 CNN 通常只输出分类概率或分割图。不同版本的 YOLO 在网络结构上各有不同，但是都继承了以下几个特点：

1. 单阶段（One-Stage）：定位和分类同步完成。它直接在特征图上回归边界框、目标置信度和类别概率，推理速度极快。
2. 划分网格：将输入图像划分为 $S \times S$ 的网格。如果一个目标的中心落入某个网格，该网格就负责预测该目标。
3. 端到端（End-to-End）。整个网络作为一个单一的实体进行训练，直接优化最终的检测性能。

| 特性     | 普通 CNN                                 | YOLO                                                         |
| -------- | ---------------------------------------- | ------------------------------------------------------------ |
| 目标     | 分类                                     | 目标检测                                                     |
| 输出     | 最终输出是一个固定长度的向量（类别概率） | 最终输出是一个高维张量，包含边界框坐标 $(x, y, w, h)$ 、置信度分数和类别概率。 |
| 空间信息 | 在全连接层中丢失了大部分空间信息         | 必须保留空间信息，利用特征图上的每个像素（或网格）来预测目标。 |
| 损失函数 | 交叉熵损失（Cross-Entropy Loss）         | 复杂的多任务损失（Multi-task Loss）：包含分类损失、定位损失（如 IoU Loss, G-IoU）和置信度损失。 |

简单来说，YOLO 就是在 CNN Backbone 的基础上，通过复杂的 Neck 结构进行多尺度特征融合，并使用独特的 Head 结构和多任务损失函数，将 CNN 从一个分类器转变为一个高效、端到端的目标检测器。

---

YOLO 的整体结构分为三个部分：

1. Backbone 从输入图像中提取关键特征。
2. Neck 融合和聚合来自 Backbone 的不同层级的特征。
3. Head 对融合后的特征进行最终的边界框和类别预测。

YOLOv7

(1)设计多种可训练的无代价提升方法，在不增加推理成本前提下显著提升实时目标检测精度；

(2)针对目标检测方法演进，发现两个新问题——重参数化模块如何替代原始模块，以及动态标签分配策略如何处理不同输出层的分配问题，并提出相应解决方案；

(3)为实时目标检测器提出"扩展"与"复合缩放"方法，能有效利用参数与计算资源；

(4)所提方法可减少顶尖实时目标检测器约40%参数量与50%计算量，同时获得更快的推理速度与更高的检测精度。

## Traing on Custom Dataset

目标：使用官方提供的 yolo12n.pt （在 MS COCO 数据集上训练）预训练权重，在以下几个数据集上做微调，期望能够在暗黑环境中实现更好的检测效果：

1. 夜间行人：<https://www.nightowls-dataset.org/>
2. 欧洲城市行人：<https://eurocity-dataset.tudelft.nl/> ，具有 day/night 标记。
3. UC Berkeley 的 BDD100K：<http://bdd-data.berkeley.edu/download.html> ，可过滤出含 person 图片和夜间图片。

首先，需要下载数据集，并按照 YOLO 格式进行标注。

## YOLO Dataset Format

与 COCO 格式将所有标注信息存储在一个 JSON 文件中不同，YOLO 格式是将每张图片的标注信息存储在一个同名的 `.txt` 文件中，最后通过一个 YAML 文件描述类别、训练集测试集路径等信息。

图片与标注文件一一对应，例如：

1. `images/train/0001.jpg`
2. `labels/train/0001.txt`

标注文件中，每一行代表一个边界框，格式为 `class_id, x_center, y_center, width, height` ，其中坐标均为相对于图片宽高的归一化值，浮点数，取值范围为 `[0, 1]` 。例如：

```text
0 0.512 0.433 0.300 0.400
2 0.215 0.600 0.100 0.150
```

YAML 文件中，定义类别名称和数据集路径，例如：

```yaml
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# COCO8 dataset (first 8 images from COCO train2017) by Ultralytics
# Documentation: https://docs.ultralytics.com/datasets/detect/coco8/
# Example usage: yolo train data=coco8.yaml
# parent
# ├── ultralytics
# └── datasets
#     └── coco8 ← downloads here (1 MB)

# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
path: coco8 # dataset root dir
train: images/train # train images (relative to 'path') 4 images
val: images/val # val images (relative to 'path') 4 images
test: # test images (optional)

# Classes
names:
  0: person
  1: bicycle
  2: car
  ......

# Download script/URL (optional)
download: https://github.com/ultralytics/assets/releases/download/v0.0.0/coco8.zip
```

YOLO 约定，图片存放的目录 `images/` 和标注文件存放的目录 `labels/` 在同一级目录中，因此在 YAML 文件中只需指定图片目录 `images/train` 和 `images/val` 即可，YOLO 会自动寻找对应的 `labels/train` 和 `labels/val` 目录。

![20251117165712](https://cdn.jsdelivr.net/gh/DerrickMarcus/picgo-image/images/20251117165712.png)
