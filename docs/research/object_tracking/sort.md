# SORT

首先是 Tracking-by-Detection 思路的几个代表算法：SORT、DeepSORT、ByteTrack、BoT-SORT 等。

> [ICIP 2016] Simple Online and Realtime Tracking
>
> arXiv: <https://arxiv.org/abs/1602.00763>
>
> IEEE: <https://ieeexplore.ieee.org/document/7533003>
>
> Code: <https://github.com/abewley/sort>

SORT 是多目标跟踪中最经典、最简洁的算法之一，常用在目标检测器（如 YOLO）之后的跟踪层（tracking layer），在检测器输入的的基础上，实时地为每个目标分配持续的身份 ID，实现帧间关联。

SORT 算法的整体流程是“卡尔曼滤波预测 + 匈牙利算法匹配 + ID 管理”：

1. 用卡尔曼滤波器预测上一帧目标的新位置。
2. 以 IoU 为代价运用匈牙利算法，将新检测框与预测框匹配。
3. 匹配成功的检测框更新状态，匹配失败的重新分配 ID 或删除 ID。

## Estimantion Model

卡尔曼滤波中，设定的状态变量为 $\mathbf{x}=[u,v,s,r,\dot{u},\dot{v},\dot{s}]^T$ ，其中：

- $u,v$ 为目标边界框的横纵坐标。
- $s$ 为边界框的尺寸/面积。
- $r$ 为边界框的宽高比 width/height，保持不变。

一般在短时间内，目标匀速运动，那么使用匀速运动模型：

$$
\mathbf{x}_{k|k-1} = \mathbf{F} \mathbf{x}_{k-1|k-1},\quad \mathbf{F} = \begin{pmatrix}
1 & 0 & 0 & 0 & 1 & 0 & 0 \\
0 & 1 & 0 & 0 & 0 & 1 & 0 \\
0 & 0 & 1 & 0 & 0 & 0 & 1 \\
0 & 0 & 0 & 1 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 1 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 1 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 1 \\
\end{pmatrix}
$$

## Data Association

卡尔曼滤波的预测阶段结束后，需要对预测框和检测框进行匹配。计算每一对预测框和检测框的交并比 IoU（intersection-over-union）：

$$
\mathrm{IoU}(A,B)=\frac{\mathrm{area}(A\cap B)}{\mathrm{area}(A\cup B)}
$$

IoU 越大，两个框越接近重合，越有可能对应同一个目标。我们希望匹配的每一对的 IoU 尽可能大，即最大化总 IoU，而匈牙利算法求解的是最小代价，因此可以设置代价为 $c_{ij}=1-\mathrm{IoU}(A_i,B_j)$ .

同时，为避免匹配中某一对 IoU 过小，还需要进行下界阈值 $\mathrm{IoU}_{\min}$ 进行过滤，宁愿不分配，也要避免分配错误的一对边界框。

轨迹生命周期管理（ID 管理）中，每个目标有 3 种可能状态：

- Active 跟踪中：匹配成功，正常更新。
- Lost 暂时丢失：本帧未匹配到检测框，但仍进行预测。
- Deleted 已删除：连续多帧未匹配到，认为目标丢失。

可以设置参数：

- `max_age` ：允许连续多少帧匹配失败就删除，也就是原文的 $T_{lost}$ 参数。
- `min_hits` ：最少连续命中多少次认为是真实目标，避免误检测。

SORT 的局限性在于：存在遮挡时会导致跟踪中断；IoU 对尺度变化较为敏感。
