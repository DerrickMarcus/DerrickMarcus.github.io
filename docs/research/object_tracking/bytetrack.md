# ByteTrack

> [ECCV 2022] ByteTrack: Multi-Object Tracking by Associating Every Detection Box
>
> arXiv: <https://arxiv.org/abs/2110.06864>
>
> Springer: <https://rd.springer.com/chapter/10.1007/978-3-031-20047-2_1>
>
> Code: <https://github.com/FoundationVision/ByteTrack>
>
> Translation: <https://zhuanlan.zhihu.com/p/645645269>

大多数的 MOT 算法，只关心置信度（分数）大于阈值的检测框来获取 ID，置信度较低的目标（例如被遮挡的）被丢弃，可能造成真值目标丢失或轨迹碎片化。本文提出，通过关联几乎每个检测框进行跟踪，对于低置信度目标，利用它和轨迹的相似度来恢复真值目标、过滤对背景的检测。

> 清除所有低置信度检测框的做法正确吗？本文答案是否定的。正如黑格尔所说，“存在即合理”。低置信度检测框有时表明目标是存在的，例如被遮挡的目标。

如果一个目标在跟踪过程中发生短暂遮挡，它的边界框置信度会发生短暂下降。但是通过降低置信度阈值并不能解决这个问题，因为仅仅降低置信度阈值又会导致非目标被错误地“框出来”，也就是 false positives。

## BYTE

> 主流方法都专注于设计更好的数据关联方式。然而本文认为检测框的使用方式决定了数据关联的上限，并且专注于在匹配过程中充分利用从高分到低分的检测框，不放弃低分检测框，可提高召回率。

几乎保留所有检测的边界框，分为高分和低分两类。首先将高分检测框与现有轨迹做关联匹配。发生遮挡、运动模糊或者尺寸改变时，某些轨迹可能找不到对应的匹配。

将低分检测框与剩余未匹配的轨迹做匹配。如果低分检测框中有真实目标，则匹配到对应的轨迹，如果没有真实目标，则因为未匹配到轨迹而丢弃。这样就能恢复低分检测框中的目标，同时过滤掉背景。

算法框架为：

1. 检测框分组：按置信度阈值 $\tau$ 将检测框分为高分组 $\mathcal{D}_{high}$ 和低分组 $\mathcal{D}_{low}$ .
2. 轨迹预测：对每一条轨迹使用卡尔曼滤波预测，得到当前帧的预测框位置。
3. Fisrt Association：对轨迹集合 $\mathcal{T}$ 和高分组检测框 $\mathcal{D}_{high}$ 做匈牙利算法最优匹配，相似度使用 IoU 距离或者 IoU + ReID。未匹配的轨迹记为 $\mathcal{T}_{remain}$ ，未匹配的检测框记为 $\mathcal{D}_{remain}$ .
4. Second Association：对 $\mathcal{T}_{remain}$ 和 $\mathcal{D}_{low}$ 做最优匹配，仅使用 IoU 距离计算代价（因为低置信度框通常外观特征不可靠）。仍未匹配的轨迹记为 $\mathcal{T}_{re-remain}$ .
5. 删除长期未匹配的轨迹 $\mathcal{T}\gets \mathcal{T}\backslash\mathcal{T}_{re-remain}$ ，然后建立新的轨迹 $\mathcal{T}\gets \mathcal{T}\cup \mathcal{D}_{remain}$ .
6. 为了保留可能临时丢失的目标，将 $\mathcal{T}_{re-remain}$ 中的轨迹放入 $\mathcal{T}_{lost}$ ，而 $\mathcal{T}_{lost}$ 中的每个轨迹存留 30 帧以上才删除它。
7. 每一帧的输出是 $\mathcal{T}$ 中轨迹的边界框和 ID。

![20251030215650](https://cdn.jsdelivr.net/gh/DerrickMarcus/picgo-image/images/20251030215650.png)

基于 BYTE 设计出跟踪器：ByteTrack = YOLOX(detector) + BYTE(association)

## Bounding Box Annotations

MOT17 数据集中，每个人的边界框覆盖了整个身体，即使他被遮挡或者有部分到了图像之外（边界框也会延伸到图像之外）。但是 YOLOX 默认会把边界框裁剪到图像之内，为避免检测出错，本文修改了 YOLOX 的数据预处理和标签分配策略：

1. 在预处理和数据增强阶段，不在裁剪超出图像的检测框，只删除完全在图像之外的检测框。
2. MOT17 数据集标注中，人的中心点可能在图像之外。YOLOX 的标签分配策略 SimOTA 要求正样本围绕目标中心分布，因此在分配正样本时可能因为中心点超出图像而失效。这种情况下，只把中心点裁到图像内，不裁剪边界框。

MOT20、HiEve 和 BDD100K 数据集的标注，已经将标注的边界框裁剪到图像内，因此无需改动 YOLOX。

## Tracklet Interpolation

MOT17 数据集中，有一些被完全遮挡的行人，YOLOX 无法输出他们的检测框。如果某个目标在被检测到的两帧之间被遮挡，可以使用线性插值“找回”中间的几帧。

假设某轨迹 $T$ 在第 $t_1$ 帧的检测框为 $B_{t1}$ ，第 $t_2$ 帧的检测框为 $B_{t2}$ ，插入中间帧：

$$
B_t=B_{t1}+(B_{t2}-B_{t1})\frac{t-t_1}{t_2-t_1}, \quad t_1<t<t_2
$$

其中检测框 $B_t=[x_{tl},y_{tl},x_{br},y_{br}]\in\mathbb{R}^4$ 代表左上和右下坐标，同时还应该限制遮挡时间不超过最大间隔才进行插值补帧 $t_2-t_1<\sigma$ .
