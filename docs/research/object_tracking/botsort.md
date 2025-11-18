# BoT-SORT

> BoT-SORT: Robust Associations Multi-Pedestrian Tracking
>
> arXiv: <https://arxiv.org/abs/2206.14651>
>
> <https://github.com/NirAharon/BoT-SORT>

BoT-SORT 把运动信息（Kalman 预测 + IoU）与外观信息（ReID 特征）融合，同时显式做相机运动补偿（CMC），并把 Kalman Filter 的状态向量从传统的“中心+面积+纵横比”改为“中心 + 宽高”，集成到 ByteTrack 中，提出了 BoT-SORT 和 BoT-SORT-ReID 两种跟踪器。两者的主干相同：

1. 基于 ByteTrack 的两阶段匹配（高置信 + 低置信检测）。
2. Kalman Filter 预测更新。
3. 相机运动补偿。
4. IoU 匹配 / 门控 / 匈牙利算法
5. 轨迹生命周期管理

区别是，在匹配时是否引入 ReID 外观特征。

## Kalman Filter

与 SORT、DeepSORT 都不同，BoT-SORT 选定的状态变量为 $\boldsymbol{x}=[x,y,w,h,\dot{x},\dot{y},\dot{w},\dot{h}]^T$ ，即二维坐标、宽度高度及其变化量。观测变量为 $\boldsymbol{z}=[x,y,w,h]^T$ .

状态转移矩阵和观测矩阵分别为：

$$
\boldsymbol{F}=\begin{pmatrix}
1 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 & 0 & 1 & 0 & 0 \\
0 & 0 & 1 & 0 & 0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1 & 0 & 0 & 0 & 1 \\
0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 \\
\end{pmatrix},\quad
\boldsymbol{H}=\begin{pmatrix}
1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
\end{pmatrix}
$$

采用与时间相关的过程噪声和测量噪声：

$$
\begin{align*}
\boldsymbol{Q}_k = \text{diag}(&(\sigma_p \hat{w}_{k-1|k-1})^2, (\sigma_p \hat{h}_{k-1|k-1})^2,\\
&(\sigma_p \hat{w}_{k-1|k-1})^2, (\sigma_p \hat{h}_{k-1|k-1})^2,\\
&(\sigma_v \hat{w}_{k-1|k-1})^2, (\sigma_v \hat{h}_{k-1|k-1})^2,\\
&(\sigma_v \hat{w}_{k-1|k-1})^2, (\sigma_v \hat{h}_{k-1|k-1})^2)
\\
\boldsymbol{R}_k = \text{diag}(&(\sigma_m \hat{w}_{k|k-1})^2, (\sigma_m \hat{h}_{k|k-1})^2,\\
&(\sigma_m \hat{w}_{k|k-1})^2, (\sigma_m \hat{h}_{k|k-1})^2)
\end{align*}
$$

过程噪声 $\boldsymbol{Q}_k$ 和测量噪声 $\boldsymbol{R}_k$ 是随边界框尺寸动态变化的对角阵， $\sigma_p,\sigma_v,\sigma_m$ 是需要设定的参数。

## Camera Motion Compensation

相机本身会运动的场景中，图像中的边界框可能发生显著变化。相机静止时，也可能受到风引起的振动或漂移。在未知相机运动数据（导航、IMU）或者相机内参矩阵时，可以使用相邻帧之间图像配准，近似看作相机运动在图像上的投影。

使用 OpenCV 中的全局运动估计（GMC，global motion compensation），这种稀疏配准技术允许忽略场景中动态物体，从而更好地估计背景的运动：

1. 提取图像关键点（keypoints）。
2. 使用稀疏光流（sparse optical flow）进行特征跟踪，并对平移分量做阈值处理。
3. 使用 RANSAC（Random Sample Consensus 随机抽样一致）估计相邻帧之间的仿射变换 $\boldsymbol{A}_{k-1}^k\in\mathbb{R}^{2\times 3}$ .

$$
\boldsymbol{A}_{k-1}^k=[\boldsymbol{M}_{2\times 2}\mid \boldsymbol{T}_{2\times 1}]=
\begin{pmatrix}
a_{11} & a_{12} & a_{13} \\
a_{21} & a_{22} & a_{23}
\end{pmatrix}
$$

使用该仿射变换可以预测第 $k-1$ 帧的边界框在第 $k$ 帧中的位置。其中平移部分 $\boldsymbol{T}$ 只影响边界框的中心坐标，不影响宽度、高度和其变化率。而旋转部分 $\boldsymbol{M}$ 对状态变量和噪声都有影响，但由于是线性变换，该仿射矩阵也可以同时同等地作用于整个状态变量，因此我们构造矩阵：

$$
\tilde{\boldsymbol{M}}_{k-1}^k=\begin{pmatrix}
\boldsymbol{M} & 0 & 0 & 0 \\
0 & \boldsymbol{M} & 0 & 0 \\
0 & 0 & \boldsymbol{M} & 0 \\
0 & 0 & 0 & \boldsymbol{M} \\
\end{pmatrix}\in\mathbb{R}^{8\times 8},\quad
\tilde{\boldsymbol{T}}_{k-1}^k=\begin{pmatrix}
a_{12} \\ a_{13} \\ 0 \\ \vdots \\ 0
\end{pmatrix}\in\mathbb{R}^8
$$

假设我们有了原始 KF 预测的状态变量和协方差 $\hat{\boldsymbol{x}}_{k|k-1},\;\boldsymbol{P}_{k|k-1}$ ，补偿后的预测应该修正为：

$$
\begin{align*}
\hat{\boldsymbol{x}}'_{k|k-1}&=\tilde{\boldsymbol{M}}_{k-1}^k \hat{\boldsymbol{x}}_{k|k-1}+\tilde{\boldsymbol{T}}_{k-1}^k \\
\boldsymbol{P}'_{k|k-1}&=\tilde{\boldsymbol{M}}_{k-1}^k \boldsymbol{P}_{k|k-1} \left(\tilde{\boldsymbol{M}}_{k-1}^{k}\right)^T
\end{align*}
$$

此结果作为最终的预测值，再与当前检测/观测 $\boldsymbol{z}$ 做更新：

$$
\begin{align*}
\boldsymbol{K}_k &= \boldsymbol{P}'_{k|k-1} \boldsymbol{H}_k^T \left(\boldsymbol{H}_k \boldsymbol{P}'_{k|k-1} \boldsymbol{H}_k^T + \boldsymbol{R}_k\right)^{-1}
\\
\hat{\boldsymbol{x}}_{k|k} &= \hat{\boldsymbol{x}}'_{k|k-1} + \boldsymbol{K}_k \left(\boldsymbol{z}_k - \boldsymbol{H}_k \hat{\boldsymbol{x}}'_{k|k-1}\right)
\\
\boldsymbol{P}_{k|k} &= \left(\boldsymbol{I} - \boldsymbol{K}_k \boldsymbol{H}_k\right) \boldsymbol{P}'_{k|k-1}
\end{align*}
$$

相机高速运动场景下，上述运动补偿至关重要。相机缓慢运动时，可忽略对 $\boldsymbol{P}_{k|k-1}$ 的补偿。

## IoU-ReID Fusion

类似于 DeepSORT，首先采用 FastReID 库的 BoT-SBS 模型（以 ResNeSt-50 主干），提取 128-D 或 256-D 的 embedding 特征嵌入。

每一条轨迹（tracklet）维护一个**外观状态** $e_i^k$ ，表示第 $i$ 条轨迹在第 $k$ 帧的平均外观特征，并使用指数滑动平均（EMA，exponential moving average）进行更新：

$$
e_i^k=\alpha e_i^{k-1} +(1-\alpha)f_i^k
$$

其中 $f_i^k$ 为当前帧检测到的边界框对应的 embedding， $\alpha=0.9$ 为动量，表示新特征只占 10% 的权重。这样可以平滑 ReID 特征。

对每一条轨迹 $i$ 的平均外观特征 $e_i^k$ 和检测 $j$ 的特征 $f_j^k$ 计算余弦相似度 $d_{i,j}^{\cos}=1-e_i^k\cdot f_j^k$ .

> 不同于 DeepSORT 把 IoU 距离和特征的余弦距离的加权平均作为最终代价，BoT-SORT 采用了更保守的方法：取最小值。

首先进行两次阈值处理，得到新的余弦距离：

$$
\hat{d}_{i,j}^{\cos}=\begin{cases}
0.5 \cdot d_{i,j}^{\cos}, & \text{if } (d_{i,j}^{\cos}<\theta_{emb})\wedge (d_{i,j}^{iou}<\theta_{iou})\\
1, & \text{else}
\end{cases}
$$

若 IoU 和余弦距离都在阈值内，说明重合程度较高，赋予一个更小的距离 $0.5 \cdot d_{i,j}^{\cos}$ ，否则直接设为 1，表示不可能匹配。原文中设置参数 $\theta_{emb}=0.25,\;\theta_{iou}=0.5$ .

最终的代价取最小值 $c_{ij}=\min\{d_{i,j}^{iou},\;\hat{d}_{i,j}^{\cos}\}$ ，这样一来：

- 如果外观相似且位置相近，则使用更小的那个距离。
- 如果外观和位置有任何一个不可信，则把外观代价设为 1，相当于仅使用 IoU 位置距离，退化为纯 IoU 匹配。

## Code Reprodution

> The following content is referenced from:
>
> [BoT-SORT实战：手把手教你实现BoT-SORT训练和测试 - 灰信网（软件开发博客聚合）](https://www.freesion.com/article/74862628185/)
>
> [BoT-SORT复现-CSDN博客](https://blog.csdn.net/Abstract_zhw/article/details/129062743)

### Installation

Step 1. Setup the Python environment.

```bash
# venv
cd ~/venv
python3 -m venv botsort_env
source ~/venv/botsort/bin/activate

# conda
conda create -n botsort_env python=3.7
conda activate botsort_env
```

Step 2: Install `torch` and `torchvision` in from the [Pytorch](https://pytorch.org/get-started/locally/). The code was tested using `torch==1.11.0+cu113` and `torchvision==0.12.0+cu113` .

```bash
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
```

Step 3: Install BoT-SORT.

```bash
git clone https://github.com/NirAharon/BoT-SORT.git
cd BoT-SORT
pip3 install -r requirements.txt
python3 setup.py develop
```

Step 4: Install [pycocotools](https://github.com/cocodataset/cocoapi).

```bash
pip3 install cython
pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```

Step 5: Others.

```bash
# Cython-bbox
pip3 install cython_bbox

# faiss cpu / gpu
pip3 install faiss-cpu
pip3 install faiss-gpu
```

You may need to change the version of `protobuf` to avoid version conflict:

### Data Preparation

This step prepares cropped person images (patches) from MOT17/MOT20 datasets, so that the FastReID pipeline can train an appearance (ReID) model for BoT-SORT(-ReID). This step *does not train anything*, it only builds the ReID training dataset.

You can create a new folder as `<datasets_dir>` (like `datasets/` in the root directory). Then go to [MOT17](https://motchallenge.net/data/MOT17/) and [MOT20](https://motchallenge.net/data/MOT20/) datasets, click the "*Get all data*" at the bottom of the page, and put the unzipped files in the following structure:

```text
<datasets_dir>
      │
      ├── MOT17
      │      ├── train
      │      └── test
      │
      └── MOT20
             ├── train
             └── test
```

Then generates detection patches for training ReID:

```bash
cd <BoT-SORT_dir>

# For MOT17
python3 fast_reid/datasets/generate_mot_patches.py --data_path <dataets_dir> --mot 17

# For MOT20
python3 fast_reid/datasets/generate_mot_patches.py --data_path <dataets_dir> --mot 20
```

It will output the results into folder `fast_reid/datasets/` by default. You can also change this path by providing `--save-path` argument.

### Models

Creat a new folder `<BoT-SORT_dir>/pretrained` and place all the models that you download below to folder `<BoT-SORT_dir>/pretrained` .

1. For **detector**, download the pretrained YOLOX weights from [ByteTrack](https://github.com/FoundationVision/ByteTrack) model zoo (MOT17/MOT20/ablation) `.pth.tar` . For multi-class MOT, use [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) or [YOLOv7](https://github.com/WongKinYiu/yolov7) weights trained on COCO (or any custom weights).

???+ success "ByteTrack Model Zoo"
    Ablation model trained on CrowdHuman and MOT17 half train, evaluated on MOT17 half val.

    | Model                                                        | MOTA | IDF1 | IDs  | FPS  |
    | ------------------------------------------------------------ | ---- | ---- | ---- | ---- |
    | ByteTrack_ablation: [google](https://drive.google.com/file/d/1iqhM-6V_r1FpOlOzrdP_Ejshgk0DxOob/view?usp=sharing), [baidu(eeo8)](https://pan.baidu.com/s/1W5eRBnxc4x9V8gm7dgdEYg) | 76.6 | 79.3 | 159  | 29.6 |

    **MOT17** test model, trained on CrowdHuman, MOT17, Cityperson and ETHZ, evaluated on MOT17 train.

    - Standard models

    | Model                                                        | MOTA | IDF1 | IDs  | FPS  |
    | ------------------------------------------------------------ | ---- | ---- | ---- | ---- |
    | bytetrack_x_mot17: [google](https://drive.google.com/file/d/1P4mY0Yyd3PPTybgZkjMYhFri88nTmJX5/view?usp=sharing), [baidu(ic0i)](https://pan.baidu.com/s/1OJKrcQa_JP9zofC6ZtGBpw) | 90.0 | 83.3 | 422  | 29.6 |
    | bytetrack_l_mot17: [google](https://drive.google.com/file/d/1XwfUuCBF4IgWBWK2H7oOhQgEj9Mrb3rz/view?usp=sharing), [baidu(1cml)](https://pan.baidu.com/s/1242adimKM6TYdeLU2qnuRA) | 88.7 | 80.7 | 460  | 43.7 |
    | bytetrack_m_mot17: [google](https://drive.google.com/file/d/11Zb0NN_Uu7JwUd9e6Nk8o2_EUfxWqsun/view?usp=sharing), [baidu(u3m4)](https://pan.baidu.com/s/1fKemO1uZfvNSLzJfURO4TQ) | 87.0 | 80.1 | 477  | 54.1 |
    | bytetrack_s_mot17: [google](https://drive.google.com/file/d/1uSmhXzyV1Zvb4TJJCzpsZOIcw7CCJLxj/view?usp=sharing), [baidu(qflm)](https://pan.baidu.com/s/1PiP1kQfgxAIrnGUbFP6Wfg) | 79.2 | 74.3 | 533  | 64.5 |

    - Light models

    | Model                                                        | MOTA | IDF1 | IDs  | Params(M) | FLOPs(G) |
    | ------------------------------------------------------------ | ---- | ---- | ---- | --------- | -------- |
    | bytetrack_nano_mot17: [google](https://drive.google.com/file/d/1AoN2AxzVwOLM0gJ15bcwqZUpFjlDV1dX/view?usp=sharing), [baidu(1ub8)](https://pan.baidu.com/s/1dMxqBPP7lFNRZ3kFgDmWdw) | 69.0 | 66.3 | 531  | 0.90      | 3.99     |
    | bytetrack_tiny_mot17: [google](https://drive.google.com/file/d/1LFAl14sql2Q5Y9aNFsX_OqsnIzUD_1ju/view?usp=sharing), [baidu(cr8i)](https://pan.baidu.com/s/1jgIqisPSDw98HJh8hqhM5w) | 77.1 | 71.5 | 519  | 5.03      | 24.45    |

    **MOT20** test model, trained on CrowdHuman and MOT20, evaluated on MOT20 train.

    | Model                                                        | MOTA | IDF1 | IDs  | FPS  |
    | ------------------------------------------------------------ | ---- | ---- | ---- | ---- |
    | bytetrack_x_mot20: [google](https://drive.google.com/file/d/1HX2_JpMOjOIj1Z9rJjoet9XNy_cCAs5U/view?usp=sharing), [baidu(3apd)](https://pan.baidu.com/s/1bowJJj0bAnbhEQ3_6_Am0A) | 93.4 | 89.3 | 1057 | 17.5 |

2. For **tracker**, download the pretrained ReID model weights `.pth` files, from [MOT17-SBS-S50](https://drive.google.com/file/d/1QZFWpoa80rqo7O-HXmlss8J8CnS7IUsN/view?usp=sharing), [MOT20-SBS-S50](https://drive.google.com/file/d/1KqPQyj6MFyftliBHEIER7m_OrGpcrJwi/view?usp=sharing).

### Training ReID

You can modify the parameters in `fast_reid/configs/Base-SBS.yml` to adjust the batch size and learing rate, etc.

Run these commands to train the ReID models:

```bash
cd <BoT-SORT_dir>

# For training MOT17
python3 fast_reid/tools/train_net.py --config-file ./fast_reid/configs/MOT17/sbs_S50.yml MODEL.DEVICE "cuda:0"

# For training MOT20
python3 fast_reid/tools/train_net.py --config-file ./fast_reid/configs/MOT20/sbs_S50.yml MODEL.DEVICE "cuda:0"
```

Then you will get the training results in folder `logs/` .

Refer to [FastReID](https://github.com/JDAI-CV/fast-reid) repository for addition explanations and options.

### Tracking

In this part we run the BoT-SORT `tools/track.py` and generate CSV text files, so that you can submit the `.txt` files to [MOTChallange](https://motchallenge.net/) and get the results in the paper. Tracking parameters can also be tuned in `tools/track.py` .

For the file format, one CSV text file contains one object instance per line. Each line contains 10 values `<frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>` . The world coordinates `x, y, z` are ignored for the 2D challenge and can be filled with -1.

(Post-processing, *optional*) After tracking, you can run `tools/interpolation.py` to perform temporal interpolation and smoothing on tracklets to bridge missed detections. This may improve IDF1/MOTA/HOTA a bit.

1. Test on MOT17

    ```bash
    cd <BoT-SORT_dir>
    python3 tools/track.py <dataets_dir/MOT17> --default-parameters --with-reid --benchmark "MOT17" --eval "test" --fp16 --fuse
    python3 tools/interpolation.py --txt_path <path_to_track_result>
    ```

2. Test on MOT20

    ```bash
    cd <BoT-SORT_dir>
    python3 tools/track.py <dataets_dir/MOT20> --default-parameters --with-reid --benchmark "MOT20" --eval "test" --fp16 --fuse
    python3 tools/interpolation.py --txt_path <path_to_track_result>
    ```

3. Evaluation on MOT17 validation set (the second half of the train set). The final score is calculated locally, no need to submit to the MOTChallenge site.

    ```bash
    cd <BoT-SORT_dir>

    # BoT-SORT
    python3 tools/track.py <dataets_dir/MOT17> --default-parameters --benchmark "MOT17" --eval "val" --fp16 --fuse

    # BoT-SORT-ReID
    python3 tools/track.py <dataets_dir/MOT17> --default-parameters --with-reid --benchmark "MOT17" --eval "val" --fp16 --fuse
    ```

4. Other experiments. You can also ignore the `--default-parameters` and set the parameters by yourself. See all the available tracking parameters by running `#!bash python3 tools/track.py -h` .
5. The commands above runs YOLOX as object detector by default. You can also use **YOLOv7** by running `#!bash python3 tools/track_yolov7.py` .

For evaluating the train and validation sets, we recommend using the official MOTChallenge evaluation code from [TrackEval](https://github.com/JonathonLuiten/TrackEval).

### Demo

Demo with BoT-SORT(-ReID) based YOLOX and multi-class:

```bash
cd <BoT-SORT_dir>

# Original example
python3 tools/demo.py video --path <path_to_video> -f yolox/exps/example/mot/yolox_x_mix_det.py -c pretrained/bytetrack_x_mot17.pth.tar --with-reid --fuse-score --fp16 --fuse --save_result

# Multi-class example
python3 tools/mc_demo.py video --path <path_to_video> -f yolox/exps/example/mot/yolox_x_mix_det.py -c pretrained/bytetrack_x_mot17.pth.tar --with-reid --fuse-score --fp16 --fuse --save_result
```

Demo with BoT-SORT(-ReID) based YOLOv7 and multi-class:

```bash
cd <BoT-SORT_dir>
python3 tools/mc_demo_yolov7.py --weights pretrained/yolov7-d6.pt --source <path_to_video/images> --fuse-score --agnostic-nms --with-reid
```

!!! note "Difference between *Tracking* and *Demo*"
    In *Tracking*, `tools/tack.py` is a batch evaluation script on MOT dataset to provide metrics. Its input source is each image in `<datasets_dir/MOT17>` / `<datasets_dir/MOT20>`, and its output is MOTChallenge-format `.txt` files which can be submitted to the official site. This module is used to reproduce the project and validate the improvement.

    In *Demo*, `tools/demo.py` is a simple script for demonstration to visualize the tracking results. Its input source is video or image stream, and its output is a video with boungding box and IDs.

### Changing Detector to Other YOLO
