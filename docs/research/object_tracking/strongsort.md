# StrongSORT

> [IEEE TMM 2023] StrongSORT: Make DeepSORT Great Again
>
> arXiv: <https://arxiv.org/abs/2202.13514>
>
> Code: <https://github.com/dyhBUPT/StrongSORT>, <https://github.com/open-mmlab/mmtracking>

Abstract:

Recently, multi-object tracking (MOT) has attracted increasing attention, and accordingly, remarkable progress has been achieved. However, the existing methods tend to use various basic models (e.g., detector and embedding models) and different training or inference tricks. As a result, the construction of a good baseline for a fair comparison is essential. In this paper, a classic tracker, i.e., DeepSORT, is first revisited, and then is significantly improved from multiple perspectives such as object detection, feature embedding, and trajectory association. The proposed tracker, named StrongSORT, contributes a strong and fair baseline to the MOT community. Moreover, two lightweight and plug-and-play algorithms are proposed to address two inherent “missing” problems of MOT: missing association and missing detection. Specifically, unlike most methods, which associate short tracklets into complete trajectories at high computational complexity, we propose an appearance-free link model (AFLink) to perform global association without appearance information, and achieve a good balance between speed and accuracy. Furthermore, we propose Gaussian-smoothed interpolation (GSI) based on Gaussian process regression to relieve missing detection. AFLink and GSI can be easily plugged into various trackers with a negligible extra computational cost (1.7 ms and 7.1 s per image, respectively, on MOT17). Finally, by fusing trongSORT with AFLink and GSI, the final tracker (StrongSORT++) achieves state-of-the-art results on multiple public benchmarks, i.e., MOT17, MOT20, DanceTrack and KITTI. Codes are available at <https://github.com/dyhBUPT/StrongSORT> nd <https://github.com/open-mmlab/mmtracking>.

Keywords:

Multi-Object Tracking, Baseline, AFLink, GSI.
