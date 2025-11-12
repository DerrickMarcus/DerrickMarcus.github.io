# Error-State Kalman Filter

> The following content is referenced from:
>
> [进一步掌握误差状态卡尔曼滤波（ESKF）-腾讯云开发者社区-腾讯云](https://cloud.tencent.com.cn/developer/article/2581698)

误差状态卡尔曼滤波（Error-State Kalman Filter, ESKF）的核心思想和传统卡尔曼滤波有所不同，它将状态分为两个部分：

1. 名义状态
2. 误差状态

这样的建模方式有几个优点：

1. 线性化效果更好。
2. 误差动态更简单。
3. 与 IMU 的工作机制完美契合。
