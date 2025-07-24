# Geometric Transformation 几何变换

!!! abstract "本章概要"
    主要讨论 3D 坐标变换中的旋转矩阵和平移向量，以 激光雷达 LiDAR 到相机 Camera 的姿态变换 为例。

MATLAB 代码实现 激光雷达和相机 坐标变换可视化：

```matlab
% LiDAR --> Camera
T_cam_lidar = [0.0, -1.0, 0.0, 0.0;
               0.6,  0.0, -0.8, -0.93094;
               0.8,  0.0,  0.6, -0.26592;
               0.0,  0.0,  0.0, 1.0];

% Camera --> LiDAR
T_lidar_cam = inv(T_cam_lidar);

% Rotation matrix and translation vector
R = T_lidar_cam(1:3, 1:3);
t = T_lidar_cam(1:3, 4);

% Unit vector of camera frame
camera_axes = transpose([1 0 0; 0 1 0; 0 0 1]);
rotated_camera_axes = R * camera_axes;
camera_origin = t;

figure;
hold on; grid on; axis equal;
xlabel('X'); ylabel('Y'); zlabel('Z');
title('Camera pose w.r.t. LiDAR');

% plot LiDAR frame
quiver3(0, 0, 0, 1, 0, 0, 'r', 'LineWidth', 2);
quiver3(0, 0, 0, 0, 1, 0, 'g', 'LineWidth', 2);
quiver3(0, 0, 0, 0, 0, 1, 'b', 'LineWidth', 2);
text(1, 0, 0, 'LiDAR X', 'Color', 'r');
text(0, 1, 0, 'LiDAR Y', 'Color', 'g');
text(0, 0, 1, 'LiDAR Z', 'Color', 'b');

% plot camera frame
quiver3(camera_origin(1), camera_origin(2), camera_origin(3), ...
        rotated_camera_axes(1,1), rotated_camera_axes(2,1), rotated_camera_axes(3,1), ...
        'r', 'LineWidth', 2);
quiver3(camera_origin(1), camera_origin(2), camera_origin(3), ...
        rotated_camera_axes(1,2), rotated_camera_axes(2,2), rotated_camera_axes(3,2), ...
        'g', 'LineWidth', 2);
quiver3(camera_origin(1), camera_origin(2), camera_origin(3), ...
        rotated_camera_axes(1,3), rotated_camera_axes(2,3), rotated_camera_axes(3,3), ...
        'b', 'LineWidth', 2);

text(camera_origin(1) + rotated_camera_axes(1,1), ...
     camera_origin(2) + rotated_camera_axes(2,1), ...
     camera_origin(3) + rotated_camera_axes(3,1), ...
     'Camera X', 'Color', 'r');
text(camera_origin(1) + rotated_camera_axes(1,2), ...
     camera_origin(2) + rotated_camera_axes(2,2), ...
     camera_origin(3) + rotated_camera_axes(3,2), ...
     'Camera Y', 'Color', 'g');
text(camera_origin(1) + rotated_camera_axes(1,3), ...
     camera_origin(2) + rotated_camera_axes(2,3), ...
     camera_origin(3) + rotated_camera_axes(3,3), ...
     'Camera Z', 'Color', 'b');

legend('LiDAR X','LiDAR Y','LiDAR Z','Camera X','Camera Y','Camera Z');
view(3);
```
