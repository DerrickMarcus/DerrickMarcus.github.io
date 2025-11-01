# Kd-Tree

kd-tree 是一种将高维空间数据（如3D点云）组织成树结构的算法，用于快速查找最近邻点。

kd-tree 是一种二叉树结构，将空间中数据进行划分，使得每个节点代表一个区域，便于快速排除不可能的区域。

以 3D 点云为例(k=3)：

1. 首先按 `x` 坐标排序，将点集一分为二。
2. 左右子树再按 `y` 坐标划分。
3. 再按 `z` 坐标……
4. 循环切分下去，直到某个最小维度或点数。

这样，对于一个查询点，只需在树上“走一遍”，就可以快速找到附近的候选点。

示例代码：使用 kd-tree 在 3D 点云中进行最近邻搜索。构造一个 3D 点云，然后用 `pcl::KdTreeFLANN` 建立 kd-tree，最后查找某个点在这个点云中的最近邻。

```cpp
#include <iostream>
#include <vector>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>

int main() {
    // 创建一个点云并填充数据
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    // 假设我们添加 100 个随机点
    for (int i = 0; i < 100; ++i) {
        pcl::PointXYZ pt;
        pt.x = 1024 * rand() / (RAND_MAX + 1.0f);
        pt.y = 1024 * rand() / (RAND_MAX + 1.0f);
        pt.z = 1024 * rand() / (RAND_MAX + 1.0f);
        cloud->points.push_back(pt);
    }
    cloud->width = cloud->points.size();
    cloud->height = 1;
    cloud->is_dense = true;

    // 创建 kd-tree 并构建索引
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(cloud);

    // 查询点
    pcl::PointXYZ searchPoint;
    searchPoint.x = 100.0;
    searchPoint.y = 100.0;
    searchPoint.z = 100.0;

    // 设置最近邻个数
    int K = 5;
    std::vector<int> pointIdxNKNSearch(K);
    std::vector<float> pointNKNSquaredDistance(K);

    std::cout << "Searching for the " << K << " nearest neighbors of ("
              << searchPoint.x << ", " << searchPoint.y << ", " << searchPoint.z << ")\n";

    if (kdtree.nearestKSearch(searchPoint, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0) {
        for (size_t i = 0; i < pointIdxNKNSearch.size(); ++i) {
            std::cout << " " << i + 1 << ". ["
                      << cloud->points[pointIdxNKNSearch[i]].x << ", "
                      << cloud->points[pointIdxNKNSearch[i]].y << ", "
                      << cloud->points[pointIdxNKNSearch[i]].z << "] "
                      << "距离平方 = " << pointNKNSquaredDistance[i] << std::endl;
        }
    }

    return 0;
}
```
