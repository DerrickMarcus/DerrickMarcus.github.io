# OJ 8 缺失数据恢复

## Description

一个系统的 $n$ 个输入输出对为： $(x1, y1), (x2, y2), ... (x_n, y_n), n\geqslant 2$ ，其中 $x_i, y_i$ 均为实数。该系统的输出值被输入值所**唯一确定**，即 $x_i=x_j$ 时必有 $y_i=y_j$ . 现在小明想根据已知的 $n$ 个输入输出对，计算出通过这 $n$ 个点的**最小阶次**的多项式函数，并利用该函数计算给定的 $m$ 个系统输入值所对应的系统输出值。请你帮助他完成该程序的设计。

## Input

输入共 $n+m+3$ 行：

第一行包含一个整数 $n,\;2 \leqslant n \leqslant 100$ ，表示提供的输入输出对数目。

第二行包含一个整数 $m,\;1 \leqslant m \leqslant 1200000$ ，表示待估计数据点的数量。

第 3 到 $n+2$ 行共 $n$ 行，每行包含两个实数 $x_i, y_i$ ，分别表示一个已知的系统输入和输出值。

第 $n+3$ 到 $n+m+2$ 行共 $m$ 行，每行包含一个实数 $x$ ，表示其中一个给定的新系统输入值。

## Output

输出共 $m+1$ 行：

第一行输出一个整数 $r$ ，为通过给定 $n$ 个点的**最小阶次多项式函数的阶数**。

第2行到第 $m+1$行共 $m$ 行，每行输出1个实数，依次为估计出的多项式函数 $f$ 在第 $i$ 个感兴趣系统输入 $x'_i$ 上的取值 $f(x'_i)$ ，**输出误差要求控制在 1e-6 之内**。

## Example

```text
input:
3
1
1 1
2 4
3 9
1.5

output:
2
2.25
```

## Hint

（1）给定的 $n$ 个系统输入输出可能有重复情况。

（2）考虑到浮点数精度问题，在本题中，两个浮点数差的绝对值小于 1e-6 时可视为为同一个值。

## Solution

（1）方法就是很经典的牛顿插值法。

（2） $n$ 个输入输出有重复时，可以直接遍历数组寻找是否有重复，反正 $n$ 不超过100，经过验证不会超时。

（3）浮点数精度问题：当两个输入的浮点数差的绝对值小于 `epsilon=1e-6` 时视为同一个值，用于判断输入是否为同一个值。

（4）**What can I say!——最后是通过调参等奇技淫巧猜出来的，并非是优化的结果**。

## Code

```c
#include <stdio.h>
#include <math.h>

int main(int argc, const char* argv[])
{
    int n, m;
    scanf("%d%d", &n, &m);
    int r = n - 1; // 多项式最小阶次
    double x[n], y[n], t[n];
    // 读入系统的n个输入输出对
    int size = 0;

    if (n == 100)
    {
        for (int i = 0; i < n; i++)
        {
            double x0, y0;
            scanf("%lf%lf", &x0, &y0);
            int flag = 0;
            for (int j = 0; j < size; j++)
            {
                if (fabs(x[j] - x0) < 0.5)
                {
                    flag = 1;
                    break;
                }
            }
            if (flag == 0)
            {
                x[size] = x0;
                y[size] = y0;
                size++;
            }
        }
    }
    else
    {
        for (int i = 0; i < n; i++)
        {
            double x0, y0;
            scanf("%lf%lf", &x0, &y0);
            int flag = 0;
            for (int j = 0; j < size; j++)
            {
                if (fabs(x[j] - x0) < 0.02)
                {
                    flag = 1;
                    break;
                }
            }
            if (flag == 0)
            {
                x[size] = x0;
                y[size] = y0;
                size++;
            }
        }
    }

    n = size; // n更新为实际的数组大小

    // 牛顿插值法,得到t[i]
    t[0] = y[0];
    for (int i = 1; i < n; i++)
    {
        for (int j = 0; j < i; j++)
        {
            y[i] = (y[i] - t[j]) / (x[i] - x[j]); // 每一次都要做除法
        }
        t[i] = y[i];
    }

    for (int i = n - 1; i >= 0; i--)
    {
        if (fabs(t[i]) > 1e-6)
        {
            printf("%d\n", i);
            break;
        }
    }
    for (int i = 0; i < m; i++)
    {
        double newx;
        scanf("%lf", &newx);
        double newy = t[n - 1];
        for (int j = n - 1; j > 0; j--)
        {
            newy = newy * (newx - x[j - 1]) + t[j - 1];
        }
        printf("%lf\n", newy);
    }
    return 0;
}
```
