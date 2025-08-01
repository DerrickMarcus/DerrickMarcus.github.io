# OJ 2军训排队

## Description

军训时需要将所有同学(包括教官)排成一队，一开始只有教官(编号为1)排在第一个，之后教官会发出如下三种命令：

`1 x y` ：编号为 y 的同学入队，排在编号为 x 的同学的后面，若要排在队首，则用0表示 x。

`2 x` ：编号为 x 的同学报告其后面的同学编号，若要报告队首同学的编号，则用0表示 x，若 x 在队尾，则报告0。

`3 x` ：编号为 x 的同学出队。

要求对于每次命令2，都输出报告同学后面同学的编号，并最后从头到尾输出队列中同学的编号。

![oj2_description](https://cdn.jsdelivr.net/gh/DerrickMarcus/picgo_image/images/oj2_description.png)

## Input

第一行为一个正整数 $N$ ，表示总共有几条命令。

接下来 $N$ 行中，每行表示一个命令；每行命令中，数字之间用空格隔开。

## Output

对于每次命令2，都输出一个编号，用换行隔开。

最后从头到尾输出队列，用换行隔开。

## Example

```text
input:
7
1 1 2
1 1 3
2 3
1 2 4
3 1
1 0 5
2 4

output:
2
0
5
3
2
4
```

## Hint

1. 同学编号为小于100000的正整数。
2. 命令的条数 $N$ 为小于200000的正整数。
3. 队列中没有人时不会出现命令2和命令3。

## Solution

使用前向后向的双向数组， `queue_next[i]` 和 `queue_prev[i]` ，分别记录同学 `i` 的下一个人和前一个人，同时根据题目条件恰好可以定义队列的第一个人为0，最后一个人为0，完美适配数组操作。当 `i` 出队时，将 `queue_next[i]` 和 `queue_prev[i]` 都标记为-1，表示已经不在队列中。

由此，对于队列的查找、插入和删除在逻辑上和双向链表类似，但是又简单很多，对于某个人 `i` ，可以直接通过数组下标定位到他的位置，以及他前后相邻的人，且操作都是常数 $O(1)$ 的，最终的时间复杂度为 $O(n)$ ，空间复杂度为 $O(n)$ .

## Code

```c
#include <stdio.h>

int main(int argc, const char* argv[])
{
    int N;
    int cmd1, cmd2, cmd3;
    scanf("%d", &N);

    int queue_next[100001] = {1, 0};
    int queue_prev[100001] = {1, 0};

    for (int i = 2; i < 100001; i++)
    {
        queue_next[i] = queue_prev[i] = -1;
    }

    for (int i = 0; i < N; i++)
    {
        scanf("%d", &cmd1);
        switch (cmd1)
        {
            case 1:
                scanf("%d%d", &cmd2, &cmd3);
                queue_next[cmd3] = queue_next[cmd2];
                queue_prev[queue_next[cmd2]] = cmd3;
                queue_next[cmd2] = cmd3;
                queue_prev[cmd3] = cmd2;
                break;

            case 2:
                scanf("%d", &cmd2);
                printf("%d\n", queue_next[cmd2]);
                break;

            case 3:
                scanf("%d", &cmd2);
                queue_prev[queue_next[cmd2]] = queue_prev[cmd2];
                queue_next[queue_prev[cmd2]] = queue_next[cmd2];
                queue_next[cmd2] = queue_prev[cmd2] = -1;
                break;
        }
    }

    int i = 0;
    while (queue_next[i])
    {
        printf("%d\n", queue_next[i]);
        i = queue_next[i];
    }

    return 0;
}
```
