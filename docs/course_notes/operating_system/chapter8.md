# 8 存储器管理(1)

## 8.1 存储器管理概述

存储器访问的局部性原理：处理器访问存储器时，无论取指令还是取数据，所访问的存储单元都趋向于聚集在一个较小的连续单元区域中。

时间上的局部性——最近的将来要用到的信息很可能就是现在正在使用的信息。主要由循环造成。

空间上的局部性——最近的将来要用到的信息很可能与现在正在使用的信息在空间上是邻近的。主要由顺序执行和数据的聚集存放造成。

例如：

```c
int arr[1024];
int sum = 0;
for(i = 0, i < 1024; i++) {
    sum += arr[i];
}
```

同时体现时间局部性（循环）和空间局部性（数组在内存中连续）。

存储器的层次结构：寄存器——高速缓存Cache——主存储器(内存)——辅助存储器(外存)。

1. 高速缓存：SRAM。用于 Data Cache, Code cache, TLB。
2. 内存：DRAM、SRAM 等。
3. 外存：软盘，硬盘，光盘、磁带等。

目前正在使用的指令或数据或者将要用到指令或数据应该在高速缓存中，过一段时间才会用到的指令或数据应该在主存储器中，暂时用不到的程序或数据应该在磁盘中。就好像拥有了缓存一样的速度、硬盘一样的容量。

!!! note
    操作系统存储管理的功能针对的是内存，因为外存属于设备(存储设备)，而 Cache 对于软件不可见。

内存管理的功能：

1. 内存分配和回收。
2. 地址变换与重定位。
3. 内存保护。
4. 内存扩充。覆盖技术，交换技术，虚拟存储器。

## 8.2 单一连续区存储管理

内存分为操作系统区和用户区。应用程序装入到用户区，可使用用户区全部空间，和操作系统共享整个内存。一次只能运行一个程序，适用于单用户、单任务的操作系统，实例：MS-DOS，单片机系统。

单一连续区存储管理需要硬件保护机构，以确保用户程序不至于偶然或无意地干扰系统区中的信息。解决方法是：使用界限寄存器，作业运行时，检查内存访问的地址，若不在界限寄存器所限定的范围内，则发生越界中断。

优点：方法简单，对硬件要求低，易于实现。

缺点：程序全部装入，很少使用的程序部分也占用内存，对要求内存空间少的程序，造成内存浪费。

## 8.3 分区存储管理

把内存分为一些大小相等或不等的分区(partition)，每个进程占用一个或几个分区。操作系统占用其中一个分区。适用于多道程序系统和分时系统。

内碎片：占用分区之内未被利用的空间。

外碎片：占用分区之间难以利用的空闲分区(通常是小空闲分区)。

(1) 固定分区（没有外碎片，可能有内碎片）

把内存划分为若干个固定大小的连续分区（大小可以不相等但要事先确定），每个分区的边界固定。

1. 如果分区大小相等：适合于多个相同大小的程序并发执行。
2. 如果分区大小不等：多个小分区、适量的中等分区、少量的大分区。根据程序的大小，分配当前空闲的、适当大小的分区。

(2) 动态分区（没有内碎片，可能有外碎片）

并不预先将内存事先划分成分区，当程序需要装入内存时系统从空闲的内存区中分配大小等于程序所需的内存空间。在进程执行过程中可以通过系统调用改变分区大小。

可使用位图、空闲链表跟踪内存使用情况：

1. 位图：内存划分成小的分配单位，一个分配单位可以小到几个字，大到几KB。1表示占用，0表示空闲。当一个占用 k 个分配单位的进程调入内存时，搜索位图，找出 k 个连续的0。
2. 空闲链表：维护一个记录已分配内存分区和空闲内存分区的链表，链表中的每一项或者表示一个进程，或者表示两个进程间的一个空闲分区。链表表项包括：指示标志空闲区(H)/进程(P)，起始地址，长度，next 指针。

分区分配算法：

1. 首次适配法：按分区的先后次序，从头查找，找到符合要求的第一个分区。分配和释放时间性能较好，但随着低端分区不断划分而产生较多小分区，每次分配时查找时间开销会增大。同时，这小分区（外碎片）可能得不到利用，造成浪费。
2. 下次适配法：按分区的先后次序，从上次分配的分区起查找(到最后分区时再回到开头)，找到符合要求的第一个分区。
3. 最佳适配法：按分区的先后次序，找到其大小与要求相差最小的空闲分区。外碎片较小，但随着运行时间的增长会积累较多小碎片。
4. 最坏适配法：按分区的先后次序，找到最大的空闲分区。系统中出现极小空闲分片的可能性比较小，不易产生小的外碎片。但较大的空闲分区不被保留，对后继的大作业运行不利。

解决外碎片问题，可采用内存紧缩技术。

(3) 伙伴系统(Buddy System)：综合固定分区技术和动态分区技术的优点。

通过不断对分大的空闲分区来获得小的空闲分区，直到获得所需要的内存分区。当一个空闲分区被对分后，其中的一部分称为另一部分的伙伴(buddy)。当内存分区被释放时，尽可能地合并空闲分区。空闲分区的分配和合并都使用2的幂次。

优点：多个空闲链表，可以立即找到空闲分区，算法简单、速度快。

缺点：申请内存总是以 $2^k$ 字节满足要求，存在内碎片。例如申请129字节会分得256字节。只归并伙伴而容易产生外碎片。申请/释放可能会导致连锁切块/合并，影响系统效率，例如当前只有一块空闲，块大小1M，申请400字节，会导致11次切块。

![08-存储器管理(1)_88](https://cdn.jsdelivr.net/gh/DerrickMarcus/picgo_image/images/08-存储器管理(1)_88.png)
