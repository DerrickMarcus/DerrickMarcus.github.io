# 差错控制

!!! abstract
    重点掌握：
    1. 了解和对比三种典型差错控制体制。
    2. 由最小码距计算纠错位数和检错位数。
    3. 会用典型检错算法进行检错判断，并针对停等 ARQ，分析其吞吐量和差错概率。

差错控制的主要方法：

（1）反馈原信息进行确认

Destination 把从 Source 处收到的 bit 串原样发送给 Source，然后由 Source 来判断：若与所发一致，则发送下一个 bit 串，否则重发这一组 bit 串。

不一定绝对可靠（因为反馈过程中也可能出错），且效率太低。

（2）检错重发（Automatic Repeat Request, **ARQ**）

Source 对准备发送的 bit 串添加冗余位，用于检错。只要有一个 bit 位不对，就需要重发。

降低了通信效率和速率。

（3）前向纠错（Forward Error Correction, **FEC**）

有一些通信系统没有反馈链路（例如电视），或者无法容忍重传的延时，因此 ARQ 方法不适用。而 FEC 的方案是“相信大多数”，Destination 对接收到的 bit 串进行纠错。

---

定义：信息 bit 串映射的结果，称为码字；码字的集合，称码字集合；码字集合连同映射关系，称为码本。

两个 bit 串（或码字）对应位之间相异的 bit 个数称为汉明距离（Hamming Distance）

码字集合中，任意两个码字之间的汉明距离的最小值，称最小汉明距离 $d_{\min}=\displaystyle\min_{i,j} (\mathbf{c}_i,\mathbf{c}_j)$ .

定理：给定码本，其纠错位数 $t$ 和检错位数 $e$ 满足 $t+e+1\leqslant d_{\min}$ .

推论：$2t+1\leqslant d_{\min},\quad e+1\leqslant d_{\min}$ .
