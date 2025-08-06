# Information Theory 信息论

!!! abstract
    重点掌握：
    1. 给定随机变量的分布，计算熵、联合熵、条件熵。
    2. 给定信源编码，计算平均码长。
    3. 链式法则的应用。
    4. 给定信道，计算互信息和信道容量。
    5. 典型 DMC 的信道容量的计算。
    6. 微分熵的计算。
    7. 用香农公式计算高斯信道的容量。

信息论的基本模型：

<div style="text-align: center;">信源 — 信源编码 — 信道编码 — 信道 — 信道译码 — 信源译码 — 信宿</div>

信源/信宿：人或机器，信息的产生和使用者。信源产生随机过程。

信源编/译码：用尽量少的 0,1 bit 表示和重建信源产生的随机过程，且最小化失真。

信道编/译码：用尽量高的效率（能效，谱效）尽量可靠地传输 0,1 bit。

信道：自然信道（空气，水，存在热噪声），人工信道（网络，存在拥塞、丢包）。

## 信源

对于时间和取值都离散的信源 $X[k]$ ，例如一段文字“通信与网络是一门好课”。为简化讨论，假设 $X[k]$ 为独立同分布的 $i.i.d.$ 随机过程（这对于一般文本不成立，因为文本上下文有关联），称为**离散无记忆信源**（Discrete Memoryless Source, **DMS**）。

记 DMS 信源（取指数有限）为：

$$
X\sim\begin{pmatrix}
x_1 & x_2 & \cdots x_N \\
p_1 & p_2 & \cdots p_N
\end{pmatrix}
$$

概率 $p_i=\mathrm{P}(X=x_i)$ .

信源编码就是将 $X$ 映射为 0,1 bit 串的过程 $f:X\to 01\cdots1$ 。对于不同的 $x_i$ ，根据码字的长度 $l_i$ 是否相等，分为**定长码**和**变长码**。

可解码条件（根据码字译码出信源符号）：

1. 定长码： $f(x_i)\neq f(x_j),\quad \forall i\neq j$ .
2. 变长码： $\forall i\neq j,\quad f(x_i)$ 不是 $f(x_j)$ 的前缀。

平均码长 $\bar{l}=\displaystyle\sum_{i=1}^N p_il_i$ . 在可解码条件下，我们希望**平均码长越小越好**，这样在表达信源 $X$ 的时候需要使用的 bit 数最少，压缩效率最高。

信息论告诉我们，DMS 的最小码长为 $\bar{l}_{\min}=-\displaystyle\sum_{i=1}^N p_i\log_2 p_i$ .

## 熵

对于 DMS，定义其熵（Entrop）为：

$$
H(X)=-\sum_{i=1}^N p_i\log_2 p_i=\mathbb{E}_X\{-\log_2 p_i\}
$$

表示该信源的不确定度（信息量）。由此可见，给定一个特定分布的信源 $X$ ，其最小码长等于熵。

上式的另外一层含义是： $-\log_2 p_i$ 代表事件 $X=x_i$ 蕴含的信息量、不确定度，概率 $p_i$ 越小，其值越大。越稀有的事件，发生时带来的信息量越大。例如，一个好学生早八课几乎此次准时上课，一次不翘。在这一学期的前10次课中他都准时上课，因此准时上课对他来说就是一件非常平常的事情，没有人会认为有什么异常。然而某一天他却翘课了，那么老师和同学们就很难没有疑问：好学生手机关机了闹钟没响？还是生病了无法上课？还是因为违规预约被学校退学了？这样就带来了很多信息量。

对于 $X\sim\begin{pmatrix}
x_1 & x_2 & \cdots x_N \\
p_1 & p_2 & \cdots p_N
\end{pmatrix}$ ，有 $0\leqslant H(X)\leqslant\log_2 N$ ：

1. 当且仅当 $\exists i,\;p_i=1,\;\forall j\neq i,\;p_j=0$ ，也就是 $X$ 完全确定时，有最小值 $H(X)=0$ ，此时没有任何信息量，因为我们完全可以唯一确定 $X$ 的取值。
2. 当且仅当 $\forall i,\;p_i=\dfrac{1}{N}$ 时，有最大值 $H(X)=\log_2 N$ 。均匀分布的是时候熵最大，因此各个取值概率完全公平地均等，没有任何偏袒，无法先验地预测某个取值的概率相比其他的更大。

## 联合熵

对于联合分布列：

$$
\begin{array}{c|cccc}
 & y_1 & y_2 & \cdots & y_N \\
\hline
x_1 & p_{11} & p_{12} & \cdots & p_{1N} \\
x_2 & p_{21} & p_{22} & \cdots & p_{2N} \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
x_M & p_{M1} & p_{M2} & \cdots & p_{MN} \\
\end{array}
$$

联合概率 $p_{ij} = \mathrm{P}(X = x_i, Y = y_j)$ ，类似之前讨论，事件 $\{X = x_i, Y = y_j\}$ 蕴含的信息量为 $-\log \mathrm{P} (X = x_i, Y = y_j) = -\log_2 p_{ij}$ .

则 $\{X,Y\}$ 的联合不确定度为**联合熵**：

$$
H(XY) = \mathbb{E}_{XY}\{-\log_2 \mathrm{P}(X = x_i, Y = y_j) \}
= - \sum_{i=1}^M \sum_{j=1}^N p_{ij} \log_2 p_{ij}
$$

## 条件熵

条件概率 $p_{i|j} = \mathrm{P}(X = x_i | Y = y_j) = \dfrac{\mathrm{P}(X = x_i, Y = y_j)}{\mathrm{P}(Y = y_j)}$ ，类似之前的讨论，条件事件 $\{X = x_i | Y = y_j\}$ 蕴含的信息量为： $-\log_2 \mathrm{P}(X = x_i | Y = y_j) = -\log p_{i|j}$ .

则 $X|Y$ 的平均不确定度为**条件熵**：

$$
H(X | Y) = \mathbb{E}_{XY}\{-\log_2 \mathrm{P}(X = x_i | Y = y_j)\}= - \sum_{i=1}^M \sum_{j=1}^N p_{ij}\log_2 p_{i|j}
$$

熵的物理意义（对通信系统设计的指导意义）：

1. 熵 $H(X)$ ： $X$ 的不确定度。这意味着，单独对信源 $X$ 进行“信源编码——信道传输——信源译码”的过程中，当信道速率（平均码长/次） $\geqslant H(X)$ 时，可在接收端无失真恢复 $X$ .
2. 联合熵 $H(X)$ ： $XY$ 的联合不确定度。这意味着，对联合信源 $X,Y$ 进行“联合信源编码——信道传输——联合信源译码”的过程中，当信道速率（平均码长/次） $\geqslant H(XY)$ 时，可在接收端无失真恢复 $X,Y$ .
3. 条件熵 $H(X|Y)$ ：在给定 $Y$ 的条件下， $X$ 残存的不确定度。这意味着，在编码器和译码器可以**共同观测** $Y$ 时，对信源 $X$ 进行“信源编码——信道传输——信源译码”的过程中，当信道速率（平均码长/次） $\geqslant H(X|Y)$ 时，可在接收端无失真恢复 $X$ .

## 链式法则

$$
H(XY)=H(X|Y)+H(Y)
$$

其物理意义是： $XY$ 的联合不确定度 = 观测 $Y$ 后 $X$ 残存的不确定度 + $Y$ 的不确定度。证明过程较容易。

若 $X,Y$ **独立**，记为 $X\perp Y$ ，则 $p_{ij}=p_ip_j,\;p_{i|j}=p_i$ ，得到 $H(X|Y)=H(X),\;H(XY)=H(X)+H(Y)$ . 此时联合熵等于 $X,Y$ 各自熵的加和，由于 $X,Y$ 独立， $Y$ 的取值不会影响到 $X$ ，因此观测 $Y$ 不会降低 $X$ 的不确定度，也对应了 $H(X|Y)=H(X)$ 这个式子。

若 $X$ 是 $Y$ 的某种**确定性映射**，即 $\exists f,X=f(Y)$ ，通过 $Y$ 可以唯一确定 $X$ 的取值，则 $p_{i|j}=\begin{cases}
1,& x_i=f(y_i) \\
0,& x_i\neq f(y_i)
\end{cases}$ ，得到 $H(X|Y)=0,\;H(XY)=H(Y)$ . 相当于 $X,Y$ 是完全绑定的，通过观测 $Y$ 就能完全确定 $X$ 的取值，完全消除 $X$ 的不确定度。此时 $XY$ 的不确定度也等于 $Y$ 的不确定度。

一些可能用到的公式（从物理意义上比较好理解）：

$$
\begin{gather*}
H(X+Y|Y) = H(X|Y) \\
H(XY) = H(X) + H(Y|X) \\
0 \leqslant H(X|Y) \leqslant H(X) \\
H(XY) \geqslant H(X) \\
H((X+Y)X) = H(XY)
\end{gather*}
$$

## 互信息

定义互信息为 $I(X;Y)=H(X)-H(X|Y)$ ，表示通过观测 $Y$ 能够消除 $X$ 多少的不确定度。

在 Venn 图中，互信息 $I(X;Y)$ 代表 $X,Y$ 两个集合的**交集**，联合熵代表 $X,Y$ 两个集合的**并集**，条件熵 $H(X|Y)$ 代表两个集合的**差集** $X\backslash Y$ .

根据 Venn 图直观理解，不难得出：

1. $I(X;Y)=H(X)-H(X|Y)=H(Y)-H(Y|X)=H(X)+H(Y)-H(XY)$ .
2. $I(Y;X)=I(X;Y)$ .
3. 自信息 $I(X;X)=H(X)$ .

若 $X,Y$ **独立**，则 $I(X;Y)=H(X)-H(X|Y)=0$ ，两个集合的交集为空，观测 $Y$ 无法消除 $X$ 任何的不确定度。

若 $X$ 是 $Y$ 的某种**确定性映射**，即 $\exists f,X=f(Y)$ ，则 $I(X;Y)=H(X)-H(X|Y)=H(X)$ . 观测 $Y$ 可以完全消除 $X$ 的不确定度。

同时我们还有 $0\leqslant I(X;Y)\leqslant \min\{H(X),H(Y)\}$ . 左边 $I(X;Y)\geqslant 0$ 的含义是“不存在欺骗”，即无论 $X,Y$ 的分布、独立性如何，通过观测 $Y$ 不会凭空增加 $X$ 的不确定度。

## 信道容量

**离散无记忆信道**（Discrete Memoryless Channel, **DMC**）模型：

$$
X \longrightarrow \boxed{p_{j|i}} \longrightarrow Y
$$

信道通常表示一个输入符号到输出符号的映射，而输出的概率分布依赖于输入。因此信道使用条件概率 $p_{j|i}$ 描述，表示输入为 $i$ 的时候输出为 $j$ 的概率。信宿处观测 $Y$ ，每观测一次 $Y$ 可以消除信源 $I(X;Y)$ 的个 **bit** 的不确定度。

!!! note
    与前述“编码译码段共同观测 $Y$ 时当信道速率（平均码长/次） $\geqslant H(X|Y)$ 时，可在接收端无失真恢复 $X$ .”不同，这样说是因为如果在发、收两端同时观测 $Y$ 的时候，我们就相当于传输的是 $Z=X|Y$ 这个条件变量，其熵即为 $H(Z)=H(X|Y)$ . 而现在，我们使用条件概率描述信道，且只在接收端/信宿观测 $Y$ ，相当于是已知 $Y$ 的前提下反推 $X$ 的后验概率，根据贝叶斯公式有 $\mathrm{P}(X|Y)=\dfrac{\mathrm{P}(XY)}{\mathrm{P}(Y)}$ 即 $p_{i|j}=\dfrac{p_{ij}}{p_j}$ . 在这里，我们关心的是通过观测 $Y$ 能够消除信源多少的不确定度。

如果给定 $X$ 的分布 $p_i$ ，且 $p_j,p_{ij}$ 的分布给定，则互信息 $I(X;Y)=H(X)+H(Y)-H(XY)$ 给定，信源 $X$ 存在最优的分布 $p_i$ 使得 $I(X;Y)$ 最大化，则 $I(X;Y)$ 的最大值称为**信道容量**（Channel Capacity）： $C=\underset{p_i}{\max}\ I(X;Y)$ .

对于一般的 DMC，其本身由 $p_{j|i}$ 决定，可以将其视为一组常量，优化函数 $C=\underset{p_i}{\max}\ I(X;Y)$ 中 $p_i$ 为优化变量， $p_{j|i}$ 为变量，将其化为只关于 $p_i,p_{j|i}$ 的形式：

$$
\begin{align*}
I(X;Y) &= H(X) + H(Y) - H(XY) \\
&= -\sum_i p_i \log_2 p_i - \sum_j p_j \log_2 p_j + \sum_{i}\sum_{j} p_{ij} \log_2 p_{ij} \\
&= -\sum_{i}\sum_{j} p_{ij} \log_2 p_i - \sum_{i}\sum_{j} p_{ij} \log_2 p_j + \sum_{i}\sum_{j} p_{ij} \log_2 p_{ij} \\
&= \sum_{i}\sum_{j} p_{ij} \log_2 \frac{p_{ij}}{p_i p_j} \\
&= \sum_{i}\sum_{j} p_{ij} \log_2 \frac{p_{j|i}}{p_j} \\
&= \sum_{i}\sum_{j} p_i p_{j|i} \log_2 \frac{p_{j|i}}{\sum_i p_i p_{j|i}}
\end{align*}
$$

优化问题即为：

$$
C=\max_{p_i}\sum_{i}\sum_{j} p_i p_{j|i} \log_2 \frac{p_{j|i}}{\sum_i p_i p_{j|i}},\quad \text{s.t. } \sum_i p_i=1,\;p_i\geqslant 0
$$

这是一个非凸的带约束优化问题，处理起来比较棘手。一般 DMC 的容量由 Blahut-Arimoto 算法给出。

## BSC 信道

**对称二进制信道**（Binary Symmetric Channel, **BSC**）：

$$
X\in\{0,1\}\longrightarrow \boxed{p_{j|i}} \longrightarrow Y\in\{0,1\}
$$

发、收端不一致的概率即差错概率为 $p_{1|0}=p_{0|1}=\varepsilon$ ，正确概率为 $p_{1|1}=p_{0|0}=1-\varepsilon$ .

BSC 信道的一种等效为：

$$
\begin{matrix}
& Z & \\
& \downarrow & \\
X \longrightarrow & \oplus & \longrightarrow Y = X \oplus Z
\end{matrix}
$$

其中 $Z\sim\begin{pmatrix}
0 & 1 \\
1-\varepsilon & \varepsilon
\end{pmatrix}$ ，即 $Z=1$ 时传输发生差错， $Z=0$ 无差错。则信道容量为 $C=\underset{p_i}{\max}\ I(X;X\oplus Z)$ .

由 $I(X;Y) = H(Y) - H(Y|X)$ 知：

$$
\begin{align*}
I(X; X \oplus Z) &= H(X \oplus Z) - H(X \oplus Z | X) \\
&= H(X \oplus Z) - H(Z) \\
&= H(X \oplus Z) + \color{red}{\varepsilon \log_2 \varepsilon + (1 - \varepsilon) \log_2 (1 - \varepsilon)}
\end{align*}
$$

上式的红色部分 $\varepsilon \log_2 \varepsilon + (1 - \varepsilon) \log_2 (1 - \varepsilon)$ 为常数，因此只需最大化 $H(Y) = H(X \oplus Z)$ 即可。由 $Y = X \oplus Z \in \{0,1\},\; H(Y) \leqslant 1$，且：

$$
H(Y) = 1 \iff Y \sim\begin{pmatrix}
0 & 1 \\
\frac{1}{2} & \frac{1}{2}
\end{pmatrix} \iff X \sim\begin{pmatrix}
0 & 1 \\
\frac{1}{2} & \frac{1}{2}
\end{pmatrix}
$$

由此得到，对于 BSC 信道，当 $X \sim\begin{pmatrix}
0 & 1 \\
\frac{1}{2} & \frac{1}{2}
\end{pmatrix}$ 时，互信息 $I(X;Y)$ 最大化，信道容量为 $C=1+\varepsilon \log_2 \varepsilon + (1 - \varepsilon) \log_2 (1 - \varepsilon)$ .

![202508032046530](https://cdn.jsdelivr.net/gh/DerrickMarcus/picgo-image/images/202508032046530.png)

!!! tip
    上述等效思想十分重要，在后面的电平信道和波形信道中同样会见到这种思想的运用。

## 微分熵

对于连续分布的信源（例如语音）和连续输入输出的信道（例如高斯信道），我们需要刻画连续分布信源的“**相对不确定度**”，并计算连续输入输出信道的容量。

对于连续随机变量 $X$ ，其概率密度函数 PDF 为 $p_X(x)$ ，定义其微分熵为：

$$
h(X)=\mathbb{E}\{-\log_2 X\}=-\int_{-\infty}^{+\infty}p_X(x)\log_2 p_X(x)\mathrm{d}x
$$

对于均匀分布的 $X\sim\mathcal{U}([0,A])$ ，微分熵为 $h(X)=\displaystyle\int_{0}^{A}\frac{1}{A}\log_2 \dfrac{1}{A}\mathrm{d}x=\log_2 A$ . 可见 $A$ 越大，也就是 $X$ 分布的区间长度越大，其 不确定度越大。

!!! tip
    上式中当 $A<1$ 时 $h(X)<0$ ，也就是微分熵可以为负数，因为其描述的是“相对不确定度”而非“绝对不确定度”。

对于高斯分布的随机变量 $X\sim\mathcal{N}(\mu,\sigma^2)$ ，其微分熵为：

$$
\begin{align*}
p_X(x) &= \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}} \\
h(X) &= -\int_{-\infty}^{\infty} p_X(x) \log_2 p_X(x) \mathrm{d}x \\
&= -\int_{-\infty}^{\infty} \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}} \log_2 \left[\frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}\right] \mathrm{d}x \\
&= -\int_{-\infty}^{\infty} \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{x^2}{2\sigma^2}} \log_2 \left[\frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{x^2}{2\sigma^2}}\right] \mathrm{d}x \qquad \text{let } \tilde{p}_X(x)=\frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{x^2}{2\sigma^2}}\sim\mathcal{N}(0,\sigma^2) \\
&= -\left[\int_{-\infty}^{\infty} \tilde{p}_X(x) \mathrm{d}x\right] \times \log_2 \frac{1}{\sqrt{2\pi\sigma^2}} + \frac{\log_2 e}{2\sigma^2} \int_{-\infty}^{\infty} \tilde{p}_X(x) x^2 \mathrm{d}x \\
&= -1\times  \log_2 \frac{1}{\sqrt{2\pi\sigma^2}} + \frac{\log_2 e}{2\sigma^2}\times \mathbb{E}(X^2) \\
&= \frac{1}{2} \log_2(2\pi\sigma^2) + \frac{1}{2} \log_2 e \\
&= \log_2\left(\sqrt{2\pi e\sigma^2}\right)
\end{align*}
$$

另外，我们还可以推导出：**在所有具有给定方差** $\sigma^2$ **的实值随机变量中，高斯分布具有最大的微分熵**。即 $\mathrm{Var}(X)=\sigma^2\Rightarrow h(X)\leqslant \log_2\left(\sqrt{2\pi e\sigma^2}\right)$ ，等号成立当且仅当 $X\sim\mathcal{N}(\mu,\sigma^2)$ 为高斯分布，且与均值 $\mu$ 无关。

首先有约束条件：概率归一化和方差已知

$$
\int_{-\infty}^{+\infty}p_X(x)\mathrm{d}x=1,\quad \int_{-\infty}^{+\infty}(x-\mu)^2p_X(x)\mathrm{d}x=\sigma^2
$$

构造拉格朗日函数：

$$
L(p_X,\lambda_1,\lambda_2)=-\int_{-\infty}^{+\infty}p_X(x)\log_2 p_X(x)\mathrm{d}x + \lambda_1\left(\int_{-\infty}^{+\infty}p_X(x)\mathrm{d}x-1\right) + \lambda_2 \left(\int_{-\infty}^{+\infty}(x-\mu)^2p_X(x)\mathrm{d}x-\sigma^2\right)
$$

此处我们使用**泛函分析中的变分法**，因为微分熵可以看作是“概率密度函数的函数”，称之为泛函：输入为一个函数，输出为一个数值的函数。要求这一个泛函最大化，且概率密度函数 $p_X(x)$ 是一个未知的函数，我们需要使用变分导数，对拉格朗日函数 $L$ 求 $p_X$ 的**变分导数**：

$$
\frac{\delta L}{\delta p_X}=-1-\log_2 p_X(x)+\lambda_1+\lambda_2(x-\mu)^2=0 \implies p_X(x)=C_1e^{C_2(x-\mu)^2}
$$

通过归一化的约束，得到 $p_X(x)$ 就是一个高斯分布。

---

对于连续幅值的联合信源 $XY\sim p_{XY}(x,y)$ ，定义**联合微分熵**为：

$$
h(XY) = -\int_{-\infty}^{\infty}\int_{-\infty}^{\infty} p_{XY}(x, y) \log_2 p_{XY}(x, y) \mathrm{d}x\mathrm{d}y
$$

对于连续幅值的条件随机变量 $X|Y\sim p_{X|Y}=\dfrac{p_{XY}(x,y)}{P_Y(y)}$ ，定义**条件微分熵**：

$$
h(X|Y) = -\int_{-\infty}^{\infty}\int_{-\infty}^{\infty} p_{XY}(x, y) \log_2 p_{X|Y}(x, y) \mathrm{d}x\mathrm{d}y
$$

同样有**链式法则**： $h(XY)=H(X)+H(Y|X)=h(Y)+h(X|Y)$ .

对于信道：

$$
X\longrightarrow\boxed{p_{Y|X}(x,y)} \longrightarrow Y
$$

互信息 $I(X;Y)=h(X)+h(Y)-h(XY)=h(X)-h(X|Y)=h(Y)-h(Y|X)$ . 其意义是通过观测 $Y$ 能够消除的 $X$ 的不确定度。这里的 $I(X;Y)\geqslant 0$ 具有“绝对”的含义，而非“相对”的含义。因为无论是离散情况还是连续情况，互信息衡量的都是不确定性的减少，而减少的量一定是非负的。对于真实的概率系统，信息不会反向流失。

信道容量为 $C=\underset{p_X(x)}{\max}\ I(X;Y)$ .

## 香农公式

首先考虑**高斯信道**：

$$
\begin{matrix}
& Z & \\
& \downarrow & \\
X \longrightarrow & \oplus & \longrightarrow Y=X+Z
\end{matrix}
$$

$Z\sim\mathcal{N}(\mu,\sigma^2)$ 为高斯分布， $X$ 的分布已知具有功率约束 $\mathbb{E}(X^2)=E_S$ .

计算：

$$
\begin{align*}
C &= \max_{p_X(x)} I(X;Y) \\
&= \max_{p_X(x)} h(Y) - h(X + Z|X) \\
&= \max_{p_X(x)} h(Y) - h(Z|X) \\
&= \max_{p_X(x)} h(Y) - h(Z) \\
&= \max_{p_X(x)} h(X + Z) - \log_2 \left(\sqrt{2\pi e \sigma^2}\right)
\end{align*}
$$

只需最大化 $h(X+Z)$ ，且由于 $X,Z$ 独立：

$$
\mathrm{Var}(X+Z)=\mathrm{Var}(X)+\mathrm{Var}(Z)=\mathbb{E}(X^2)-(\mathbb{E}(X))^2+\sigma^2 \leqslant E_S+\sigma^2
$$

首先，微分熵不依赖于均值，只依赖于分布的形状，因此为了使微分熵最大化，我们令 $X$ 为零均值的： $\mathbb{E}(X)=0$ ，此时 $\mathrm{Var}(X+Z)$ 的方差为定值，根据前述结论，给定方差时高斯分布的微分熵最大，因此 $X+Z$ 为高斯分布 $X+Z\sim\mathcal{N}(0,E_S+\sigma^2)$ . 再根据 $X,Z$ 独立，相减得到 $X$ 也为高斯分布 $X\sim\mathcal{N}(0,E_S)$ ，此时 $h(X+Z)_{\max}=\log_2\sqrt{2\pi e(E_S+\sigma^2)}$ ，带入得到：

$$
C=\log_2\sqrt{2\pi e(E_S+\sigma^2)} - \log_2 \sqrt{2\pi e \sigma^2}=\frac{1}{2}\log_2\left(1+\frac{E_S}{\sigma^2}\right) \quad \mathsf{bit\ per\ channel\ use}
$$

进一步推广到随机过程的形式：**加性高斯白噪声信道**（Additive White Gaussian Noise Channel）

$$
\begin{matrix}
& N(t) & \\
& \downarrow & \\
X(t) \longrightarrow\boxed{\mathrm{LPF}_W(f)}\longrightarrow & \oplus & \longrightarrow Y(t)=X(t)+N(t) \\
\mathsf{low\ pass\ filter\ with\ band\ width\ } W & &
\end{matrix}
$$

加性高斯白噪声AWGN $N(t)$ 的功率谱密度 $S_N(f)=\dfrac{n_0}{2}$ .

由 Nyquist 准则（后面的章节会介绍），单位时间内最多可以使用高斯信道 $2W$ 次，即 $2W$ channel use / second，信道使用的间隔为 $T=\dfrac{1}{2W}$ .

通信功率 $P$ 为单位时间内的能耗 $P=\dfrac{E_S}{T}\Rightarrow E_S=\dfrac{P}{2W}$ ，带入之前高斯信道推导得到的结论，得到大名鼎鼎的**香农公式**：

$$
C=2W\frac{1}{2}\log_2\left(1+\frac{E_S}{\sigma^2}\right)=\boxed{W\log_2\left(1+\frac{P}{Wn_0}\right)} \quad\mathsf{bit/second}
$$

其中可以定义信**信噪比**（Signal-to-Noise Ratio, **SNR**）： $\mathrm{SNR}=\dfrac{P}{Wn_0}$ .

<br>

**低 SNR 区**，即 $\dfrac{P}{Wn_0} \to 0$ ，由泰勒展开：

$$
C = W \frac{P}{Wn_0} \log_2 e = 1.44 \frac{P}{n_0}
$$

容量 $C$ 与功率 $P$ 呈线性关系，两边乘以传输时长 $T$ ，则：

$$
CT = 1.44 PT = 1.44 \frac{E}{n_0}
$$

前沿应用：超宽带通信 UWB（Ultra Wide Band），高能效通信 Green Com.

**高 SNR 区**，即 $\dfrac{P}{Wn_0} \gg 1$，则 $1 + \dfrac{P}{Wn_0} \approx \dfrac{P}{Wn_0}$ ：

$$
C = W \log_2 \mathrm{SNR} = \frac{W}{10} \log_2 (10\cdot \mathrm{SNR}_{\text{dB}}) = 0.33 W \cdot\mathrm{SNR}_{\text{dB}}
$$

容量 $C$ 与带宽 $W$ 呈线性关系。前沿应用：多天线通信 MIMO（等效提升  $W$ ），密集蜂窝网络（空间复用提升 $W$ ）。

## 附录

离散随机变量最大熵的证明方法：

证明 1：

引理：Jenson 不等式，对于上凸函数 $f(x)$ ，若 $\sum_{i=1}^N\lambda_i=1$ ，有：

$$
\sum_{i=1}^N\lambda_i f(x_i)\leqslant f\left(\sum_{i=1}^N\lambda_i x_i\right)
$$

对于 $f(x)=\log_2x,\;f''(x)<0$ ，也即 $f(x)$ 为上凸函数，则：

$$
H(x)=-\sum_{i=1}^N p_i\log_2 p_i=\sum_{i=1}^N p_i\log_2\left(\frac{1}{p_i}\right)\leqslant \log_2\left(\sum_{i=1}^N p_i\cdot\dfrac{1}{p_i}\right)=\log_2N
$$

当且仅当 $p_1=\cdots=p_N=\dfrac{1}{N}$ 时， $H(x)_{\max}=\log_2 N$ .

---

证明 2：

由于有约束条件 $\sum_{i=1}^N p_i=1$ ，考虑使用拉格朗日乘子法。拉格朗日函数为：

$$
L(p_1,\cdots,p_N,\lambda)=-\sum_{i=1}^N p_i\log_2 p_i+\lambda\left(\sum_{i=1}^N p_i-1\right)
$$

求偏导得到 $\dfrac{\partial L}{\partial p_i}=-(\log_2 p_i+\dfrac{1}{\log2})+\lambda=0$ ，因此拉格朗日函数求得最大值时有 $p_1=\cdots=p_N=\dfrac{1}{N}$ ， $H(x)_{\max}=\log_2 N$ .

---

证明 3：

假设有任意参数 $q_i,\cdots,q_N$ ，根据常见不等式 $x>0\implies\log x\leqslant x-1$ 有：

$$
\sum_{i=1}^N p_i\log_2\left(\frac{q_i}{p_i}\right)\leqslant
\frac{1}{\log_2}\sum_{i=1}^N p_i\left(\frac{q_i}{p_i}-1\right)=\frac{1}{\log_2}\sum_{i=1}^N\left(q_i-p_i\right)=\frac{1}{\log_2}\left(\sum_{i=1}^N q_i-1\right)
$$

我们令 $q_i=\dfrac{1}{N}$ ，则化为：

$$
H(X)-\log_2 N \leqslant0 \implies H(X)\leqslant \log_2 N
$$

$H(x)_{\max}=\log_2 N$ ，取等条件为 $\forall i,\;p_i=\dfrac{1}{N}$ .
