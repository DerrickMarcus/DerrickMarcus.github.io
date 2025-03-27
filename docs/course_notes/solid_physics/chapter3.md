---
comments: true
---

# 第三章 固体电子论

## 3.1 索末菲自由电子论

### 3.1.1 波函数与 E-k 关系

令势能 $U=0$ ，薛定谔方程：$-\dfrac{\hbar^2}{2m}\nabla^2\psi(\vec{r})=E\psi(\vec{r})$。

由 $E=\dfrac{p^2}{2m},\vec{p}=\hbar\vec{k}$ ，得到 $E(k)=\dfrac{\hbar^2k^2}{2m},k=\dfrac{\sqrt{2mE}}{\hbar}$ 。

质量 $m$ 越大，抛物线 $E(k)$ 越胖。

波函数为平面波 $\psi_k(\vec{r})=\dfrac{1}{\sqrt{V}}\exp(i\vec{k}\cdot\vec{r})$ （已经归一化）。

### 3.1.2 能级与态密度

晶体的宏观边长为 $L_x,L_y,L_z$ ，原胞三个基矢长度分别为 $a_x,a_y,a_z$ ，沿三个基矢方向的原胞数为 $N_x,N_y,N_z$ ，有 $L_i=N_ia_i\;(i=x,y,z)$ 。

**三维情形**：波函数的波矢 $\vec{k}=\dfrac{2\pi n_x}{L_x}\hat{x}+\dfrac{2\pi n_y}{L_y}\hat{y}+\dfrac{2\pi n_z}{L_z}\hat{z},\;n_x,n_y,n_z\in \mathbb{Z}$ 。每一个离散取值的 $\vec{k}$ 代表一个电子运动可能的状态（即本征态），这些本征态在 $\vec{k}$ 空间排列成点阵，每一个量子态在 $\vec{k}$ 空间所占体积：$\dfrac{(2\pi)^3}{L_xL_yL_z}=\dfrac{8\pi^3}{V}$。

$\vec{k}$ 空间点阵密度为其倒数 $\rho(\vec{k})=\dfrac{V}{8\pi^3}$ ，考虑电子自旋简并度为2，单位 $\vec{k}$ 空间本征态数目 $g(\vec{k})=2\rho(\vec{k})=\dfrac{V}{4\pi^3}$ 。

能量为 $E$ 的球体中电子能态总数为 $Z(E)=g(\vec{k})\dfrac43\pi(\dfrac{\sqrt{2mE}}{\hbar})^3$ ，能态密度为 $N(E)=\dfrac{\mathrm{d}Z}{\mathrm{d}E}=\dfrac{V}{2\pi^2}(\dfrac{2m}{\hbar^2})^{3/2}\sqrt{E}$ 。

二维、一维情形见教材作业题。

## 3.2 周期势场中电子运动状态

### 3.2.1 布洛赫定理

在晶格周期性势场中，电子波函数满足： $\psi(\vec{r}+\vec{R}_n)=e^{i\vec{k}\cdot\vec{R}_n}\psi(\vec{r})$ 。平移一个晶格矢量 $\vec{R}_n$ 后波函数增加相位因子。

### 3.2.2 近自由电子近似

### 3.2.3 紧束缚近似

## 3.3 费米统计分布
