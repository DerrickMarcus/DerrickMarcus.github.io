# 3 多元随机变量

多元分布的特征函数：

$$
\phi_{\boldsymbol{X}}(\boldsymbol{\omega})=\mathrm{E}\left(\mathrm{e}^{\mathrm{j}\boldsymbol{\omega}^T\boldsymbol{X}}\right)=\mathrm{E}\left(\mathrm{e}^{\mathrm{j}(\omega_1 X_1+\cdots \omega_n X_n)}\right)
$$

由此可推出：独立随机变量的**加和**的特征函数，等于各个分量的特征函数的**乘积**。

混合矩：

$$
\mathrm{E}\left(X_1^{k_1}X_2^{k_2}\cdots X_n^{k_n}\right)=(-\mathrm{j})^{k_1+\cdots+k_n}\frac{\partial^{k_1+\cdots+k_n} \phi_{\boldsymbol{X}}(\omega)}{\partial \omega_1^{k_1}\cdots \partial \omega_n^{k_n}}\bigg|_{\boldsymbol{\omega}=\boldsymbol{0}}
$$

线性变换： $\boldsymbol{A}\boldsymbol{X}+\boldsymbol{b}$ 的特征函数为：

$$
\phi_{\boldsymbol{A}\boldsymbol{X}+\boldsymbol{b}}(\boldsymbol{\omega})=\mathrm{e}^{\mathrm{j}\boldsymbol{\omega}^T\boldsymbol{b}}\phi_{\boldsymbol{X}}(\boldsymbol{A}^T\boldsymbol{\omega})
$$
