# 第31篇：怎样知道数值解算得对不对

单位区间无量纲问题 `-u''=f`，中心差分、三对角消元；核心仅使用Python标准库。所有结果来自实际运行，故意错误对照明确标注，不能当成目标问题的正确解。

## 运行

在本公共仓库根目录：

```bash
python code/pde-031/src/verify_poisson.py
python code/pde-031/src/plot_verification.py
```

第一条仅依赖标准库；第二条需要Matplotlib。配置为 `config/problems.json`，结果在本目录 `results/`，图在 `code/pde-031/figures/`。本次运行Python3.12.14、Matplotlib3.10.8；不以墙钟时间评价计算成本。

## 问题与对照

三个已知解，各使用N=4,8,16,32,64,128：

- sine：u=sin(πx)，f=π²sin(πx)，边界0、0。
- quadratic：u=1+x+x(1-x)，f=2，边界1、2。
- quartic：u=1+x+x²(1-x)²，f=-2+12x-12x²，边界1、2。

四次问题另外设四类诊断，共18组正确计算、24组故意诊断：

| 标签 | 改动 | 最终状态含义 |
| --- | --- | --- |
| correct | 原问题，三对角消元 | direct_solve只说明完成消元 |
| biased_source | 实际源项加0.1 | 实际装配系统解出，目标问题不同 |
| omit_right | 内部装配未加入右边界2，输出端点仍标2 | 实际装配端点为0，不能从输出端点推断装配正确 |
| fixed_jacobi | 从1+x出发，固定20轮 | iteration_budget，未依据残差宣称收敛 |
| balanced_perturbation | 消元后加0.02sin(2πx) | perturbed_after_solve，返回向量故意不满足局部方程 |

## 误差与残差

内部A对角2、副对角-1，b包含h²源项与移项的边界贡献，残差为b-Au。记录区分b_target与b_used；target_pde_residual_inf把目标残差除以h²。error_inf比较连续准确解的节点；target_algebraic_error_inf比较**目标问题的准确离散解**，错误输入组这一项包含实现偏差；discretization_error_inf只衡量目标离散解与连续解的差别。

error_l2h为sqrt(h∑内部节点误差平方)，不等同连续重构的L²误差。sine准确离散参考为(π²/λ_h)sin(πx_i)，λ_h=4sin²(πh/2)/h²，由正弦加法公式验证；quartic准确离散参考为u_m+h²x(1-x)，quadratic的节点则准确满足格式。这些闭式参考仅用于测量，不参与求解器更新。

q位于半格控制体面，q_i=-(U_(i+1)-U_i)/h。控制体覆盖(h/2,1-h/2)，balance_target=q_last-q_first-h∑f_target。q_used与balance_used使用**实际装配边界**及源项，omit_right右端按0计算；输出数组和实际装配数组在nodes.csv分列保存。不能把这些面通量直接称为物理端点通量。

## 原始结果

| 文件 | 行数 | 内容 |
| --- | --- | --- |
| loads.csv | 1722 | 目标/实际源项、右端、边界，连续与离散参考 |
| history.csv | 162 | 每个状态的残差、误差与停止状态 |
| iterates.csv | 6642 | 逐轮内部向量；Jacobi含初值及全部20轮 |
| nodes.csv | 1806 | 全节点的输出值、装配值、准确值和误差 |
| fluxes.csv | 1764 | 目标输出与实际装配数组的面通量 |
| balances.csv | 1722 | 各控制体局部平衡缺陷 |
| summary.csv | 42 | 每组最终误差、残差、局部/总体平衡与状态 |
| refinement.csv | 35 | 相邻网格差异、两网格误差阶与三网格差异阶 |
| run.txt | 日志 | 设置说明、实际行数与固定四单元结果 |

p_error需要两张网格的已知解误差；p_difference需要三张网格。小于1e-11的误差/差异不估阶，字段为空并保留原因；最后一组差异缺少第三网格，也不填阶数。并非把N=128解当作真解。

源项偏差与相同节点扰动在共同节点间可抵消，故二阶网格间差异不能排除它们。全局平衡也会因局部正负缺陷相消而成立。原始字段保留这些情况，读者可以自行重新计算。
