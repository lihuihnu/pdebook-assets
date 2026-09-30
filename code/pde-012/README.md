# 一维热方程显式推进

无量纲 u_t=u_xx，0<x<1，零 Dirichlet 边界，初值 sin(pi*x)。

在本目录执行 `python src/heat_explicit.py`。配置见 config/heat.json；N=20，M=100，T=0.1，r=0.4。Python 3.12.14，无第三方求解依赖。results/history.csv 保存全部101层；summary.csv 保存四个时刻；run.txt 保存真实运行日志。误差为相对解析解 exp(-pi²t)sin(pi*x) 的最大节点绝对误差。

脚本实际检查手算一步、旧层不变、边界、非负性、最大值以及离散正弦模态闭式结果。参数守卫限定本入门实验 r<=0.5，并非拒绝所有其他方法。
