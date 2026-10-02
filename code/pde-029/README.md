# 第29篇：稀疏线性系统与迭代求解

核心算法仅使用Python标准库；绘图另需Matplotlib。输入在config/problems.json。使用无量纲单位区间、零Dirichlet边界，三点中心差分方程整体乘h²；内部矩阵对角为2，上下副对角为−1±ch/2。

五张网格N=4、8、16、24、32，三个问题：

- c=0，f=1，u=x(1−x)/2；
- c=0，f=1+x，u=(4x−3x²−x³)/6；
- c=8，f=1+8(1/2−x)，u=x(1−x)/2。

这些多项式在所用中心格式中有精确节点解；比较的是代数求解误差。一般中心对流格式不由这些例子获得精度或无振荡保证。

Poisson比较Jacobi、CG、完整GMRES与GMRES(12)；非对称问题只运行两种GMRES，共50组PDE线性求解。另保存旋转矩阵的GMRES(1)停滞和完整GMRES两步求解，共52组。初值为零，真实欧氏残差不超过max(10⁻¹⁴,10⁻⁸||b||₂)时停止。GMRES(1)反例故意在6步达到上限，其失败状态保留。

运行：

~~~sh
python code/pde-029/src/sparse_solvers.py
python code/pde-029/src/plot_solvers.py --results code/pde-029/results --output figures/pde-029
~~~

源码先给出正文固定四单元CG函数，再给CSR乘法、Poisson专用Jacobi、CG短递推和GMRES。GMRES使用两遍修改Gram–Schmidt、Givens正交旋转和上三角回代；无预条件。CG只对已知SPD的两个模型调用，曲率检查并不是一般SPD判定器。

六份CSV与run.txt为真实输出：matrix保存实际非零项，loads保存右端与精确节点；history保留每轮真实残差、递推/小问题残差、误差及矩阵乘法计数；iterates保存全部Krylov迭代向量和Jacobi的第0、1、2轮、二的幂检查点及最终向量；nodes为最终节点；summary保存52组状态。每轮误差参考使用已知解析节点，仅用于本实验诊断，实际求解器没有解析解也能按真实残差停止。

矩阵乘法计数包含算法和真实残差检查，不含用于误差诊断的平方和；不报告运行时间排名。CSR非零项省去显式零，列号和行指针仍需存储。

实际环境：Python3.12.14；绘图Matplotlib3.10.8。作者审校使用独立有理数矩阵/载荷、Jacobi正弦谱和成熟实现对照，相关开发依赖不属于读者核心程序。
