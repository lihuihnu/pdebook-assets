# 第30篇：非线性方程与 Newton 迭代

核心程序仅用Python标准库；绘图另需Matplotlib。单位区间、零Dirichlet边界，内部未知量n=N−1。方程−u″+μu³=f按h²整体缩放，F=A U+μh²U³−b，A的对角为2、副对角为−1。Jacobian对角为2+3μh²U²、副对角为−1，每轮用前向消元和回代解修正。

已知连续解4x(1−x)也是本格式的精确节点解；右端f=8+μ[4x(1−x)]³。解析节点只用于输出误差诊断，Newton方向、回溯和停止条件不使用解析解。

config/problems.json包含五网格N=2、4、8、16、32，三组μ=0、10、1000，两种内部常数初值0和4，两个更新规则full与damped，共60组PDE求解。边界始终为零，内部初值4并不改变边界条件。

外层在||F||₂≤max(atol,rtol||b||₂)时停止；rtol=10⁻¹⁰、atol=10⁻¹³、step_tol=10⁻¹⁴，最多80次接受修正。阻尼从1开始减半，sigma=10⁻⁴，每轮至多减半20次，实际试算包括1到2⁻²⁰的21个候选。小步长而大残差报告stagnation；耗尽回溯次数报告line_search_failed。非有限数、线性消元失败和达到轮数上限使用独立状态。

另有五组全步标量计算：三个正/负平方根初值、一个零导数和一个两点循环。标量残差尺度为2；后两组故意保留zero_derivative与maxiter失败。本文不声称Armijo能使循环例子从零初值到达根。

运行：

~~~sh
python code/pde-030/src/newton.py
python code/pde-030/src/plot_newton.py --results code/pde-030/results --output figures/pde-030
~~~

九份CSV和run.txt为实际输出：

- loads：171个内部载荷、坐标与精确节点；
- history：413个迭代点的非线性残差、误差、接受步长、线性残差、回溯次数与残差计算累计次数；第零轮没有接受步字段；
- iterates：全部4813个迭代分量；
- linear_steps：4129个分量的实际F、Jacobian对角、Newton修正及Jδ+F；副对角固定为−1；
- trials：397个实际候选的步长、残差、残差平方函数、Armijo上界与接受状态；k为修正前的迭代点，trial=0表示全步；
- nodes：804个最终节点，包含两个零边界；
- summary：60组PDE停止状态、轮数、残差计算次数与最终误差；
- scalar_history、scalar_summary：26个标量迭代点与五组状态；
- run.txt：Python版本、配置、输出行数和各组实际状态。

history第k+1行的lambda、linear_residual、step_norm对应从第k点到第k+1点的更新。step_norm是||λδ||₂。接受后的非线性残差来自候选点的实际重新计算；它与本轮线性残差不同。残差计算计数包括初始检查和被拒绝的试算，不包括误差诊断、Jacobian装配或线性消元，不用作耗时排名。

实际环境Python3.12.14，绘图Matplotlib3.10.8。独立审校使用有理数模型与Jacobian恒等式、NumPy2.3.5稠密线性求解，以及四个选定系统的70位Decimal稠密主元消元；这些是开发检查依赖，不是读者核心程序依赖。
