# pde-010 数值实验

本文实验把二维 Poisson 方程的五点格式真正装成一个小型线性系统，重点核对二维节点编号、五点邻接和 Dirichlet 边界进入右端的规则。

连续问题为

$$-\Delta u=2[x(1-x)+y(1-y)],\qquad (x,y)\in(0,1)^2,$$

边界条件为

$$u=0\qquad\text{on }\partial\Omega,$$

解析函数取

$$u_\star(x,y)=x(1-x)y(1-y).$$

取 `N=4`，所以每个方向有 `3` 个内部节点，共有 `9` 个未知量。

## 运行

从仓库根目录运行：

```bash
python3 code/pde-010/src/solve_poisson_2d.py
```

本轮实际运行环境：

- Python 3.13.5；
- 第三方 Python 依赖：无。

脚本生成：

```text
code/pde-010/results/n4_solution.csv
code/pde-010/results/n4_stencil_rows.csv
```

实际运行摘要：

```text
N=4, interior_unknowns=9
max nodal error = 6.939e-18
max residual = 3.469e-17
nonzero-Dirichlet regression max error = 4.441e-16
```

这些量处于双精度浮点舍入误差量级。

## 编号

正文采用一维数学编号

$$k=i+(j-1)n,\qquad n=N-1,$$

其中 `k` 从 `1` 开始。

Python 列表从 `0` 开始，因此程序用

```python
def vector_index(i, j, n):
    return (j - 1) * n + (i - 1)
```

作为存储下标，并用 `mathematical_index()` 恢复正文中的 `1` 起始编号。

## 装配

每个内部节点的对角系数为 `4`。若上下左右邻点仍是内部节点，就在对应列写入 `-1`；若邻点落在 Dirichlet 边界，则把已知边界值移到右端。

主算例使用零边界，但程序另用

$$u(x,y)=1+x+y,\qquad -\Delta u=0$$

做非零 Dirichlet regression，以避免零边界把右端装配中的符号错误掩盖掉。

## 检查

程序使用断言核对：

- `N=4` 时共有 9 个内部未知量；
- 九个 `(i,j) -> k` 映射与正文完全一致；
- `N=4` 的 9×9 五点系数矩阵逐项正确；
- 编号连续的 `(3,1)->3` 与 `(1,2)->4` 不产生错误耦合；
- 中心节点 `k=5` 只与 `k=2,4,5,6,8` 发生非零耦合；
- 解出的九个节点值与解析函数一致到浮点容差；
- `A U-b` 的最大残差处于浮点舍入误差量级；
- 非零 Dirichlet regression 通过。

## 结果文件

`n4_solution.csv` 保存：

```text
k,i,j,x,y,U,exact,error
```

`n4_stencil_rows.csv` 保存：

```text
k,i,j,left_k,right_k,down_k,up_k,rhs
```

其中不存在的内部邻点留空。这个文件可以直接检查二维物理邻接与一维编号之间的关系。

## 范围

这里仍使用普通二维列表保存 9×9 矩阵，并沿用前文的小系统高斯消元。这样做只是为了让代码直接对应本篇的五点装配过程；稀疏矩阵存储、Jacobi/Gauss–Seidel 以及更大的线性系统求解属于后续内容。
