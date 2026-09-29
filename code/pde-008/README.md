# pde-008：边界条件怎样进入离散方程

这是《把方程算明白》第 8 篇的公开配套代码。实验比较同一个一维 Poisson 方程在三种右边界条件下形成的离散系统。

连续方程为

$$-u''(x)=2,\qquad 0<x<1.$$

解析函数统一取

$$u_\star(x)=1+x-x^2.$$

三种边界描述为：

- Dirichlet：`u(0)=1, u(1)=1`；
- Dirichlet–Neumann：`u(0)=1, u'(1)=-1`；
- Dirichlet–Robin：`u(0)=1, u'(1)+u(1)=0`。

三组条件对应同一个解析解，因此可以只比较边界条件怎样改变离散系统。

## 运行

从仓库根目录运行：

```bash
python3 code/pde-008/src/solve_boundary_cases.py
```

运行环境：

- Python 3.13.5；
- 第三方 Python 依赖：无。

脚本生成：

```text
code/pde-008/results/boundary_cases.csv
```

## 数值系统

取 `N=4`、`h=1/4`。

Dirichlet 系统直接求 `U_1,U_2,U_3`。

Neumann 与 Robin 系统把右端点量 `U_4` 保留为未知量，并分别增加一条边界方程：

```text
Neumann: U2 - 4 U3 + 3 U4 = -1/2
Robin:   U2 - 4 U3 + (7/2) U4 = 0
```

## 检查

程序使用断言核对：

- 三组内部 Poisson 方程；
- 各自的 Dirichlet / Neumann / Robin 边界条件；
- 全部节点与解析函数 `u_*(x)=1+x-x^2` 的节点值；
- 三种边界处理最终得到完全相同的节点结果。

纯 Neumann 情形只在正文中讨论非唯一性与相容条件，不放进当前求解脚本。
