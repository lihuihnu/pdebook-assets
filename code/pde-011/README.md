# pde-011 数值实验

本文实验在 pde-010 的同一个 9×9 二维 Poisson 五点系统上，实际运行 Jacobi 与 Gauss-Seidel 两种定常迭代方法，并逐轮记录残差、迭代误差和中心节点值。

固定系统为

$$A\mathbf U=\mathbf b,$$

其中

$$\mathbf b=\frac1{128}(6,7,6,7,8,7,6,7,6)^T,$$

离散系统解为

$$\mathbf U_h=\frac1{256}(9,12,9,12,16,12,9,12,9)^T.$$

两种方法都从

$$\mathbf U^{(0)}=\mathbf 0$$

开始，并使用同一个停止条件

$$\|\mathbf r^{(m)}\|_\infty<10^{-10},\qquad \mathbf r^{(m)}=\mathbf b-A\mathbf U^{(m)}.$$

## 运行

从仓库根目录运行：

```bash
python3 code/pde-011/src/stationary_iterations.py
```

实际运行环境：

- Python 3.13.5；
- 第三方 Python 依赖：无。

脚本生成：

```text
code/pde-011/results/iteration_history.csv
code/pde-011/results/final_solutions.csv
```

实际运行摘要：

```text
tolerance = 1.0e-10
Jacobi: iterations=60, residual_inf=7.276e-11, iteration_error_inf=6.185e-11
Gauss-Seidel: iterations=31, residual_inf=7.731e-11, iteration_error_inf=3.865e-11
```

这里的迭代轮数只对应当前固定系统、当前逐行编号、零初始猜测和上述停止条件，不作为一般性的算法快慢结论。

## iteration_history.csv

字段为：

```text
method,iteration,residual_inf,iteration_error_inf,center_value
```

其中：

- `iteration=0` 记录零初始猜测；
- `residual_inf` 是 $\|\mathbf r^{(m)}\|_\infty$；
- `iteration_error_inf` 是已知教学解下的 $\|\mathbf U^{(m)}-\mathbf U_h\|_\infty$；
- `center_value` 是中心节点 $k=5$ 的当前值。

停止条件只使用残差。已知的 $\mathbf U_h$ 仅用于事后记录迭代误差，不参与 Jacobi 或 Gauss-Seidel 的更新和停止判断。

## final_solutions.csv

字段为：

```text
method,iterations,k,i,j,U,exact,error
```

它保存两种方法停止时九个内部节点的结果，并按照 pde-010 的

$$k=i+(j-1)3$$

恢复二维下标。

## 正文回归检查

脚本用断言核对：

- $b_5=1/16$；
- $U_{h,5}=1/16$；
- Jacobi 第一轮中心值 $U_5^{(1)}=1/64$；
- Jacobi 第二轮中心值 $U_5^{(2)}=15/512$；
- 零初始猜测的残差最大分量为 $1/16$；
- 两种方法最终都满足统一残差阈值。

当前实验还得到：

- Jacobi 在第 60 轮首次满足停止条件；
- Gauss-Seidel 在第 31 轮首次满足停止条件。

这些是本实验的实际结果，不推广为一般定理。

## 范围

本实验的目的只是让正文中的迭代更新、残差和停止条件真正运行起来。它不讨论谱半径、条件数、SOR、CG、GMRES、预条件或稀疏矩阵存储。

残差曲线必须直接读取 `iteration_history.csv`，不得手工重造数据。
