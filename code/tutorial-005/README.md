# 数值求解教程 005：无黏性 Burgers 方程的激波、稀疏波与熵选择

这是《把方程算明白》数值求解教程第 5 篇的**公开读者代码与真实冻结数据**。六幅已审定图片见 [assets/tutorial-005](../../assets/tutorial-005/)。本目录提供一阶 Godunov 有限体积求解器、未经熵修正的 Roe 反例、固定全局耗散系数 α=1 的 Lax–Friedrichs 对照、独立验证器和五组冻结实验输出。

正文的唯一权威稿保存在作者私库；本公共目录不另行维护数学算法，也不会为绘图重算 PDE。

## 环境与快速测试

正式计算环境：Python 3.12.3、NumPy 2.3.5。数值求解、独立验证和测试不依赖 Matplotlib、SciPy 或 PETSc。在已克隆的公共仓库根目录执行：

~~~bash
python3 -m pip install "numpy==2.3.5"
python3 -m unittest discover -s code/tutorial-005/tests -p 'test_*.py' -v
~~~

测试包含共享面守恒、精确单元平均、CFL 门禁、Roe 非熵反例、边界质量账目及五组数值实验回归。测试写入临时目录，不覆盖发布的冻结结果。

## 重新运行五组数值实验

下列命令均在**公共仓库根目录**运行。每次写入事先不存在的新目录；不要覆盖公共 results 目录中的首次冻结科学数据。新运行的 Git SHA 只表示读者当前工作树，而不是作者首次运行时的来源 SHA。

~~~bash
R=code/tutorial-005
SHA="$(git rev-parse HEAD)"

# ET005-01：周期光滑波；破裂后的时刻不构造虚假光滑精确解。
python3 "$R/src/run_ET005_01.py" --config "$R/config/design_v01.json" \
  --out /tmp/tutorial005-reader-ET005-01 --source-sha "$SHA"
python3 "$R/src/validate_ET005_01.py" --config "$R/config/design_v01.json" \
  --out /tmp/tutorial005-reader-ET005-01

# ET005-02：固定左右外态的非周期熵激波。
python3 "$R/src/run_ET005_02.py" --config "$R/config/ET005-02.json" \
  --design "$R/config/design_v01.json" \
  --out /tmp/tutorial005-reader-ET005-02 --source-sha "$SHA"
python3 "$R/src/validate_ET005_02.py" --config "$R/config/ET005-02.json" \
  --design "$R/config/design_v01.json" --out /tmp/tutorial005-reader-ET005-02

# ET005-03：Godunov 稀疏扇与未经熵修正的 Roe 静止非熵阶跃。
python3 "$R/src/run_ET005_03.py" --config "$R/config/ET005-03.json" \
  --design "$R/config/design_v01.json" \
  --out /tmp/tutorial005-reader-ET005-03 --source-sha "$SHA"
python3 "$R/src/validate_ET005_03.py" --config "$R/config/ET005-03.json" \
  --design "$R/config/design_v01.json" --out /tmp/tutorial005-reader-ET005-03

# ET005-04：两个 Riemann 问题 × 四级 N=100,200,400,800 网格。
python3 "$R/src/run_ET005_04.py" --config "$R/config/ET005-04.json" \
  --design "$R/config/design_v01.json" \
  --out /tmp/tutorial005-reader-ET005-04 --source-sha "$SHA"
python3 "$R/src/validate_ET005_04.py" --config "$R/config/ET005-04.json" \
  --design "$R/config/design_v01.json" --archive-root "$R/results" \
  --out /tmp/tutorial005-reader-ET005-04

# ET005-05：Godunov 与全局固定 α=1 的 LF，三个请求 CFL。
python3 "$R/src/run_ET005_05.py" --config "$R/config/ET005-05.json" \
  --design "$R/config/design_v01.json" \
  --out /tmp/tutorial005-reader-ET005-05 --source-sha "$SHA"
python3 "$R/src/validate_ET005_05.py" --config "$R/config/ET005-05.json" \
  --design "$R/config/design_v01.json" --archive-root "$R/results" \
  --out /tmp/tutorial005-reader-ET005-05
~~~

新设备、依赖版本与平台的浮点误差可能略有不同；依靠独立容差审计判断可信度，不保证重新运行的 CSV 与首次冻结字节一致。历史 run.json 的原始生成状态为 computed_unvalidated，独立 checks.json 的 PASS 才代表该批数据正式通过科学审计。

## 冻结数据与算法索引

- src/：14 个 Python 文件，包含 Burgers 数值通量、单元平均有限体积更新、参考解、五组驱动及五个独立验证器。
- tests/：五个测试模块，正式私库验收中共 32 项测试通过。
- config/：共享的 design_v01.json 以及 ET005-02～05 的冻结参数。
- results/ET005-01/：1000 行周期快照、600 行光滑参考平均、200 步质量账目与时刻指标。
- results/ET005-02/：激波单元平均/精确参考和 64 步非周期质量收支。
- results/ET005-03/：两方法 1600 行快照与未熵修正 Roe 的真实非熵负例。
- results/ET005-04/：八组真实网格误差、3000 行终态快照与 946 步账目。
- results/ET005-05/：六组通量/CFL 指标、1200 行快照与 512 步账目。

每组归档包含 run.json、独立 checks.json 及相应的 CSV。公共仓库根目录 manifest.json 记录所有公开副本与私库冻结源的 Git blob 身份。

## 读者需要留意的边界

周期正弦初值首次梯度破裂于 t*=2/π；在其后不使用已经多值的光滑特征反演伪装精确熵解。跨声速问题中未经熵修正 Roe 可以严格守恒却停留在错误静止跃迁，这不代表经过熵修正的 Roe 也必然错误。网格细化时激波从 N=100 到 N=200 的真实单元平均误差**增加**，不可删点或擅画理想收敛阶。当前固定 α=1 的 LF 对照只证明本实验里的现象，不保证任意 CFL 越大总越准确。
