# 数值求解教程 004：二维照片去噪的热扩散与正则化扩散

这是《把方程算明白》数值求解教程第 4 篇的**公开读者代码与冻结实验结果**。与正文对应的七幅 PNG 在本仓库的 assets/tutorial-004/ 下。

本篇解决一个具体问题：热方程可以抑制噪点，但继续扩散也会模糊真实轮廓。我们把每个像素作为有限体积单元，使用同一共享面的反向通量推进热扩散和梯度正则化非线性扩散。实际摄影图在人工时间 t=2 时两法 PSNR 均高于输入；到 t=16 又都低于带噪输入。程序不会为美化画面自动裁切灰度、增加锐化或偷偷调整参数。

## 环境

正式私库数值试验使用 Python 3.12.3、NumPy 2.3.5、SciPy 1.17.0、Pillow 11.3.0。建议使用独立虚拟环境，在公共仓库根目录执行：

~~~bash
python3 -m pip install "numpy==2.3.5" "scipy==1.17.0" "pillow==11.3.0"
~~~

运行实验或读者复现代码不需要 Matplotlib。文中正式七张 PNG 已经单独完成出版级验证。

## 最快上手：检查和小系统

~~~bash
python3 -m unittest discover -s code/tutorial-004/tests -p "test_*.py" -v
python3 code/tutorial-004/src/reader_reproduce.py --case check
~~~

第一条用小型数组检查 2×2 手算、常数保持、共享面、显式步长、零通量和时间误差；第二条只检查真实冻结 CSV 的 SHA256、合成图的原始像素与噪声种子，**不运行新的 PDE**。

## 复现正式实验的基础示例

全部命令均在公共仓库根目录执行，结果写入**新建的本地临时目录**，不覆盖本仓库已经发布的正式 CSV。使用当前 checkout 的完整 Git SHA 作为新一次运行的来源标识，切勿冒称自己重跑的输出属于作者首次冻结实验。

~~~bash
# ET004-01：固定空间网格上的余弦模态和显式时间误差。
python3 code/tutorial-004/src/run_ET004_01.py \
  --config code/tutorial-004/config/ET004-01.json \
  --out /tmp/tutorial004-reader-et01 \
  --source-sha "$(git rev-parse HEAD)"

# ET004-02：从固定合成图与噪声重新计算热扩散/正则化扩散。
python3 code/tutorial-004/src/run_ET004_02.py \
  --config code/tutorial-004/config/ET004-02.json \
  --out /tmp/tutorial004-reader-et02 \
  --source-sha "$(git rev-parse HEAD)"

# ET004-05：使用已发布的 CC0 原照片、裁剪及冻结许可/来源信息。
python3 code/tutorial-004/src/run_ET004_05.py \
  --config code/tutorial-004/config/ET004-05.json \
  --fixture-dir code/tutorial-004/data \
  --provenance code/tutorial-004/results/ET004-05/source-provenance.json \
  --out /tmp/tutorial004-reader-et05 \
  --source-sha "$(git rev-parse HEAD)"
~~~

每组重算会执行真实 PDE 更新，较大图像可能需要一定时间。原实验使用 float64，时间步 dt=0.2，主要参数 sigma=1.5、kappa=0.075；非线性系数在每个时间层更新。公开数据源只能用于教学复现，不能因自行重算后的机器误差而替换作者首次冻结的输出。

## 九组参数扫描与时间步细化（ET004-03、04）

原正式计算依赖 ET004-02 保存的超过 5 MB 的无损快照。为了让公共仓库保持轻量，未复制整套 NPZ；而是附带 src/reader_reproduce.py，从**同一生成器与冻结的输入种子**重建合成图和噪声、验证原始像素 SHA256，再调用**完全相同的共享面求解器**逐组重算，并与公开 CSV 逐点比较。

~~~bash
# 纯只读身份核对，立即运行。
python3 code/tutorial-004/src/reader_reproduce.py --case check

# 九个预注册 (sigma, kappa) 点，t=16，绝不自动优化。
python3 code/tutorial-004/src/reader_reproduce.py \
  --case grid --out /tmp/tutorial004-reader-grid

# 原主参数下五个时间步的自收敛比较（耗时比 grid 更长）。
python3 code/tutorial-004/src/reader_reproduce.py \
  --case time --out /tmp/tutorial004-reader-time
~~~

输出的 reader_grid.csv、reader_time.csv 与 reader_report.json **只是你自己运行的比较记录**；冻结官方对照在 results/ET004-03/grid_metrics.csv、results/ET004-04/self_convergence.csv 中。脚本使用一个绝对误差容差比较运行结果，避免把浮点平台差异误当作科学参数优化。

## 目录与原始证据

- src/diffusion_solver.py：唯一共享面有限体积更新、梯度正则化、Neumann 零通量、显式 Euler 和步长合法性。
- src/image_inputs.py、src/metrics.py：固定合成参考、PCG64 噪声与全局/平坦区/矩形边缘指标。
- src/run_ET004_01.py、src/run_ET004_02.py、src/run_ET004_05.py：从原科学实现逐字同步的正式算法驱动，输出到新的路径。
- src/reader_reproduce.py：教学复现 ET004-03 九点扫描和 ET004-04 五级时间步研究，不引用私库大型归档，也不产生新的正式结果。
- src/prepare_ET004_05_camera.py、src/validate_ET004_05.py：原始摄影来源准备及独立检验实现；正式来源验证可能需要 scikit-image==0.25.2。
- tests/test_diffusion_solver.py：与原私库一致的单元测试。
- config/ET004-01.json～ET004-05.json：五组实验冻结配置（原文逐字同步）。
- results/：作者首次正式运行得到的六份透明 CSV 和 ET004-05 的来源/许可 JSON；没有手写预测数据。
- data/camera_original_cc0_512.png、data/camera_clean_cc0.png：scikit-image 0.25.2 的 CC0 摄影原图与 [128:384,128:384] 固定裁剪；摄影署名 Lav Varshney。未经额外加噪的照片**不是**已知的物理无噪声真值。

## 解读边界

真实摄影的同一带噪输入 PSNR 约 21.957 dB；在预先指定的六个时刻中，t=2 的热扩散/正则化扩散分别为 23.500/25.672 dB；t=16 时降至 19.293/20.070 dB。不能据此认定 t=2 是连续时间上的严格最优，也不能认为更高 PSNR 保证每一处边缘完全保持。九组参数中较高的采样值不代表适用于任意照片。

公私图像、代码、配置与结果的准确 Git blob 身份见本仓库根目录 manifest.json。本文完整 Markdown 正文由私有书稿统一维护；本目录不存第二份正文或作者运行日志、工作流、字体文件。
