# mahony-madgwick-ahrs

《Mahony 与 Madgwick 姿态解算从零讲透》系列文章的仿真与绘图脚本。
文中所有数字、所有图都由这里生成，改参数重跑即可复现。

仅含 `sim/` 脚本；文章正文与配图不在此仓库。

## 快速开始

```bash
pip install -r sim/requirements.txt

python sim/sim_ahrs.py       # 6 轴仿真 -> sim/out/*.csv + metrics.json
python sim/sim_marg.py       # 9 轴（MARG）验证 -> marg.csv/.json（含两项自检断言）
python sim/make_diagrams.py  # 图 1、2、3、5、7（原理图）
python sim/make_figs.py      # 图 4、6、8–12（数据图）
```

图表默认输出到 `../figs/`（1200 px 博客版）与 `../figs/wechat/`（1600 px 公众号版），
脚本会自动建目录；想改字号线宽改 `sim/figstyle.py`。

## 自检脚本

```bash
python sim/_audit_c.py      # 正文 C 函数机械转 Python，与仿真逐步比对四元数轨迹
python sim/_verify_c.py     # 9 轴 14 个关键表达式做语法树级比对
python sim/_audit_marg2.py  # 三种 MARG 写法的定点偏差对比
python sim/_audit_numbers.py # 校验文中数字与 metrics.json 是否一致
```

前三个加 `_audit_numbers.py` 需要**系列原文 md**（正文不在本仓库）。默认去找
`../Mahony与Madgwick姿态解算-算法系列02.md`，找不到就用环境变量指定：

```bash
AHR_MD=/path/to/Mahony与Madgwick姿态解算-算法系列02.md python sim/_audit_c.py
```

## 约定

- 四元数 `q0..q3`，`q` 为机体系 → 世界系，欧拉角按 ZYX 提取（yaw-pitch-roll），与文中 C 代码逐行对应。
- 世界系 `z` 轴向上，静止加速度计读数 `+z`；9 轴版本世界系 `x` 轴对齐磁北，地磁倾角 60° 向下。
- 陀螺零偏 rad/s，加速度计单位 g。
- 仿真条件：100 Hz、20 s 正弦轨迹，加速度计噪声 0.06 g，陀螺噪声 0.003 rad/s，初始误差 10°。
- 稳态 RMSE 窗口 5–20 s；收敛时间用 0.5 s 滑动平均后判定。

## 环境

Python 3.9+，`numpy>=1.26`、`matplotlib>=3.8`。无其他依赖，不需要硬件。
