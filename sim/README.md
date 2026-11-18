# sim — 文章配图的仿真与绘图脚本

配套《Mahony 与 Madgwick 姿态解算从零讲透》。文中所有数字、所有图都由这里的脚本生成，改参数重跑即可复现。

## 用法

```bash
pip install -r requirements.txt

python sim_ahrs.py       # 6 轴仿真：轨迹跟踪 / 收敛 / 冲击 / yaw 漂移 / 参数扫描 / 符号写反
python sim_marg.py       # 9 轴（MARG）验证：6×4 雅可比与 yaw 收敛（内置两项断言）
python make_diagrams.py  # 图 1、2、3、5、7（原理图），输出两套尺寸
python make_figs.py      # 图 4、6、8–12（数据图），输出两套尺寸
```

## 三个自检脚本（改完公式/代码建议跑一遍）

```bash
python _audit_c.py    # 把正文里的 C 函数机械转成 Python 直接执行，与仿真逐步比对四元数轨迹
python _verify_c.py   # 9 轴那 14 个关键表达式做语法树级比对（符号/系数/括号层级）
python _audit_marg2.py  # 对比三种 MARG 写法的定点偏差，说明为什么必须用 R^T b 那种
```

`_audit_c.py` 覆盖 `mahonyUpdate` / `madgwickUpdate` / `madgwickUpdateMARG` / `getRPY`，偏差应在 1e-9 以内（`getRPY` 约 3e-8，来自 57.29578 这个常数的舍入）。

## 产物

```
out/
  metrics.json      # 6 轴全部指标（文章表格直接取自这里）
  track.csv         # 20 s 轨迹跟踪曲线
  conv.csv          # 30° 初始误差收敛
  impact.csv        # 0.4 g 冲击响应（含自适应增益）
  yaw.csv           # 60 s yaw 漂移
  sign.csv          # 符号写反对照
  marg.json/.csv    # 9 轴 yaw 收敛
../figs/            # 1200 px 博客版
../figs/wechat/     # 1600 px 公众号版
```

## 约定

- 四元数 `q0..q3`，`q` 为"机体系 → 世界系"，欧拉角按 ZYX 顺序（yaw-pitch-roll）提取，与文中 C 代码逐行对应。
- 世界系 `z` 轴向上，加速度计静止读数取 `+z`；9 轴版本里世界系 `x` 轴对齐磁北（与官方 MadgwickAHRS 一致），地磁倾角 60°、向下。
- 陀螺零偏 rad/s，加速度计单位 g。
- 仿真：100 Hz、20 s 正弦轨迹，加速度计噪声 0.06 g，陀螺噪声 0.003 rad/s，初始姿态误差 10°。
- 稳态 RMSE 统计窗口为 5–20 s（跳过初始收敛段）；收敛时间用 0.5 s 滑动平均后的误差判定，避免噪声抖动造成误判。
- 想换自己的数据：把 csv 按同样列名喂进 `make_figs.py` 即可。
