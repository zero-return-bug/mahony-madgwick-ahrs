#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""数字一致性审计：正文里每个引用仿真的数字，都从 out/*.json 重算一遍再比对。

任何一次改参数而忘了改正文，这里都会报出来。

用法： python sim/_audit_numbers.py
"""
import io
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
MD = os.environ.get("AHR_MD") or os.path.join(
    HERE, os.pardir, "Mahony与Madgwick姿态解算-算法系列02.md")

M = json.load(io.open(os.path.join(HERE, "out", "metrics.json"), encoding="utf-8"))
G = json.load(io.open(os.path.join(HERE, "out", "marg.json"), encoding="utf-8"))
if not os.path.isfile(MD):
    import sys
    sys.exit("SKIP: 需要系列原文 %s（用 AHR_MD=<路径> 指定）" % MD)
md = io.open(MD, encoding="utf-8").read()

checks = []


def c(value, fmt, label):
    checks.append((fmt % value, label))


# §0 / §5.2 主表
c(M["track"]["mahony_rmse_roll"], "%.2f°", "Mahony roll RMSE")
c(M["track"]["mahony_rmse_pitch"], "%.2f°", "Mahony pitch RMSE")
c(M["track"]["madgwick_rmse_roll"], "%.2f°", "Madgwick roll RMSE")
c(M["track"]["madgwick_rmse_pitch"], "%.2f°", "Madgwick pitch RMSE")
c(M["conv"]["mahony_t_1deg"], "%.2f s", "Mahony 收敛")
c(M["conv"]["madgwick_t_1deg"], "%.2f s", "Madgwick 收敛")
c(M["impact"]["mahony_peak"], "%.1f°", "冲击峰值 Mahony")
c(M["impact"]["madgwick_peak"], "%.1f°", "冲击峰值 Madgwick")
c(M["impact"]["adaptive_peak"], "%.1f°", "自适应峰值")
c(M["impact"]["mahony_residual"], "%.1f°", "冲击残余 Mahony")
c(M["impact"]["madgwick_residual"], "%.1f°", "冲击残余 Madgwick")
c(M["yaw"]["mahony_end"], "%.1f°", "60 s yaw Mahony")
c(M["yaw"]["madgwick_end"], "%.1f°", "60 s yaw Madgwick")
c(M["bigbias"]["mahony_rmse"], "%.2f°", "大零偏 Mahony")
c(M["bigbias"]["madgwick_rmse"], "%.2f°", "大零偏 Madgwick")
c(M["sign"]["mahony_ok_rmse"], "%.1f°", "Mahony 正确符号")
c(M["sign"]["mahony_bad_rmse"], "%.1f°", "Mahony 反向符号")
c(M["sign"]["madgwick_ok_rmse"], "%.1f°", "Madgwick 正确符号")
c(M["sign"]["madgwick_bad_rmse"], "%.1f°", "Madgwick 反向符号")

# 参数扫描表
for r in M["sweep_kp"]:
    c(r["rmse_roll"], "%.2f°", "Kp=%g roll" % r["kp"])
    c(r["rmse_pitch"], "%.2f°", "Kp=%g pitch" % r["kp"])
for r in M["sweep_beta"]:
    c(r["rmse_roll"], "%.2f°", "beta=%g roll" % r["beta"])
    c(r["rmse_pitch"], "%.2f°", "beta=%g pitch" % r["beta"])

# β 与最大修正速率
for b in (0.02, 0.05, 0.08, 0.15, 0.30):
    c(abs(round(b * 2 * 57.29578, 1)), "%.1f°/s", "2beta=%g" % b)

# 9 轴
c(G["err9_conv_1deg_s"], "%.1f s", "9 轴收敛")
c(round(G["yaw6_end_deg"]), "%.0f°", "6 轴 25 s yaw")

bad = []
for needle, label in checks:
    if needle not in md:
        bad.append((label, needle))

print("检查 %d 个数字：" % len(checks))
if bad:
    for label, needle in bad:
        print("  ✗ %-22s 正文里找不到 %s" % (label, needle))
    print("结论：%d 处对不上" % len(bad))
    raise SystemExit(1)
print("  ✓ 全部能在正文中找到，且与 out/*.json 一致")
