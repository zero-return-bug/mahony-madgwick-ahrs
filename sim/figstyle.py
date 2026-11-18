#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""图表统一样式：一套代码出两个规格（blog 1200px / wechat 1600px）"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
FIG_BLOG = os.path.abspath(os.path.join(ROOT, os.pardir, "figs"))
FIG_WX = os.path.join(FIG_BLOG, "wechat")
for _d in (FIG_BLOG, FIG_WX):
    os.makedirs(_d, exist_ok=True)

for _f in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"):
    if os.path.exists(_f):
        try:
            fm.fontManager.addfont(_f)
        except Exception:
            pass

# 配色（白底稿）
C_M = "#1f6feb"      # Mahony
C_D = "#e8873a"      # Madgwick
C_T = "#555f6b"      # 真值 / 参考
C_BAD = "#d1242f"    # 失效
C_OK = "#2ea043"     # 正常
C_A = "#7b4dd8"      # 辅助
INK = "#1b1f24"
MUTED = "#6b7280"
PANEL = "#f6f8fa"

S = {
    "blog": dict(dpi=150, figsize=(8.0, 4.4), fs=10.5, lw=1.5, ms=4.0, lgd=9.5),
    "wechat": dict(dpi=200, figsize=(8.0, 4.5), fs=15.5, lw=2.6, ms=7.0, lgd=14.0),
}


def apply(style):
    s = S[style]
    plt.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "font.family": "sans-serif",
        "axes.unicode_minus": False,
        "figure.dpi": s["dpi"],
        "savefig.dpi": s["dpi"],
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.facecolor": "white",
        "font.size": s["fs"],
        "axes.labelsize": s["fs"],
        "axes.titlesize": s["fs"] + 1,
        "axes.linewidth": 1.0,
        "axes.edgecolor": "#3a4048",
        "axes.grid": True,
        "grid.color": "#e6e9ee",
        "grid.linewidth": 0.9,
        "legend.fontsize": s["lgd"],
        "legend.frameon": False,
        "xtick.labelsize": s["fs"] - 1,
        "ytick.labelsize": s["fs"] - 1,
        "lines.linewidth": s["lw"],
        "lines.markersize": s["ms"],
    })
    return s


def save(fig, name, style):
    d = FIG_BLOG if style == "blog" else FIG_WX
    p = os.path.join(d, name + ".png")
    fig.savefig(p, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    return p
