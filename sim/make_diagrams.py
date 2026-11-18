#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成图 1–5：原理示意图（万向锁 / 叉积几何 / 两个框图 / 收敛律）"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from figstyle import (C_A, C_BAD, C_D, C_M, C_OK, C_T, INK, MUTED,  # noqa: E402
                      apply, save)


# ---------------------------------------------------------------- 绘图小工具
def box(ax, x0, y0, w, h, text, fc="#eef4ff", ec="#8fa9cc", fs=10):
    ax.add_patch(FancyBboxPatch((x0, y0), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.10",
                                fc=fc, ec=ec, lw=1.3, zorder=3))
    ax.text(x0 + w / 2, y0 + h / 2, text, ha="center", va="center",
            fontsize=fs, color=INK, zorder=4, linespacing=1.4)


def arrow(ax, p0, p1, color=INK, lw=1.3, ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=14,
                                 color=color, lw=lw, ls=ls, shrinkA=0, shrinkB=0,
                                 zorder=5))


def route(ax, pts, color=INK, lw=1.3, ls="-"):
    for i in range(len(pts) - 2):
        ax.plot([pts[i][0], pts[i + 1][0]], [pts[i][1], pts[i + 1][1]],
                color=color, lw=lw, ls=ls, zorder=2, solid_capstyle="round")
    arrow(ax, pts[-2], pts[-1], color, lw, ls)


def tag(ax, x, y, text, color, fs, ha="left", va="center", pad=0.25):
    ax.text(x, y, text, color=color, fontsize=fs, ha=ha, va=va, zorder=7,
            bbox=dict(boxstyle="round,pad=%.2f" % pad, fc="white", ec="none", alpha=0.92))


def blank(figsize):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.grid(False)
    return fig, ax


# ------------------------------------------------------------------ 图 1
def fig01(style, s):
    fs = s["fs"] - 1
    fig, ax = plt.subplots(figsize=(s["figsize"][0] * 1.3, s["figsize"][1] * 0.95))
    ax.set_xlim(-0.2, 10.4)
    ax.set_ylim(-0.4, 5.6)
    ax.axis("off")
    ax.grid(False)

    L = 1.55
    ex = np.array([0.866, -0.5]) * L
    ey = np.array([-0.866, -0.5]) * L
    ez = np.array([0.0, 1.0]) * L

    def stick(cxy, vec, color, lw=2.0, ls="-"):
        ax.add_patch(FancyArrowPatch(cxy, (cxy[0] + vec[0], cxy[1] + vec[1]),
                                     arrowstyle="-|>", mutation_scale=17, color=color,
                                     lw=lw, ls=ls, shrinkA=0, shrinkB=0, zorder=4))
        ax.plot([cxy[0]], [cxy[1]], "o", color=INK, ms=5, zorder=6)

    def lab(x, y, text, color, ha="center"):
        ax.text(x, y, text, color=color, fontsize=fs, fontweight="bold", ha=ha,
                va="center")

    # ---- 左：正常三轴
    c1 = (2.15, 3.05)
    ax.text(2.15, 5.35, "① 三轴正交：3 个独立自由度", ha="center",
            fontsize=fs + 1, color=INK, fontweight="bold")
    stick(c1, ex, C_M)
    stick(c1, ey, C_OK)
    stick(c1, ez, C_A)
    tx, ty = c1[0] + ex[0], c1[1] + ex[1]
    lab(tx + 0.18, ty - 0.14, "X  roll", C_M, ha="left")
    tx, ty = c1[0] + ey[0], c1[1] + ey[1]
    lab(tx - 0.18, ty - 0.14, "Y  pitch", C_OK, ha="right")
    lab(c1[0], c1[1] + ez[1] + 0.26, "Z  yaw", C_A)

    # ---- 右：pitch=90°，X 与 Z 重合
    c2 = (7.55, 3.05)
    ax.text(7.55, 5.35, "② pitch = 90°：X 与 Z 压成一根轴", ha="center",
            fontsize=fs + 1, color=INK, fontweight="bold")
    stick(c2, ey, C_OK)
    stick(c2, (0.13, 1.55), C_M)
    stick(c2, (-0.13, 1.55), C_A, ls="--")
    lab(c2[0] + ey[0] - 0.18, c2[1] + ey[1] - 0.14, "Y  pitch", C_OK, ha="right")
    lab(c2[0] - 0.30, c2[1] + 1.80, "X  roll", C_M, ha="right")
    lab(c2[0] + 0.38, c2[1] + 1.80, "Z  yaw", C_A, ha="left")

    ax.annotate("", xy=(c2[0] + 0.13, c2[1] + 1.30), xytext=(c2[0] - 0.13, c2[1] + 1.30),
                arrowprops=dict(arrowstyle="<->", color=C_BAD, lw=1.4))
    tag(ax, c2[0] + 1.60, c2[1] + 0.62, "两根轴重合 → 自由度 3 变 2", C_BAD, fs)
    ax.annotate("", xy=(c2[0] + 0.16, c2[1] + 1.25), xytext=(c2[0] + 1.50, c2[1] + 0.66),
                arrowprops=dict(arrowstyle="->", color=C_BAD, lw=1.2))

    ax.plot([5.0, 5.0], [0.15, 4.95], color="#e6e9ee", lw=1.2, zorder=0)
    ax.text(5.0, 0.05, "万向锁的本质不是“角度算不出来”，而是两个自由度退化成一个——"
                       "此后改 roll 与改 yaw 变成同一件事",
            ha="center", va="bottom", fontsize=fs - 1, color=MUTED)
    return save(fig, "fig01_gimbal", style)


# ------------------------------------------------------------------ 图 2
def fig02(style, s):
    fs = s["fs"] - 1
    fig, ax = plt.subplots(figsize=(s["figsize"][0] * 1.2, s["figsize"][1] * 1.12))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.grid(False)

    def P(x, y, z):
        """等轴测投影：y 轴指向右上（代表纵深）"""
        return np.array(x) + 0.5 * np.array(y), 0.42 * np.array(y) + np.array(z)

    # 单位球与三条大圆
    th = np.linspace(0, 2 * np.pi, 240)
    ax.plot(np.cos(th), np.sin(th), color="#e4e8ed", lw=1.2, zorder=1)
    for a1, a2 in [((1, 0, 0), (0, 1, 0)), ((0, 1, 0), (0, 0, 1)), ((1, 0, 0), (0, 0, 1))]:
        p = np.outer(np.cos(th), a1) + np.outer(np.sin(th), a2)
        xx, yy = P(p[:, 0], p[:, 1], p[:, 2])
        ax.plot(xx, yy, color="#e4e8ed", lw=1.0, zorder=1)

    thd = np.deg2rad(34.0)
    a = np.array([np.sin(thd), 0.0, np.cos(thd)])
    v = np.array([0.0, 0.0, 1.0])
    e = np.cross(a, v)

    def vec(w0, color, lw=2.6, scale=1.0, ls="-"):
        x, y = P(w0[0] * scale, w0[1] * scale, w0[2] * scale)
        ax.add_patch(FancyArrowPatch((0, 0), (x, y), arrowstyle="-|>",
                                     mutation_scale=19, color=color, lw=lw, ls=ls,
                                     shrinkA=0, shrinkB=0, zorder=5))
        return x, y

    vec(v, C_T, lw=2.8)
    vec(a, C_M, lw=2.8)
    vec(e, C_BAD, lw=2.4, scale=0.72)

    # 夹角弧
    tt = np.linspace(0, thd, 60)
    arc = np.outer(np.cos(tt), v) + np.outer(np.sin(tt), a / np.linalg.norm(a) * 0.0)
    arc = np.outer(np.cos(tt), v) + np.outer(np.sin(tt), np.array([1.0, 0, 0]))
    arc /= np.linalg.norm(arc, axis=1, keepdims=True)
    ax.plot(arc[:, 0] * 0.92, arc[:, 2] * 0.92, color=C_A, lw=1.8, zorder=6)

    ax.text(0.20, 0.90, "θ", color=C_A, fontsize=fs + 1, fontweight="bold")
    ax.text(-0.10, 1.02, "v：估计的重力方向\n（R 的第三行）", color=C_T, fontsize=fs,
            ha="right", va="top", linespacing=1.4)
    ex_, ey_ = P(a[0], a[1], a[2])
    ax.text(ex_ + 0.14, ey_ - 0.02, "a：加速度计实测方向\n（已归一化）", color=C_M,
            fontsize=fs, ha="left", va="top", linespacing=1.4)
    X, Y = P(*e * 0.72)
    ax.text(X - 0.10, Y - 0.16, "e = a × v\n|e| ≈ sinθ ≈ θ", color=C_BAD, fontsize=fs,
            ha="right", va="top", linespacing=1.4)

    ax.text(0.0, -1.30, "叉积天生就是“误差轴 + 误差角”：方向给出该绕哪根轴修正，"
                        "模长给出修正多少\n（|a|=|v|=1 时 |e| = sinθ，小角度下就正比于误差角）",
            ha="center", va="top", fontsize=fs - 1, color=MUTED, linespacing=1.5)
    ax.set_xlim(-1.25, 1.55)
    ax.set_ylim(-1.62, 1.55)
    return save(fig, "fig03_cross", style)


# ------------------------------------------------------------------ 图 3
def fig03(style, s):
    fs = s["fs"] - 2
    fig, ax = blank((s["figsize"][0] * 1.22, s["figsize"][1] * 1.12))
    h = 0.86
    y1, y2 = 3.62, 0.78

    box(ax, 0.10, y1, 1.42, h, "MPU6050\n6 轴 IMU", fc="#f2f4f7", ec="#c3c9d2", fs=fs)
    box(ax, 1.78, y1, 1.42, h, "陀螺仪 ω\n(rad/s)")
    box(ax, 3.42, y1, 0.92, h, "⊕", fs=fs + 4)
    box(ax, 4.56, y1, 1.66, h, "dq/dt = ½ q $\\otimes$ ω\n一阶积分", fs=fs - 1)
    box(ax, 6.44, y1, 1.42, h, "归一化\n‖q‖ = 1", fc="#eaf7ee", ec="#9ecfae")
    box(ax, 8.08, y1, 1.58, h, "q → roll/pitch\n/yaw", fc="#fff7e8", ec="#e0c08a", fs=fs)

    box(ax, 1.78, y2, 1.42, h, "加速度计 a\n(g)", fc="#f2f4f7", ec="#c3c9d2")
    box(ax, 3.42, y2, 0.92, h, "归一化\nâ", fc="#eaf7ee", ec="#9ecfae", fs=fs)
    box(ax, 4.56, y2, 1.42, h, "v = R 第三行\n(估计重力)")
    box(ax, 6.20, y2, 1.10, h, "叉积\ne = â × v", fc="#fdecec", ec="#e5a3a7")
    box(ax, 7.52, y2, 1.90, h, "PI 控制器\nKp·e + Ki∫e dt", fc="#f1ecfd", ec="#b6a2e0")

    for x0, x1 in [(1.52, 1.78), (3.20, 3.42), (4.34, 4.56), (6.22, 6.44),
                   (7.86, 8.08)]:
        arrow(ax, (x0, y1 + h / 2), (x1, y1 + h / 2))
    for x0, x1 in [(3.20, 3.42), (4.34, 4.56), (5.98, 6.20), (7.30, 7.52)]:
        arrow(ax, (x0, y2 + h / 2), (x1, y2 + h / 2))

    route(ax, [(0.81, y1), (0.81, y2 + h / 2), (1.78, y2 + h / 2)])
    route(ax, [(7.15, y1), (7.15, 2.30), (5.27, 2.30), (5.27, y2 + h)],
          color=C_A, ls="--")
    tag(ax, 5.40, 2.42, "当前姿态 q（用于算 v）", C_A, fs - 1)
    route(ax, [(9.42, y2 + h / 2), (9.78, y2 + h / 2), (9.78, 3.12), (3.88, 3.12),
               (3.88, y1)], color="#7b4dd8")
    tag(ax, 4.30, 3.30, "角速度修正量 Δω", "#7b4dd8", fs - 1)

    ax.text(0.10, 4.80, "Mahony：把“姿态误差”当角速度，用 PI 拧回去",
            fontsize=fs + 1, color=INK, fontweight="bold")
    return save(fig, "fig02_mahony_block", style)


# ------------------------------------------------------------------ 图 4
def fig04(style, s):
    fs = s["fs"] - 2
    fig, ax = blank((s["figsize"][0] * 1.22, s["figsize"][1] * 1.12))
    h = 0.86
    y1, y2 = 3.62, 0.78

    box(ax, 0.10, y1, 1.42, h, "MPU6050\n6 轴 IMU", fc="#f2f4f7", ec="#c3c9d2", fs=fs)
    box(ax, 1.78, y1, 1.42, h, "陀螺仪 ω\n(rad/s)")
    box(ax, 3.42, y1, 0.92, h, "⊕", fs=fs + 4)
    box(ax, 4.56, y1, 1.76, h, "dq/dt = ½ q $\\otimes$ ω\n− β·grad f 后积分", fs=fs - 1)
    box(ax, 6.52, y1, 1.30, h, "归一化\n‖q‖ = 1", fc="#eaf7ee", ec="#9ecfae")
    box(ax, 8.02, y1, 1.56, h, "q → roll/pitch\n/yaw", fc="#fff7e8", ec="#e0c08a", fs=fs)

    box(ax, 1.78, y2, 1.42, h, "加速度计 a\n(g)", fc="#f2f4f7", ec="#c3c9d2")
    box(ax, 3.42, y2, 1.02, h, "归一化\nâ", fc="#eaf7ee", ec="#9ecfae", fs=fs)
    box(ax, 4.62, y2, 1.30, h, "残差 f(q,â)\n3×1", fc="#fdecec", ec="#e5a3a7")
    box(ax, 6.12, y2, 1.34, h, "grad f = J^T f\n3×4 雅可比")
    box(ax, 7.66, y2, 1.66, h, "−β·grad f / ‖grad f‖\n只算 1 次", fc="#f1ecfd", ec="#b6a2e0",
        fs=fs)

    for x0, x1 in [(1.52, 1.78), (3.20, 3.42), (4.34, 4.56), (6.32, 6.52),
                   (7.82, 8.02)]:
        arrow(ax, (x0, y1 + h / 2), (x1, y1 + h / 2))
    for x0, x1 in [(3.20, 3.42), (4.44, 4.62), (5.92, 6.12), (7.46, 7.66)]:
        arrow(ax, (x0, y2 + h / 2), (x1, y2 + h / 2))

    route(ax, [(0.81, y1), (0.81, y2 + h / 2), (1.78, y2 + h / 2)])
    route(ax, [(7.17, y1), (7.17, 2.34), (5.27, 2.34), (5.27, y2 + h)],
          color=C_A, ls="--")
    tag(ax, 5.40, 2.46, "当前姿态 q（用于算 J 与 f）", C_A, fs - 1)
    route(ax, [(9.32, y2 + h / 2), (9.78, y2 + h / 2), (9.78, 3.12), (3.88, 3.12),
               (3.88, y1)], color="#7b4dd8")
    tag(ax, 4.30, 3.30, "−β·grad f（唯一参数 β）", "#7b4dd8", fs - 1)

    ax.text(0.10, 4.80, "Madgwick：把“姿态误差”写成代价函数，沿负梯度走一步",
            fontsize=fs + 1, color=INK, fontweight="bold")
    return save(fig, "fig05_madgwick_block", style)


# ------------------------------------------------------------------ 图 5
def fig05(style, s):
    fs = s["fs"] - 1
    fig, ax = plt.subplots(figsize=(s["figsize"][0], s["figsize"][1] * 0.95))
    t = np.linspace(0, 6, 600)
    kp, beta = 2.0, 0.08
    pi = 30.0 * np.exp(-kp * t)
    gr = np.maximum(30.0 - 2 * beta * 180 / np.pi * t, 0)

    ax.plot(t, pi, color=C_M, lw=s["lw"] * 1.4, label="PI 反馈：误差 ∝ exp(−Kp·t)")
    ax.plot(t, gr, color=C_D, lw=s["lw"] * 1.4, label="梯度项：误差 ∝ θ0 − 2β·t")
    ax.axhline(0, color=MUTED, lw=0.9)
    ax.set_xlim(0, 6)
    ax.set_ylim(-1.5, 33)
    ax.set_xlabel("时间 t (s)")
    ax.set_ylabel("姿态误差 (°)")
    ax.set_title("两者都是“比例修正”，但形状不同：指数 vs 线性", pad=8)
    ax.annotate("误差越小，PI 修得越慢\n（渐近，永远到不了 0）", xy=(1.05, 4.2),
                xytext=(0.55, 20.5), color=C_M, fontsize=fs,
                arrowprops=dict(arrowstyle="->", color=C_M, lw=1.1))
    ax.annotate("斜率恒为 2β（最大修正角速度）\n误差小了也不减速", xy=(2.45, 13.6),
                xytext=(2.80, 22.5), color=C_D, fontsize=fs,
                arrowprops=dict(arrowstyle="->", color=C_D, lw=1.1))
    ax.legend(loc="upper right")
    return save(fig, "fig07_converge_law", style)


def main():
    for style in ("blog", "wechat"):
        s = apply(style)
        for fn in (fig01, fig02, fig03, fig04, fig05):
            print(fn.__name__, style, "->", fn(style, s))


if __name__ == "__main__":
    main()
