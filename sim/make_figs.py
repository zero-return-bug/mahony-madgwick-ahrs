#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成图 6–11：仿真曲线。只读 sim/out/ 下的数据，不硬编码数字。"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt  # noqa: E402
from figstyle import (C_A, C_BAD, C_D, C_M, C_OK, C_T, INK, MUTED,  # noqa: E402
                      apply, save)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")


def load_csv(name, skip=0):
    return np.genfromtxt(os.path.join(OUT, name), delimiter=",", names=True,
                         skip_header=skip, encoding="utf-8")


def M():
    with open(os.path.join(OUT, "metrics.json"), encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------------ 图 6
def fig06(style, s):
    d = load_csv("track.csv")
    m = M()
    t = d["t"]
    fig, axs = plt.subplots(3, 1, figsize=(s["figsize"][0], s["figsize"][1] * 1.55),
                            sharex=True)
    for ax, (k, name) in zip(axs, [("roll", "roll (°)"), ("pitch", "pitch (°)"),
                                   ("yaw", "yaw (°)")]):
        ax.plot(t, d[k + "_t"], color=C_T, lw=s["lw"] * 1.2, ls="--", label="真值")
        ax.plot(t, d[k + "_m"], color=C_M, lw=s["lw"] * 0.9, label="Mahony")
        ax.plot(t, d[k + "_d"], color=C_D, lw=s["lw"] * 0.9, alpha=0.85, label="Madgwick")
        ax.set_ylabel(name)
        ax.margins(x=0.01)
    axs[0].legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0),
                  handlelength=1.6, columnspacing=1.6)
    fig.suptitle("20 s 三轴正弦轨迹跟踪（初始误差 10°，含陀螺零偏与噪声）", y=1.02)
    axs[2].set_xlabel("时间 t (s)")
    fig.text(0.01, -0.015, "注：6 轴方案没有绝对 yaw 参考，yaw 两条线跟着陀螺零偏一起漂——"
                           "这不是算法精度问题，是观测性问题",
             color=MUTED, fontsize=s["fs"] - 3, ha="left")
    fig.align_ylabels(axs)
    return save(fig, "fig08_tracking", style)


# ------------------------------------------------------------------ 图 7
def fig07(style, s):
    d = load_csv("conv.csv")
    m = M()["conv"]
    t, em, ed = d["t"], d["err_mahony"], d["err_madgwick"]
    fig, ax = plt.subplots(figsize=s["figsize"])
    ax.semilogy(t, em, color=C_M, label="Mahony  Kp=2.0")
    ax.semilogy(t, ed, color=C_D, label="Madgwick  β=0.08")
    ax.axhline(1.0, color=C_OK, ls=":", lw=s["lw"] * 0.9)
    ax.text(0.15, 1.12, "1° 门限", color=C_OK, ha="left", fontsize=s["fs"] - 2)
    ax.axvline(m["mahony_t_1deg"], color=C_M, ls="--", lw=s["lw"] * 0.7, alpha=0.6)
    ax.axvline(m["madgwick_t_1deg"], color=C_D, ls="--", lw=s["lw"] * 0.7, alpha=0.6)
    ax.annotate("%.2f s" % m["mahony_t_1deg"],
                xy=(m["mahony_t_1deg"], 3.0), xytext=(m["mahony_t_1deg"] + 0.5, 9),
                color=C_M, fontsize=s["fs"] - 1)
    ax.annotate("%.2f s" % m["madgwick_t_1deg"],
                xy=(m["madgwick_t_1deg"], 3.0), xytext=(m["madgwick_t_1deg"] + 0.5, 22),
                color=C_D, fontsize=s["fs"] - 1)
    ax.set_xlabel("时间 t (s)")
    ax.set_ylabel("姿态误差 (°，对数轴)")
    ax.set_title("30° 初始误差的收敛过程：PI 指数收敛 vs 梯度线性收敛", pad=8)
    ax.set_xlim(0, 8)
    ax.legend(loc="upper right")
    return save(fig, "fig09_init_conv", style)


# ------------------------------------------------------------------ 图 8
def fig08(style, s):
    d = load_csv("impact.csv")
    m = M()["impact"]
    t = d["t"]
    fig, ax = plt.subplots(figsize=s["figsize"])
    ax.axvspan(8.0, 9.0, color="#ffd9a0", alpha=0.55, lw=0)
    ax.plot(t, d["err_mahony"], color=C_M, label="Mahony  Kp=2.0")
    ax.plot(t, d["err_madgwick"], color=C_D, label="Madgwick  β=0.08")
    ax.plot(t, d["err_adaptive"], color=C_OK, ls="-.", label="Mahony + 自适应增益")
    ax.set_xlabel("时间 t (s)")
    ax.set_ylabel("倾斜误差 (°)")
    ax.set_title("0.4 g 水平加速度冲击（阴影 = 冲击持续 1 s）", pad=8)
    ax.set_xlim(6.5, 12)
    ax.set_ylim(0, max(d["err_mahony"].max(), 14) * 1.35)

    ax.annotate("峰值 %.1f°" % m["mahony_peak"],
                xy=(8.75, m["mahony_peak"]), xytext=(9.35, m["mahony_peak"] + 2.6),
                color=C_M, fontsize=s["fs"] - 1,
                arrowprops=dict(arrowstyle="->", color=C_M, lw=1.0))
    ax.annotate("峰值 %.1f°" % m["madgwick_peak"],
                xy=(8.95, m["madgwick_peak"]), xytext=(10.0, m["madgwick_peak"] - 3.4),
                color=C_D, fontsize=s["fs"] - 1,
                arrowprops=dict(arrowstyle="->", color=C_D, lw=1.0))
    ax.annotate("压到 %.1f°" % m["adaptive_peak"],
                xy=(8.55, m["adaptive_peak"]), xytext=(9.5, 11.5),
                color=C_OK, fontsize=s["fs"] - 1,
                arrowprops=dict(arrowstyle="->", color=C_OK, lw=1.0))

    ax2 = ax.twinx()
    k = max(1, int(0.4 / 0.01))
    sm = np.convolve(d["acc_norm"], np.ones(k) / k, mode="same")
    sm[:k] = d["acc_norm"][:k]
    ax2.plot(t, sm, color=MUTED, lw=s["lw"] * 1.2, ls=":", alpha=0.75)
    ax2.set_ylabel("|a| 模长 (g)", color=MUTED)
    ax2.tick_params(colors=MUTED)
    ax2.grid(False)
    ax2.set_ylim(0.75, 1.4)
    ax2.axhline(1.0, color=MUTED, lw=0.8, ls=":")

    ax.legend(loc="upper left", ncol=1)
    return save(fig, "fig10_impact", style)


# ------------------------------------------------------------------ 图 9
def fig09(style, s):
    d = load_csv("yaw.csv")
    m = M()["yaw"]
    t = d["t"]
    fig, ax = plt.subplots(figsize=s["figsize"])
    ax.plot(t, d["gyro_only"], color=C_T, ls="--", lw=s["lw"] * 1.1,
            label="纯陀螺积分（无修正）")
    ax.plot(t, d["mahony"], color=C_M, lw=s["lw"] * 1.1, label="Mahony")
    ax.plot(t, d["madgwick"], color=C_D, lw=s["lw"] * 1.1, ls="-.", label="Madgwick")
    ax.set_xlabel("时间 t (s)")
    ax.set_ylabel("yaw 漂移 (°)")
    ax.set_title("静止 60 s、yaw 零偏 0.57°/s：三条线几乎重合", pad=8)
    ax.set_xlim(0, 60)
    ax.text(1.2, d["gyro_only"][-1] * 0.55,
            "%.1f°/min 的漂移，6 轴算法一点也消不掉" % m["gyro_end"],
            color=INK, fontsize=s["fs"] - 1,
            bbox=dict(boxstyle="round,pad=0.4", fc="#fff6f6", ec="#f1c0c4", lw=0.9))
    ax.legend(loc="upper left")
    return save(fig, "fig11_yaw", style)


# ------------------------------------------------------------------ 图 10
def fig10(style, s):
    m = M()
    kp = m["sweep_kp"]
    bt = m["sweep_beta"]
    fig, axs = plt.subplots(1, 2, figsize=(s["figsize"][0] * 1.25, s["figsize"][1]))
    xs = [r["kp"] for r in kp]
    xi = range(len(kp))
    axs[0].plot(xi, [r["rmse_roll"] for r in kp], "o-", color=C_M, label="roll RMSE")
    axs[0].plot(xi, [r["rmse_pitch"] for r in kp], "s--", color=C_A, label="pitch RMSE")
    axs[0].set_xticks(range(len(xs)))
    axs[0].set_xticklabels(["%g\n(Ki=%g)" % (r["kp"], r["ki"]) for r in kp])
    axs[0].set_xlim(-0.4, len(xs) - 0.6)
    bi = min(range(len(kp)), key=lambda i: kp[i]["rmse_roll"] + kp[i]["rmse_pitch"])
    axs[0].plot([bi], [kp[bi]["rmse_roll"]], "*", ms=s["ms"] * 4, color=C_OK,
                zorder=5, label="最优")
    axs[0].set_xlabel("Mahony 比例增益 Kp")
    axs[0].set_ylabel("稳态 RMSE (°)")
    axs[0].set_title("Mahony：Kp 扫描", pad=6)
    axs[0].legend()

    xi = range(len(bt))
    axs[1].plot(xi, [r["rmse_roll"] for r in bt], "o-", color=C_D, label="roll RMSE")
    axs[1].plot(xi, [r["rmse_pitch"] for r in bt], "s--", color=C_A, label="pitch RMSE")
    axs[1].set_xticks(range(len(bt)))
    axs[1].set_xticklabels(["%g" % r["beta"] for r in bt])
    axs[1].set_xlim(-0.4, len(bt) - 0.6)
    bi = min(range(len(bt)), key=lambda i: bt[i]["rmse_roll"] + bt[i]["rmse_pitch"])
    axs[1].plot([bi], [bt[bi]["rmse_roll"]], "*", ms=s["ms"] * 4, color=C_OK,
                zorder=5, label="最优")
    axs[1].set_xlabel("Madgwick 梯度步长 β")
    axs[1].set_ylabel("稳态 RMSE (°)")
    axs[1].set_title("Madgwick：β 扫描", pad=6)
    axs[1].legend()
    fig.suptitle("参数敏感度：两头都变差，中间有谷（同一组数据、同一套噪声）",
                 y=1.02)
    fig.tight_layout()
    return save(fig, "fig12_sweep", style)


# ------------------------------------------------------------------ 图 11
def fig11(style, s):
    d = load_csv("sign.csv")
    m = M()["sign"]
    t = d["t"]
    fig, axs = plt.subplots(1, 2, figsize=(s["figsize"][0] * 1.25, s["figsize"][1]),
                            sharey=True)
    axs[0].semilogy(t, d["mahony_ok"], color=C_M, label="e = a × v（对）")
    axs[0].semilogy(t, d["mahony_rev"], color=C_BAD, label="e = v × a（错）")
    axs[0].set_title("Mahony：叉积方向写反", pad=6)
    axs[1].semilogy(t, d["madgwick_ok"], color=C_D, label="−β·grad f（对）")
    axs[1].semilogy(t, d["madgwick_rev"], color=C_BAD, label="+β·grad f（错）")
    axs[1].set_title("Madgwick：梯度符号写反", pad=6)
    for ax in axs:
        ax.set_xlabel("时间 t (s)")
        ax.legend(loc="center right")
        ax.set_ylim(0.5, 400)
    axs[0].set_ylabel("姿态误差 (°，对数轴)")
    axs[1].annotate("RMSE %.0f°" % m["madgwick_bad_rmse"], xy=(0.35, 55),
                    color=C_BAD, fontsize=s["fs"] - 2)
    axs[0].annotate("RMSE %.0f°" % m["mahony_bad_rmse"], xy=(3.3, 38),
                    color=C_BAD, fontsize=s["fs"] - 2)
    fig.suptitle("一个乘号/一个减号写错，姿态从能用变成不能用", y=1.02)
    fig.tight_layout()
    return save(fig, "fig04_sign", style)


# ------------------------------------------------------------------ 图 12
def fig12(style, s):
    d = load_csv("marg.csv")
    with open(os.path.join(OUT, "marg.json"), encoding="utf-8") as f:
        marg = json.load(f)
    t = d["t"]
    fig, ax = plt.subplots(figsize=s["figsize"])
    ax.plot(t, d["yaw6"], color=C_T, ls="--", lw=s["lw"] * 1.1, label="6 轴：yaw 一直漂")
    ax.plot(t, d["yaw9"], color=C_OK, lw=s["lw"] * 1.3, label="9 轴：磁力计把 yaw 拉回来")
    ax.axhline(0, color=MUTED, lw=0.9)
    ax.set_xlabel("时间 t (s)")
    ax.set_ylabel("yaw (°)")
    ax.set_title("同一个 30° 初始 yaw 误差：6 轴越漂越远，9 轴收敛到 0", pad=8)
    ax.set_xlim(0, 25)
    ax.annotate("终点 %.0f°" % marg["yaw6_end_deg"],
                xy=(25, marg["yaw6_end_deg"]), xytext=(20.0, marg["yaw6_end_deg"] - 6),
                color=C_T, fontsize=s["fs"] - 1,
                arrowprops=dict(arrowstyle="->", color=C_T, lw=1.0))
    ax.annotate("%.1f s 收敛到 1° 内" % marg["err9_conv_1deg_s"],
                xy=(marg["err9_conv_1deg_s"], 1.0), xytext=(10.5, 11),
                color=C_OK, fontsize=s["fs"] - 1,
                arrowprops=dict(arrowstyle="->", color=C_OK, lw=1.0))
    ax.legend(loc="center right")
    return save(fig, "fig06_marg_yaw", style)


def main():
    for style in ("blog", "wechat"):
        s = apply(style)
        for fn in (fig06, fig07, fig08, fig09, fig10, fig11, fig12):
            print(fn.__name__, style, "->", fn(style, s))


if __name__ == "__main__":
    main()
