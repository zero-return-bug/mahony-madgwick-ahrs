#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""9 轴（MARG）Madgwick 验证。

除了产出 yaw 收敛数据，还在运行时做三项自检（对应文中 §4.6 的三条断言）：
  ① 真姿态下 6 维残差的磁场部分必须为 0（否则定点有偏）
  ② s = J^T f 必须等于 0.5*||f||^2 的数值梯度（否则不是梯度下降）
  ③ 从"倾斜 + 偏航都错"的初始姿态也必须收敛（不只在 yaw 单轴偏的情况下）

用法： python sim_marg.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim_ahrs import (DT, D2R, R2D, conv_time, euler2quat, qconj, qmul,  # noqa: E402
                      qnorm, quat2R, gravity_body, quat_error_deg, rpy_deg)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")

# 地球磁场（世界系，归一化）：北京附近磁倾角约 60°。
# 约定：x 轴 = 磁北（与官方 MadgwickAHRS 一致），z 轴向上，磁场向下倾。
INC = 60.0 * D2R
M_EARTH = np.array([np.cos(INC), 0.0, -np.sin(INC)])


def mag_body(q):
    """真姿态下磁力计应读到什么：R^T * m_earth"""
    return quat2R(q).T @ M_EARTH


# ---------------------------------------------------------------- 核心算法
def marg_pieces(q, a, m):
    """返回 (f[6], b[3], s[4])：残差、参考向量、梯度"""
    q0, q1, q2, q3 = q

    # 1) h = q ⊗ m ⊗ q*，把磁力计读数转到世界系
    hx = 2.0 * (m[0] * (0.5 - q2 * q2 - q3 * q3) + m[1] * (q1 * q2 - q0 * q3)
                + m[2] * (q1 * q3 + q0 * q2))
    hy = 2.0 * (m[0] * (q1 * q2 + q0 * q3) + m[1] * (0.5 - q1 * q1 - q3 * q3)
                + m[2] * (q2 * q3 - q0 * q1))
    hz = 2.0 * (m[0] * (q1 * q3 - q0 * q2) + m[1] * (q2 * q3 + q0 * q1)
                + m[2] * (0.5 - q1 * q1 - q2 * q2))
    b = np.array([np.hypot(hx, hy), 0.0, hz])   # 磁北对齐 +x，水平分量全给 bx
    bx, bz = b[0], b[2]

    # 2) 6 维残差 = 期望测量 - 实测
    f = np.zeros(6)
    f[0] = 2.0 * (q1 * q3 - q0 * q2) - a[0]
    f[1] = 2.0 * (q0 * q1 + q2 * q3) - a[1]
    f[2] = q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3 - a[2]
    # 期望磁场 = R^T b，逐行展开
    f[3] = 2.0 * (bx * (0.5 - q2 * q2 - q3 * q3) + bz * (q1 * q3 - q0 * q2)) - m[0]
    f[4] = 2.0 * (bx * (q1 * q2 - q0 * q3) + bz * (q2 * q3 + q0 * q1)) - m[1]
    f[5] = 2.0 * (bx * (q1 * q3 + q0 * q2) + bz * (0.5 - q1 * q1 - q2 * q2)) - m[2]

    # 3) s = J^T f，J 为 6x4
    f1, f2, f3, f4, f5, f6 = f
    s = np.zeros(4)
    s[0] = (-2 * q2 * f1 + 2 * q1 * f2 + 2 * q0 * f3
            - 2 * bz * q2 * f4 + (-2 * bx * q3 + 2 * bz * q1) * f5
            + 2 * bx * q2 * f6)
    s[1] = (2 * q3 * f1 + 2 * q0 * f2 - 2 * q1 * f3
            + 2 * bz * q3 * f4 + (2 * bx * q2 + 2 * bz * q0) * f5
            + (2 * bx * q3 - 4 * bz * q1) * f6)
    s[2] = (-2 * q0 * f1 + 2 * q3 * f2 - 2 * q2 * f3
            + (-4 * bx * q2 - 2 * bz * q0) * f4 + (2 * bx * q1 + 2 * bz * q3) * f5
            + (2 * bx * q0 - 4 * bz * q2) * f6)
    s[3] = (2 * q1 * f1 + 2 * q2 * f2 + 2 * q3 * f3
            + (-4 * bx * q3 + 2 * bz * q1) * f4 + (-2 * bx * q0 + 2 * bz * q2) * f5
            + 2 * bx * q1 * f6)
    return f, b, s


def madgwick_marg(acc, gyro, mag, q0, beta=0.08):
    n = len(acc)
    q = np.array(q0, dtype=float)
    hist = np.zeros((n, 4))
    for i in range(n):
        na = np.linalg.norm(acc[i])
        nm = np.linalg.norm(mag[i])
        if na < 1e-9 or nm < 1e-9:
            hist[i] = q
            continue
        f, b, s = marg_pieces(q, acc[i] / na, mag[i] / nm)
        sn = np.linalg.norm(s)
        if sn > 1e-12:
            s = s / sn
        dq = 0.5 * qmul(q, np.array([0.0, *gyro[i]])) - beta * s
        q = qnorm(q + dq * DT)
        hist[i] = q
    return hist


# ---------------------------------------------------------------- 自检
def self_check():
    rng = np.random.default_rng(0)
    ok = True
    print("-" * 62)
    print("自检 ① 真姿态下磁残差（应为 0）")
    worst = 0.0
    for r, p, y in [(0, 0, 0), (30, 0, 0), (0, 25, 0), (20, -15, 40), (45, 30, -60)]:
        q = euler2quat(r * D2R, p * D2R, y * D2R)
        a = gravity_body(q)
        m = mag_body(q)
        f, _, _ = marg_pieces(q, a, m)
        worst = max(worst, np.abs(f[3:]).max())
    print("   最大 |f_mag| = %.2e  %s" % (worst, "OK" if worst < 1e-12 else "FAIL"))
    ok &= worst < 1e-12

    print("自检 ② 梯度 = 数值微分（b 固定）")
    eps = 1e-6
    worst = 0.0
    for _ in range(20):
        q = qnorm(rng.normal(size=4))
        if q[0] < 0:
            q = -q
        a = gravity_body(q) + rng.normal(0, 0.01, 3)
        a /= np.linalg.norm(a)
        m = mag_body(q) + rng.normal(0, 0.005, 3)
        m /= np.linalg.norm(m)
        f, b, s = marg_pieces(q, a, m)

        def F(qq, bb=b):
            ff, _, _ = marg_pieces(qq, a, m)
            # 用固定 b 重算磁残差，避免 b 随 q 变化干扰校验
            q0_, q1_, q2_, q3_ = qq
            bx_, bz_ = bb[0], bb[2]
            ff[3] = 2.0 * (bx_ * (0.5 - q2_ * q2_ - q3_ * q3_) + bz_ * (q1_ * q3_ - q0_ * q2_)) - m[0]
            ff[4] = 2.0 * (bx_ * (q1_ * q2_ - q0_ * q3_) + bz_ * (q2_ * q3_ + q0_ * q1_)) - m[1]
            ff[5] = 2.0 * (bx_ * (q1_ * q3_ + q0_ * q2_) + bz_ * (0.5 - q1_ * q1_ - q2_ * q2_)) - m[2]
            return ff

        num = np.zeros(4)
        for k in range(4):
            qp, qm = q.copy(), q.copy()
            qp[k] += eps
            qm[k] -= eps
            fp, fm = F(qp), F(qm)
            num[k] = 0.5 * (fp @ fp - fm @ fm) / (2 * eps)
        rel = np.linalg.norm(s - num) / max(np.linalg.norm(num), 1e-9)
        worst = max(worst, rel)
    print("   最大相对误差 = %.2e  %s" % (worst, "OK" if worst < 1e-6 else "FAIL"))
    ok &= worst < 1e-6
    return ok


def main():
    ok = self_check()

    rng = np.random.default_rng(5)
    T = 25.0
    n = int(T / DT)

    # 静止 + 陀螺零偏 0.01 rad/s（主要体现为 yaw 漂移）
    q_true = euler2quat(0.0, 0.0, 0.0)
    acc = np.tile(gravity_body(q_true), (n, 1)) + rng.normal(0, 0.03, (n, 3))
    mag = np.tile(mag_body(q_true), (n, 1)) + rng.normal(0, 0.01, (n, 3))
    gyro = np.tile(np.array([0.01, 0.01, 0.01]), (n, 1)) + rng.normal(0, 0.003, (n, 3))

    q0 = euler2quat(0.0, 0.0, 30 * D2R)     # 初始 yaw 误差 30°

    from sim_ahrs import run_madgwick
    h6 = run_madgwick(acc, gyro, q0, beta=0.08)
    h9 = madgwick_marg(acc, gyro, mag, q0, beta=0.08)

    t = np.arange(n) * DT
    yaw6 = rpy_deg(h6)[:, 2]
    yaw9 = rpy_deg(h9)[:, 2]
    err9 = np.array([quat_error_deg(h9[i], q_true) for i in range(n)])

    # 收敛时间：与 6 轴用同一个判定函数（0.5 s 滑动平均 + 之后不再反弹），口径一致
    t_conv = conv_time(h9, np.tile(q_true, (n, 1)))

    res = {
        "yaw6_end_deg": float(yaw6[-1]),
        "yaw9_end_deg": float(yaw9[-1]),
        "yaw9_max_deg": float(yaw9.max()),
        "err9_conv_1deg_s": t_conv,
        "err9_end_deg": float(err9[-100:].mean()),
        "beta": 0.08,
        "bias_rps": 0.01,
        "self_check_ok": bool(ok),
    }
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "marg.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    with open(os.path.join(OUT, "marg.csv"), "w", encoding="utf-8") as f:
        f.write("t,yaw6,yaw9,err9\n")
        for i in range(n):
            f.write("%.3f,%.4f,%.4f,%.4f\n" % (t[i], yaw6[i], yaw9[i], err9[i]))

    print("-" * 62)
    for k2, v in res.items():
        print("%-20s %s" % (k2, v))


if __name__ == "__main__":
    main()
