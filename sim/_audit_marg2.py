#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对比三种 MARG 写法：
   A_art : 文中 §4.6 现版本（b=[0,by,bz]，north=+y）
   A_fix : 同约定但残差改成真正的 R^T b - m
   B_std : 官方 Madgwick 约定（b=[bx,0,bz]，north=+x）
检验：真姿态残差是否为 0、梯度是否精确、随机初始姿态能否收敛。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim_ahrs import (DT, D2R, euler2quat, qnorm, qmul, quat2R,
                      gravity_body, quat_error_deg, rpy_deg)

INC = 60.0 * D2R
M_Y = np.array([0.0, np.cos(INC), -np.sin(INC)])   # north = +y
M_X = np.array([np.cos(INC), 0.0, -np.sin(INC)])   # north = +x


def b_from_m(q, m, north):
    h = quat2R(q) @ m
    if north == "y":
        return np.array([0.0, np.hypot(h[0], h[1]), h[2]])
    return np.array([np.hypot(h[0], h[1]), 0.0, h[2]])


def residual(q, a, m, b, variant):
    q0, q1, q2, q3 = q
    f = np.zeros(6)
    f[0] = 2 * (q1 * q3 - q0 * q2) - a[0]
    f[1] = 2 * (q0 * q1 + q2 * q3) - a[1]
    f[2] = q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3 - a[2]
    if variant == "A_art":
        by, bz = b[1], b[2]
        f[3] = 2 * (by * (q1 * q2 - q0 * q3) + bz * (q1 * q3 + q0 * q2)) - m[0]
        f[4] = 2 * (by * (0.5 - q1 * q1 - q3 * q3) + bz * (q2 * q3 - q0 * q1)) - m[1]
        f[5] = 2 * (by * (q2 * q3 + q0 * q1) + bz * (0.5 - q1 * q1 - q2 * q2)) - m[2]
    else:
        e = quat2R(q).T @ b
        f[3:] = e - m
    return f


def grad(q, b, f, variant):
    q0, q1, q2, q3 = q
    f1, f2, f3, f4, f5, f6 = f
    s = np.zeros(4)
    s[0] = -2 * q2 * f1 + 2 * q1 * f2 + 2 * q0 * f3
    s[1] = 2 * q3 * f1 + 2 * q0 * f2 - 2 * q1 * f3
    s[2] = -2 * q0 * f1 + 2 * q3 * f2 - 2 * q2 * f3
    s[3] = 2 * q1 * f1 + 2 * q2 * f2 + 2 * q3 * f3

    if variant == "A_art":
        by, bz = b[1], b[2]
        s[0] += -2 * by * q3 * f4 + 2 * bz * q2 * f4 - 2 * bz * q1 * f5 + 2 * by * q1 * f6
        s[1] += (2 * by * q2 * f4 + 2 * bz * q3 * f4 - 4 * by * q1 * f5
                 - 2 * bz * q0 * f5 + 2 * by * q0 * f6 - 4 * bz * q1 * f6)
        s[2] += (2 * by * q1 * f4 + 2 * bz * q0 * f4 + 2 * bz * q3 * f5
                 + 2 * by * q3 * f6 - 4 * bz * q2 * f6)
        s[3] += (-2 * by * q0 * f4 + 2 * bz * q1 * f4 - 4 * by * q3 * f5
                 - 2 * bz * q2 * f5 + 2 * by * q2 * f6)
    else:
        bx, bz = b[0], b[2]
        s[0] += (-2 * bz * q2) * f4 + (-2 * bx * q3 + 2 * bz * q1) * f5 + (2 * bx * q2) * f6
        s[1] += (2 * bz * q3) * f4 + (2 * bx * q2 + 2 * bz * q0) * f5 + (2 * bx * q3 - 4 * bz * q1) * f6
        s[2] += (-4 * bx * q2 - 2 * bz * q0) * f4 + (2 * bx * q1 + 2 * bz * q3) * f5 + (2 * bx * q0 - 4 * bz * q2) * f6
        s[3] += (-4 * bx * q3 + 2 * bz * q1) * f4 + (-2 * bx * q0 + 2 * bz * q2) * f5 + (2 * bx * q1) * f6
    return s


VARIANTS = [("A_art", "y"), ("A_fix", "y"), ("B_std", "x")]

print("=" * 70)
print("① 真姿态处磁残差（应为 0）")
for name, north in VARIANTS:
    w = 0.0
    for r, p, y in [(0, 0, 0), (30, 0, 0), (0, 25, 0), (20, -15, 40), (45, 30, -60)]:
        q = euler2quat(r * D2R, p * D2R, y * D2R)
        mw = M_Y if north == "y" else M_X
        m = quat2R(q).T @ mw
        m = m / np.linalg.norm(m)
        b = b_from_m(q, m, north)
        a = gravity_body(q)
        f = residual(q, a, m, b, name)
        w = max(w, np.abs(f[3:]).max())
    print("   %-6s 最大 |f_mag| = %.6f" % (name, w))

print("=" * 70)
print("② 梯度精确性（b 固定，与算法实际做法一致）")
rng = np.random.default_rng(1)
for name, north in VARIANTS:
    mw = M_Y if north == "y" else M_X
    rel_w = 0.0
    for _ in range(5):
        q = qnorm(rng.normal(size=4))
        if q[0] < 0:
            q = -q
        a = gravity_body(q) + rng.normal(0, 0.01, 3)
        a = a / np.linalg.norm(a)
        m = quat2R(q).T @ mw + rng.normal(0, 0.005, 3)
        m = m / np.linalg.norm(m)
        b = b_from_m(q, m, north)
        f = residual(q, a, m, b, name)
        s_ana = grad(q, b, f, name)
        s_num = np.zeros(4)
        eps = 1e-6
        for k in range(4):
            qp, qm = q.copy(), q.copy()
            qp[k] += eps
            qm[k] -= eps
            fp = residual(qp, a, m, b, name)
            fm = residual(qm, a, m, b, name)
            s_num[k] = 0.5 * (fp @ fp - fm @ fm) / (2 * eps)
        rel_w = max(rel_w, np.linalg.norm(s_ana - s_num) / max(np.linalg.norm(s_num), 1e-9))
    print("   %-6s 最大相对误差 = %.2e" % (name, rel_w))


def run(name, north, acc, gyro, mag, q0, beta=0.08):
    n = len(acc)
    q = np.array(q0, float)
    hist = np.zeros((n, 4))
    for i in range(n):
        a = acc[i] / np.linalg.norm(acc[i])
        m = mag[i] / np.linalg.norm(mag[i])
        b = b_from_m(q, m, north)
        f = residual(q, a, m, b, name)
        s = grad(q, b, f, name)
        sn = np.linalg.norm(s)
        if sn > 1e-12:
            s = s / sn
        dq = 0.5 * qmul(q, np.array([0.0, *gyro[i]])) - beta * s
        q = qnorm(q + dq * DT)
        hist[i] = q
    return hist


print("=" * 70)
print("③ 从「倾斜 + 偏航都错」的初始姿态收敛（静态，40 s）")
n = int(40.0 / DT)
for name, north in VARIANTS:
    mw = M_Y if north == "y" else M_X
    q_true = euler2quat(15 * D2R, -10 * D2R, 25 * D2R)
    acc = np.tile(gravity_body(q_true), (n, 1)) + rng.normal(0, 0.02, (n, 3))
    mag = np.tile(quat2R(q_true).T @ mw, (n, 1)) + rng.normal(0, 0.005, (n, 3))
    gyro = np.tile([0.0, 0.0, 0.0], (n, 1)) + rng.normal(0, 0.002, (n, 3))
    cases = [("倾斜也错 20/15", qmul(euler2quat(20 * D2R, 15 * D2R, -30 * D2R), q_true)),
             ("仅 yaw 偏 30°", euler2quat(15 * D2R, -10 * D2R, 55 * D2R))]
    for label, q0x in cases:
        h = run(name, north, acc, gyro, mag, q0x)
        e = np.array([quat_error_deg(h[i], q_true) for i in range(n)])
        yr = rpy_deg(h[-1500:])[:, 2].mean() - 25.0
        print("   %-6s %-14s 末段姿态误差 %5.2f°   yaw 偏 %6.2f°"
              % (name, label, e[-1500:].mean(), yr))
