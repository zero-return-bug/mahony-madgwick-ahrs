#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""独立审计：9 轴 MARG 版本的残差是否等价于 R^T b - m，梯度是否精确，
   以及从任意初始姿态出发是否收敛到真值（不只是 yaw 偏 30° 的特例）。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim_ahrs import (DT, D2R, R2D, euler2quat, qnorm, quat2R,
                      gravity_body, quat_error_deg, rpy_deg)

INC = 60.0 * D2R
M_EARTH = np.array([0.0, np.cos(INC), -np.sin(INC)])


def mag_body(q):
    return quat2R(q).T @ M_EARTH


def resid_article(q, a, m):
    """文中 §4.6 的 6 维残差"""
    q0, q1, q2, q3 = q
    a = a / np.linalg.norm(a)
    m = m / np.linalg.norm(m)
    hx = 2 * (m[0] * (0.5 - q2 * q2 - q3 * q3) + m[1] * (q1 * q2 - q0 * q3)
              + m[2] * (q1 * q3 + q0 * q2))
    hy = 2 * (m[0] * (q1 * q2 + q0 * q3) + m[1] * (0.5 - q1 * q1 - q3 * q3)
              + m[2] * (q2 * q3 - q0 * q1))
    hz = 2 * (m[0] * (q1 * q3 - q0 * q2) + m[1] * (q2 * q3 + q0 * q1)
              + m[2] * (0.5 - q1 * q1 - q2 * q2))
    by, bz = np.hypot(hx, hy), hz
    f1 = 2 * (q1 * q3 - q0 * q2) - a[0]
    f2 = 2 * (q0 * q1 + q2 * q3) - a[1]
    f3 = q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3 - a[2]
    f4 = 2 * (by * (q1 * q2 - q0 * q3) + bz * (q1 * q3 + q0 * q2)) - m[0]
    f5 = 2 * (by * (0.5 - q1 * q1 - q3 * q3) + bz * (q2 * q3 - q0 * q1)) - m[1]
    f6 = 2 * (by * (q2 * q3 + q0 * q1) + bz * (0.5 - q1 * q1 - q2 * q2)) - m[2]
    return np.array([f1, f2, f3, f4, f5, f6]), (by, bz)


def resid_true(q, a, m):
    """按 R^T b - m 直接构造（b 用同样的 [0,by,bz] 约定）"""
    a = a / np.linalg.norm(a)
    m = m / np.linalg.norm(m)
    q0, q1, q2, q3 = q
    h = quat2R(q) @ m
    b = np.array([0.0, np.hypot(h[0], h[1]), h[2]])
    exp_a = quat2R(q).T @ np.array([0.0, 0.0, 1.0])
    exp_m = quat2R(q).T @ b
    return np.concatenate([exp_a - a, exp_m - m]), b


def grad_article(f, by, bz, q):
    q0, q1, q2, q3 = q
    f1, f2, f3, f4, f5, f6 = f
    s0 = (-2 * q2 * f1 + 2 * q1 * f2 + 2 * q0 * f3
          - 2 * by * q3 * f4 + 2 * bz * q2 * f4
          - 2 * bz * q1 * f5 + 2 * by * q1 * f6)
    s1 = (2 * q3 * f1 + 2 * q0 * f2 - 2 * q1 * f3
          + 2 * by * q2 * f4 + 2 * bz * q3 * f4
          - 4 * by * q1 * f5 - 2 * bz * q0 * f5
          + 2 * by * q0 * f6 - 4 * bz * q1 * f6)
    s2 = (-2 * q0 * f1 + 2 * q3 * f2 - 2 * q2 * f3
          + 2 * by * q1 * f4 + 2 * bz * q0 * f4
          + 2 * bz * q3 * f5
          + 2 * by * q3 * f6 - 4 * bz * q2 * f6)
    s3 = (2 * q1 * f1 + 2 * q2 * f2 + 2 * q3 * f3
          - 2 * by * q0 * f4 + 2 * bz * q1 * f4
          - 4 * by * q3 * f5 - 2 * bz * q2 * f5
          + 2 * by * q2 * f6)
    return np.array([s0, s1, s2, s3])


# ---------------------------------------------------------------- 1) 残差等价性
print("=" * 68)
print("1) 文中 f4..f6 与 R^T b - m 是否等价（随机 8 个姿态）")
rng = np.random.default_rng(0)
worst = 0.0
for _ in range(8):
    q = qnorm(rng.normal(size=4))
    if q[0] < 0:
        q = -q
    a = gravity_body(q) + rng.normal(0, 0.01, 3)
    m = mag_body(q) + rng.normal(0, 0.005, 3)
    f_art, (by, bz) = resid_article(q, a, m)
    f_true, b = resid_true(q, a, m)
    d = np.abs(f_art - f_true).max()
    worst = max(worst, d)
    print("   q0=%.3f  max|Δf|=%.4f   b_art=(0,%.3f,%.3f) b_true=(%.3f,%.3f,%.3f)"
          % (q[0], d, by, bz, b[0], b[1], b[2]))
print("   最大偏差 %.4f  ->  %s" % (worst, "等价" if worst < 1e-6 else "不等价"))

# ---------------------------------------------------------------- 2) 梯度精确性
print("=" * 68)
print("2) 文中 s0..s3 是否等于 d/dq [0.5*||f||^2]（数值微分校验）")
eps = 1e-6
q = qnorm(euler2quat(20 * D2R, -15 * D2R, 40 * D2R))
a = gravity_body(q) + rng.normal(0, 0.01, 3)
m = mag_body(q) + rng.normal(0, 0.005, 3)
f, (by, bz) = resid_article(q, a, m)
s_ana = grad_article(f, by, bz, q)
s_num = np.zeros(4)
for k in range(4):
    qp, qm = q.copy(), q.copy()
    qp[k] += eps
    qm[k] -= eps
    fp, _ = resid_article(qp, a, m)
    fm, _ = resid_article(qm, a, m)
    s_num[k] = 0.5 * (fp @ fp - fm @ fm) / (2 * eps)
print("   解析:", np.array2string(s_ana, precision=3))
print("   数值:", np.array2string(s_num, precision=3))
print("   相对误差: %.2e" % (np.linalg.norm(s_ana - s_num) / max(np.linalg.norm(s_num), 1e-9)))

# ---------------------------------------------------------------- 3) 定点检验
print("=" * 68)
print("3) 真姿态处残差是否为零（这是定点条件）")
mx = 0.0
for th_r, th_p, th_y in [(0, 0, 0), (30, 0, 0), (0, 25, 0), (20, -15, 40),
                         (45, 30, -60), (10, 60, 120)]:
    q = euler2quat(th_r * D2R, th_p * D2R, th_y * D2R)
    a, m = gravity_body(q), mag_body(q)
    f_art, _ = resid_article(q, a, m)
    f_true, _ = resid_true(q, a, m)
    mx = max(mx, np.abs(f_art[3:]).max())
    print("   rpy=(%4d,%4d,%4d)  |f_mag_article|=%.4f  |f_mag_true|=%.4f"
          % (th_r, th_p, th_y, np.abs(f_art[3:]).max(), np.abs(f_true[3:]).max()))
print("   文中版本在真姿态处的最大磁残差: %.4f  %s"
      % (mx, "（为 0，定点正确）" if mx < 1e-6 else "（非零，定点有偏）"))
