#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mahony / Madgwick 姿态解算仿真
--------------------------------
生成带噪声的 6 轴 IMU 数据，跑两种算法，输出指标与曲线数据到 out/。
公式与文中 C 代码逐行对应，保证图、表、代码三者同源。

用法： python sim_ahrs.py
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)

DT = 0.01
D2R = np.pi / 180.0
R2D = 180.0 / np.pi
G = 1.0  # 加速度以 g 为单位


# ---------------------------------------------------------------- 四元数工具
def qmul(p, q):
    p0, p1, p2, p3 = p
    q0, q1, q2, q3 = q
    return np.array([
        p0 * q0 - p1 * q1 - p2 * q2 - p3 * q3,
        p0 * q1 + p1 * q0 + p2 * q3 - p3 * q2,
        p0 * q2 - p1 * q3 + p2 * q0 + p3 * q1,
        p0 * q3 + p1 * q2 - p2 * q1 + p3 * q0,
    ])


def qconj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def qnorm(q):
    n = np.linalg.norm(q)
    return q / n if n > 1e-12 else np.array([1.0, 0.0, 0.0, 0.0])


def euler2quat(roll, pitch, yaw):
    """ZYX 顺序，body->world，输入为弧度"""
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
    qz = np.array([cy, 0.0, 0.0, sy])
    qy = np.array([cp, 0.0, sp, 0.0])
    qx = np.array([cr, sr, 0.0, 0.0])
    return qnorm(qmul(qmul(qz, qy), qx))


def quat2euler(q):
    """返回 roll/pitch/yaw（弧度），与文中 getRPY() 完全一致"""
    q0, q1, q2, q3 = q
    roll = np.arctan2(2.0 * (q0 * q1 + q2 * q3), 1.0 - 2.0 * (q1 * q1 + q2 * q2))
    pitch = np.arcsin(np.clip(2.0 * (q0 * q2 - q3 * q1), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (q0 * q3 + q1 * q2), 1.0 - 2.0 * (q2 * q2 + q3 * q3))
    return roll, pitch, yaw


def quat2R(q):
    """body->world 旋转矩阵"""
    q0, q1, q2, q3 = q
    return np.array([
        [1 - 2 * (q2 * q2 + q3 * q3), 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)],
        [2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1 * q1 + q3 * q3), 2 * (q2 * q3 - q0 * q1)],
        [2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), 1 - 2 * (q1 * q1 + q2 * q2)],
    ])


def gravity_body(q):
    """估计重力方向 v —— 就是 R 的第三行"""
    return quat2R(q)[2, :]


def quat_error_deg(q_est, q_true):
    """两个姿态之间的夹角（度）"""
    d = qmul(qconj(q_true), q_est)
    w = np.clip(abs(d[0]), -1.0, 1.0)
    return 2.0 * np.arccos(w) * R2D


# ------------------------------------------------------------ 参考轨迹与传感器
def ref_euler(t):
    """三轴同时摆动的正弦轨迹（弧度）"""
    roll = 30.0 * D2R * np.sin(2 * np.pi * 0.15 * t)
    pitch = 25.0 * D2R * np.sin(2 * np.pi * 0.17 * t + 0.5)
    yaw = 40.0 * D2R * np.sin(2 * np.pi * 0.11 * t)
    return roll, pitch, yaw


def ref_omega(t, h=1e-4):
    """由四元数微分反推机体系角速度： q_dot = 0.5 q ⊗ [0,ω]"""
    qa = euler2quat(*ref_euler(t - h))
    qb = euler2quat(*ref_euler(t + h))
    qc = euler2quat(*ref_euler(t))
    dq = (qb - qa) / (2 * h)
    omega = 2.0 * qmul(qconj(qc), dq)  # [0, wx, wy, wz]
    return omega[1:]


def gen_imu(t_end, gyro_bias, gyro_noise, acc_noise, seed=7,
            shock_t=None, shock_g=0.0, shock_dur=1.0, static=False):
    n = int(round(t_end / DT))
    t = np.arange(n) * DT
    rng = np.random.default_rng(seed)

    q_true = np.zeros((n, 4))
    omega = np.zeros((n, 3))
    acc = np.zeros((n, 3))
    gyro = np.zeros((n, 3))

    bias = np.asarray(gyro_bias, dtype=float)

    for i, ti in enumerate(t):
        if static:
            q = euler2quat(0.0, 0.0, 0.0)
            w = np.zeros(3)
        else:
            q = euler2quat(*ref_euler(ti))
            w = ref_omega(ti)
        q_true[i] = q

        a_body = gravity_body(q).copy()
        if shock_t is not None and shock_t <= ti < shock_t + shock_dur:
            s = 0.5 - 0.5 * np.cos(2 * np.pi * (ti - shock_t) / shock_dur)  # 升余弦脉冲
            a_world = np.array([shock_g * s, 0.0, 0.0])
            a_body = a_body + quat2R(q).T @ a_world

        acc[i] = a_body + rng.normal(0.0, acc_noise, 3)
        gyro[i] = w + bias + rng.normal(0.0, gyro_noise, 3)
        omega[i] = w

    return t, q_true, omega, acc, gyro


# ---------------------------------------------------------------- 两种算法
def run_mahony(acc, gyro, q0, Kp=2.0, Ki=0.05, sign=+1, adaptive=False, adapt_th=0.05):
    n = len(acc)
    q = np.array(q0, dtype=float)
    e_int = np.zeros(3)
    hist = np.zeros((n, 4))
    for i in range(n):
        a = acc[i]
        na = np.linalg.norm(a)
        if na < 1e-6:
            hist[i] = q
            continue
        a = a / na
        v = gravity_body(q)
        e = sign * np.cross(a, v)

        w = 1.0
        if adaptive:
            w = max(0.0, 1.0 - abs(na - 1.0) / adapt_th) if abs(na - 1.0) < adapt_th else 0.0

        if w > 0.0:
            e_int = e_int + Ki * e * DT
        g = gyro[i] + w * Kp * e + e_int

        dq = 0.5 * qmul(q, np.array([0.0, g[0], g[1], g[2]]))
        q = qnorm(q + dq * DT)
        hist[i] = q
    return hist


def run_madgwick(acc, gyro, q0, beta=0.08, sign=-1):
    n = len(acc)
    q = np.array(q0, dtype=float)
    hist = np.zeros((n, 4))
    for i in range(n):
        a = acc[i]
        na = np.linalg.norm(a)
        if na < 1e-6:
            hist[i] = q
            continue
        a = a / na
        q0_, q1_, q2_, q3_ = q
        f1 = 2.0 * (q1_ * q3_ - q0_ * q2_) - a[0]
        f2 = 2.0 * (q0_ * q1_ + q2_ * q3_) - a[1]
        f3 = q0_ * q0_ - q1_ * q1_ - q2_ * q2_ + q3_ * q3_ - a[2]

        s0 = -2.0 * q2_ * f1 + 2.0 * q1_ * f2 + 2.0 * q0_ * f3
        s1 = 2.0 * q3_ * f1 + 2.0 * q0_ * f2 - 2.0 * q1_ * f3
        s2 = -2.0 * q0_ * f1 + 2.0 * q3_ * f2 - 2.0 * q2_ * f3
        s3 = 2.0 * q1_ * f1 + 2.0 * q2_ * f2 + 2.0 * q3_ * f3

        sn = np.sqrt(s0 * s0 + s1 * s1 + s2 * s2 + s3 * s3)
        if sn > 1e-9:
            s0, s1, s2, s3 = s0 / sn, s1 / sn, s2 / sn, s3 / sn

        g = gyro[i]
        dq = 0.5 * qmul(q, np.array([0.0, g[0], g[1], g[2]])) \
            + sign * beta * np.array([s0, s1, s2, s3])
        q = qnorm(q + dq * DT)
        hist[i] = q
    return hist


def run_gyro_only(gyro, q0):
    n = len(gyro)
    q = np.array(q0, dtype=float)
    hist = np.zeros((n, 4))
    for i in range(n):
        dq = 0.5 * qmul(q, np.array([0.0, *gyro[i]]))
        q = qnorm(q + dq * DT)
        hist[i] = q
    return hist


# ---------------------------------------------------------------- 指标
def rpy_deg(hist):
    return np.array([[x * R2D for x in quat2euler(q)] for q in hist])


def ang_err_series(hist, q_true):
    return np.array([quat_error_deg(hist[i], q_true[i]) for i in range(len(hist))])


def tilt_err_series(hist, q_true):
    """倾斜误差：估计重力方向与真实重力方向的夹角（与 yaw 无关）"""
    out = np.zeros(len(hist))
    for i in range(len(hist)):
        c = float(np.clip(np.dot(gravity_body(hist[i]), gravity_body(q_true[i])), -1.0, 1.0))
        out[i] = np.arccos(c) * R2D
    return out


def rmse_rp(hist, q_true):
    rpy_e = rpy_deg(hist)
    rpy_t = rpy_deg(q_true)
    d = rpy_e[:, :2] - rpy_t[:, :2]
    d = (d + 180.0) % 360.0 - 180.0
    return np.sqrt(np.mean(d ** 2, axis=0))


def rmse_rp_win(hist, q_true, t, t0=5.0):
    """稳态窗口（跳过初始收敛段）的 roll/pitch RMSE"""
    m = t >= t0
    return rmse_rp(hist[m], q_true[m])


def conv_time(hist, q_true, thr=1.0):
    """误差（0.5 s 滑动平均，抑制噪声抖动）首次稳定进入 thr 以内的时间"""
    err = ang_err_series(hist, q_true)
    k = max(1, int(0.5 / DT))
    ker = np.ones(k) / k
    sm = np.convolve(err, ker, mode="same")
    sm[: k] = err[: k]
    idx = np.where(sm < thr)[0]
    if len(idx) == 0:
        return np.nan
    for i in idx:
        if np.all(sm[i:] < thr * 1.6):  # 之后不再反弹出阈值
            return float(i * DT)
    return float(idx[0] * DT)


def main():
    M = {}

    # ---------- 实验 1：轨迹跟踪（10° 初始误差） ----------
    t, q_true, omega, acc, gyro = gen_imu(
        20.0, gyro_bias=[0.01, 0.01, 0.01], gyro_noise=0.003, acc_noise=0.06, seed=11)
    q0 = euler2quat(8 * D2R, 6 * D2R, 0.0)
    qm = run_mahony(acc, gyro, q0, Kp=2.0, Ki=0.05)
    qd = run_madgwick(acc, gyro, q0, beta=0.08)

    rm = rmse_rp_win(qm, q_true, t)
    rd = rmse_rp_win(qd, q_true, t)
    M["track"] = {
        "mahony_rmse_roll": float(rm[0]), "mahony_rmse_pitch": float(rm[1]),
        "madgwick_rmse_roll": float(rd[0]), "madgwick_rmse_pitch": float(rd[1]),
        "initial_error_deg": float(quat_error_deg(q0, q_true[0])),
        "window": "5-20 s",
        "acc_noise_g": 0.06,
        "gyro_noise_rps": 0.003,
    }

    rpy_t, rpy_m, rpy_d = rpy_deg(q_true), rpy_deg(qm), rpy_deg(qd)
    with open(os.path.join(OUT, "track.csv"), "w", encoding="utf-8") as f:
        f.write("t,roll_t,pitch_t,yaw_t,roll_m,pitch_m,yaw_m,roll_d,pitch_d,yaw_d\n")
        for i in range(len(t)):
            f.write("%.3f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n" % (
                t[i], *rpy_t[i], *rpy_m[i], *rpy_d[i]))

    # ---------- 实验 2：30° 初始误差收敛（静置） ----------
    t2, q2, _, acc2, gyro2 = gen_imu(
        12.0, gyro_bias=[0.0, 0.0, 0.0], gyro_noise=0.003, acc_noise=0.06,
        seed=23, static=True)
    axis = np.array([0.8, 0.6, 0.0])
    axis = axis / np.linalg.norm(axis)
    th = 30.0 * D2R
    q0c = qmul(np.array([np.cos(th / 2), *(np.sin(th / 2) * axis)]), q2[0])
    qm2 = run_mahony(acc2, gyro2, q0c, Kp=2.0, Ki=0.05)
    qd2 = run_madgwick(acc2, gyro2, q0c, beta=0.08)
    em = ang_err_series(qm2, q2)
    ed = ang_err_series(qd2, q2)
    M["conv"] = {
        "mahony_t_1deg": float(conv_time(qm2, q2)),
        "madgwick_t_1deg": float(conv_time(qd2, q2)),
        "mahony_t_5deg": float(conv_time(qm2, q2, 5.0)),
        "madgwick_t_5deg": float(conv_time(qd2, q2, 5.0)),
    }
    with open(os.path.join(OUT, "conv.csv"), "w", encoding="utf-8") as f:
        f.write("t,err_mahony,err_madgwick\n")
        for i in range(len(t2)):
            f.write("%.3f,%.4f,%.4f\n" % (t2[i], em[i], ed[i]))

    # ---------- 实验 3：0.4 g 水平加速度冲击 ----------
    t3, q3, _, acc3, gyro3 = gen_imu(
        14.0, gyro_bias=[0.01, 0.01, 0.01], gyro_noise=0.003, acc_noise=0.06,
        seed=31, shock_t=8.0, shock_g=0.4, shock_dur=1.0)
    qm3 = run_mahony(acc3, gyro3, q3[0], Kp=2.0, Ki=0.05)
    qd3 = run_madgwick(acc3, gyro3, q3[0], beta=0.08)
    qa3 = run_mahony(acc3, gyro3, q3[0], Kp=2.0, Ki=0.05, adaptive=True)
    e3m = tilt_err_series(qm3, q3)
    e3d = tilt_err_series(qd3, q3)
    e3a = tilt_err_series(qa3, q3)
    win = (t3 >= 8.0) & (t3 <= 9.6)
    res = (t3 >= 10.0) & (t3 <= 13.0)
    M["impact"] = {
        "mahony_peak": float(e3m[win].max()),
        "madgwick_peak": float(e3d[win].max()),
        "adaptive_peak": float(e3a[win].max()),
        "mahony_residual": float(e3m[res].mean()),
        "madgwick_residual": float(e3d[res].mean()),
        "adaptive_residual": float(e3a[res].mean()),
        "steady_tilt_deg": float(np.degrees(np.arctan(0.4))),
    }
    with open(os.path.join(OUT, "impact.csv"), "w", encoding="utf-8") as f:
        f.write("t,acc_norm,err_mahony,err_madgwick,err_adaptive\n")
        for i in range(len(t3)):
            f.write("%.3f,%.4f,%.4f,%.4f,%.4f\n" % (
                t3[i], np.linalg.norm(acc3[i]), e3m[i], e3d[i], e3a[i]))

    # ---------- 实验 4：60 s yaw 漂移（零偏 0.57°/s） ----------
    bias_yaw = 0.57 * D2R
    t4, q4, _, acc4, gyro4 = gen_imu(
        60.0, gyro_bias=[0.0, 0.0, bias_yaw], gyro_noise=0.003, acc_noise=0.06,
        seed=41, static=True)
    qm4 = run_mahony(acc4, gyro4, q4[0], Kp=2.0, Ki=0.05)
    qd4 = run_madgwick(acc4, gyro4, q4[0], beta=0.08)
    qg4 = run_gyro_only(gyro4, q4[0])
    g_m = rpy_deg(qm4)[:, 2]
    g_d = rpy_deg(qd4)[:, 2]
    g_g = rpy_deg(qg4)[:, 2]
    M["yaw"] = {
        "bias_dps": 0.57,
        "mahony_end": float(g_m[-1]),
        "madgwick_end": float(g_d[-1]),
        "gyro_end": float(g_g[-1]),
    }
    with open(os.path.join(OUT, "yaw.csv"), "w", encoding="utf-8") as f:
        f.write("t,gyro_only,mahony,madgwick\n")
        for i in range(len(t4)):
            f.write("%.3f,%.4f,%.4f,%.4f\n" % (t4[i], g_g[i], g_m[i], g_d[i]))

    # ---------- 实验 5：大零偏（0.05 rad/s） ----------
    t5, q5, _, acc5, gyro5 = gen_imu(
        20.0, gyro_bias=[0.05, 0.05, 0.05], gyro_noise=0.003, acc_noise=0.06, seed=51)
    qm5 = run_mahony(acc5, gyro5, q5[0], Kp=2.0, Ki=0.05)
    qd5 = run_madgwick(acc5, gyro5, q5[0], beta=0.08)
    r5m = rmse_rp_win(qm5, q5, t5)
    r5d = rmse_rp_win(qd5, q5, t5)
    M["bigbias"] = {
        "mahony_rmse": float(np.hypot(*r5m)),
        "madgwick_rmse": float(np.hypot(*r5d)),
    }

    # ---------- 实验 6：参数扫描 ----------
    kp_list = [0.5, 1.0, 2.0, 4.0]
    ki_list = [0.01, 0.02, 0.05, 0.10]
    sweep_kp = []
    for kp, ki in zip(kp_list, ki_list):
        h = run_mahony(acc, gyro, q0, Kp=kp, Ki=ki)
        r = rmse_rp_win(h, q_true, t)
        sweep_kp.append({"kp": kp, "ki": ki, "rmse_roll": float(r[0]), "rmse_pitch": float(r[1])})

    beta_list = [0.02, 0.05, 0.08, 0.15, 0.30]
    sweep_beta = []
    for b in beta_list:
        h = run_madgwick(acc, gyro, q0, beta=b)
        r = rmse_rp_win(h, q_true, t)
        sweep_beta.append({"beta": b, "rmse_roll": float(r[0]), "rmse_pitch": float(r[1])})
    M["sweep_kp"] = sweep_kp
    M["sweep_beta"] = sweep_beta

    # ---------- 实验 7：符号写反 ----------
    t7, q7, _, acc7, gyro7 = gen_imu(
        6.0, gyro_bias=[0.01, 0.01, 0.01], gyro_noise=0.003, acc_noise=0.06, seed=61)
    q7ok = run_mahony(acc7, gyro7, euler2quat(8 * D2R, 6 * D2R, 0), Kp=2.0, Ki=0.05, sign=+1)
    q7bad = run_mahony(acc7, gyro7, euler2quat(8 * D2R, 6 * D2R, 0), Kp=2.0, Ki=0.05, sign=-1)
    q8ok = run_madgwick(acc7, gyro7, euler2quat(8 * D2R, 6 * D2R, 0), beta=0.08, sign=-1)
    q8bad = run_madgwick(acc7, gyro7, euler2quat(8 * D2R, 6 * D2R, 0), beta=0.08, sign=+1)
    e7ok = ang_err_series(q7ok, q7)
    e7bad = ang_err_series(q7bad, q7)
    e8ok = ang_err_series(q8ok, q7)
    e8bad = ang_err_series(q8bad, q7)
    M["sign"] = {
        "mahony_ok_rmse": float(np.hypot(*rmse_rp(q7ok, q7))),
        "mahony_bad_rmse": float(np.hypot(*rmse_rp(q7bad, q7))),
        "madgwick_ok_rmse": float(np.hypot(*rmse_rp(q8ok, q7))),
        "madgwick_bad_rmse": float(np.hypot(*rmse_rp(q8bad, q7))),
    }
    with open(os.path.join(OUT, "sign.csv"), "w", encoding="utf-8") as f:
        f.write("t,mahony_ok,mahony_rev,madgwick_ok,madgwick_rev\n")
        for i in range(len(t7)):
            f.write("%.3f,%.4f,%.4f,%.4f,%.4f\n" % (
                t7[i], e7ok[i], e7bad[i], e8ok[i], e8bad[i]))

    with open(os.path.join(OUT, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(M, f, ensure_ascii=False, indent=2)

    # ---------- 控制台摘要 ----------
    def p(k, v):
        print("%-28s %s" % (k, v))

    p("RMSE roll  Mahony/Madgwick", "%.2f° / %.2f°" % (
        M["track"]["mahony_rmse_roll"], M["track"]["madgwick_rmse_roll"]))
    p("RMSE pitch Mahony/Madgwick", "%.2f° / %.2f°" % (
        M["track"]["mahony_rmse_pitch"], M["track"]["madgwick_rmse_pitch"]))
    p("30° 收敛到 1°  Mahony/Madgwick", "%.2f s / %.2f s" % (
        M["conv"]["mahony_t_1deg"], M["conv"]["madgwick_t_1deg"]))
    p("0.4g 冲击峰值 Mahony/Madgwick", "%.1f° / %.1f°" % (
        M["impact"]["mahony_peak"], M["impact"]["madgwick_peak"]))
    p("冲击后残余  Mahony/Madgwick", "%.1f° / %.1f°" % (
        M["impact"]["mahony_residual"], M["impact"]["madgwick_residual"]))
    p("自适应增益 峰值/残余", "%.1f° / %.1f°" % (
        M["impact"]["adaptive_peak"], M["impact"]["adaptive_residual"]))
    p("60 s yaw  Mahony/Madgwick/纯积分", "%.1f° / %.1f° / %.1f°" % (
        M["yaw"]["mahony_end"], M["yaw"]["madgwick_end"], M["yaw"]["gyro_end"]))
    p("大零偏 RMSE Mahony/Madgwick", "%.2f° / %.2f°" % (
        M["bigbias"]["mahony_rmse"], M["bigbias"]["madgwick_rmse"]))
    p("符号写反 RMSE M/R", "%.1f° / %.1f°" % (
        M["sign"]["mahony_bad_rmse"], M["sign"]["madgwick_bad_rmse"]))
    print("\n输出目录:", OUT)


if __name__ == "__main__":
    main()
